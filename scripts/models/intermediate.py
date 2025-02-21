import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchvision.models import ResNet50_Weights

from constants import ROOT, TRAIN_CSV, FLICKR8K_IMG_DIR, PAD
from runner_config import TRANSFORM, NUM_WORKERS, SHUFFLE, PIN_MEMORY, VOCAB_THRESHOLD, EMBED_SIZE, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, \
	FREEZE_ENCODER, ENCODER_LR, DECODER_LR
from scripts.dataset.flickr_dataloader import FlickrDataLoader
from scripts.dataset.flickr_dataset import FlickrDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.models.image_captioning import ImageCaptioning


class Encoder(nn.Module):
	"""
	Encoder class that uses a pretrained ResNet-50 model to extract features from images.
	"""

	def __init__(self, embed_size: int, freeze: bool, dropout) -> None:
		"""
		Constructor for the EncoderResnet class

		:param dropout:
		:param freeze: Whether to freeze the weights of the ResNet-50 model during training
		:param embed_size: Size of the embedding vector
		"""
		super().__init__()
		self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
		in_features = self.resnet.fc.in_features
		self.resnet = nn.Sequential(
			*list(self.resnet.children())[:-1]  # Remove the last FC layer (Classification layer)
		)

		if freeze:
			# Permanently mark the parameters so that gradients are not computed for them during the backward pass.
			# This means that these parameters will not be updated during training.
			for param in self.resnet.parameters():
				param.requires_grad = False

		# Add a linear layer to transform the features to the embedding size
		self.linear = nn.Sequential(
			nn.Linear(in_features, embed_size),
			nn.ReLU(),
			nn.Dropout(dropout)
		)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the encoder

		:param image: Input image tensor of shape (batch_size, 3, 224, 224)
		:return: 1D feature vector of shape (batch_size, feature_dim)
		"""
		features = self.resnet(image)  # Shape: (batch_size, 2048, 1, 1) where feature_dim=2048
		features = features.view(features.size(0), -1)  # Shape: (batch_size, 2048)
		features = self.linear(features)  # Shape: (batch_size, embed_size)
		return features


class Decoder(nn.Module):
	"""
	Decoder class that uses an LSTM to generate captions for images.
	"""

	def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, dropout: float, num_layers: int,
	             padding_idx: int) -> None:
		"""
		Constructor for the DecoderLSTM class

		:param embed_size: Size of the word embeddings
		:param hidden_size: Size of the hidden state of the LSTM
		:param vocab_size: Size of the vocabulary
		:param dropout: Dropout probability
		:param num_layers: Number of layers in the LSTM
		:param padding_idx: Index of the padding token in the vocabulary
		"""
		super().__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size

		# Project image features to initialize hidden and cell states
		self.init_h = nn.Linear(embed_size, num_layers * hidden_size)
		self.init_c = nn.Linear(embed_size, num_layers * hidden_size)

		self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
		self.layer_norm = nn.LayerNorm(hidden_size)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the decoder

		:param features: Image feature vectors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		print(f"Captions shape: {captions.shape}\n")
		print(f"Captions: {captions[0]}\n")
		print(f"Captions: {captions}\n")
		print(f"Features shape: {features.shape}\n")

		# Initialize hidden and cell states using image features
		batch_size = features.size(0)
		h = self.init_h(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
		c = self.init_c(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
		print(f"Hidden shape: {h.shape}\nCell shape: {c.shape}\n")

		# Embed and pack sequences
		embeddings = self.embed(captions)  # Shape: (batch_size, max_caption_length, embed_size)
		print(f"Embeddings shape: {embeddings.shape}\n")
		embeddings = self.dropout(embeddings)

		# LSTM forward
		lstm_out, _ = self.lstm(embeddings, (h, c))
		lstm_out = self.layer_norm(lstm_out)

		outputs = self.linear(lstm_out)  # Shape: (batch_size, max_caption_length + 1, vocab_size)
		return outputs


if __name__ == "__main__":
	config = {
		"encoder": "resnet50",
		"decoder": "LSTM",
		"batch_size": BATCH_SIZE,
		"embed_size": EMBED_SIZE,
		"hidden_size": HIDDEN_SIZE,
		"num_layers": NUM_LAYERS,
		"dropout": DROPOUT,
		"freeze_encoder": FREEZE_ENCODER,
		"encoder_lr": ENCODER_LR,
		"decoder_lr": DECODER_LR,
		"vocab": {
			"freq_threshold": VOCAB_THRESHOLD
		}
	}

	train_df = pd.read_csv(str(os.path.join(ROOT, TRAIN_CSV)))
	vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])
	img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
	train_dataset = FlickrDataset(img_dir, train_df, vocab, transform=TRANSFORM)
	pad_idx = vocab.to_idx(PAD)
	encoder = Encoder(config["embed_size"], config["freeze_encoder"], dropout=config["dropout"])
	decoder = Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
	model = ImageCaptioning(encoder, decoder)

	train_dataloader = FlickrDataLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
	optimizer = Adam([
		{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
		{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
	])
	images_, captions_, images_id = next(iter(train_dataloader))
	targets = captions_[:, 1:]  # Remove the <SOS> token | Shape: (batch_size, seq_len - 1)
	print(f"Initial captions shape: {captions_.shape}\n")
	print(f"Targets shape: {targets.shape}\n")
	outputs_ = model(images_, captions_[:, :-1])
	print(f"Outputs shape: {outputs_.shape}\n")
	loss = criterion(
		outputs_.reshape(-1, outputs_.size(-1)),  # Shape: (batch_size * (seq_len - 1), vocab_size)
		targets.reshape(-1)  # Shape: (batch_size * (seq_len - 1))
	)
	print(f"Loss: {loss}\n")
