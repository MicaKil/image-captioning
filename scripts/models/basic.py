import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class EncoderResnet(nn.Module):
	"""
	Encoder class that uses a pretrained ResNet-50 model to extract features from images.
	"""

	def __init__(self, embed_size: int, freeze: bool) -> None:
		"""
		Constructor for the EncoderResnet class

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
		self.linear = nn.Linear(in_features, embed_size)

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


class DecoderLSTM(nn.Module):
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
		self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the decoder

		:param features: Image feature vectors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		embeddings = self.embed(captions)  # Shape: (batch_size, max_caption_length, embed_size)
		embeddings = self.dropout(embeddings)
		features = features.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
		lstm_out, _ = self.lstm(
			torch.cat((features, embeddings), dim=1)  # concatenate the image features and caption embeddings
		)  # Shape: (batch_size, max_caption_length + 1, hidden_size)
		outputs = self.linear(lstm_out)  # Shape: (batch_size, max_caption_length + 1, vocab_size)

		return outputs


class ImageCaptioning(nn.Module):
	"""
	Image captioning model that combines an Encoder and Decoder.
	"""

	def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, dropout: float, num_layers: int,
	             padding_idx: int = 0, freeze_encoder=True):
		"""
		Constructor for the ImageCaptioning class

		:param embed_size: Size of the embedding vector
		:param hidden_size: Size of the hidden state of the LSTM
		:param vocab_size: Size of the vocabulary
		:param dropout: Dropout probability
		:param num_layers: Number of layers in the LSTM
		:param padding_idx: Index of the padding token in the vocabulary
		:param freeze_encoder: Whether to freeze the weights of the Encoder during training
		"""
		super().__init__()
		self.encoder = EncoderResnet(embed_size, freeze_encoder)
		self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, dropout, num_layers, padding_idx)

	def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the ImageCaptioning model

		:param images: Input image tensors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		features = self.encoder(images)  # Shape: (batch_size, embed_size)
		outputs = self.decoder(features, captions)  # Shape: (batch_size, max_caption_length + 1, vocab_size)
		return outputs
