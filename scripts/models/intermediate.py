import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


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

	def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]) -> torch.Tensor:
		"""
		Forward pass of the decoder

		:param features: Image feature vectors
		:param captions: Caption word indices
		:param lengths: Actual caption lengths excluding padding
		:return: Predicted word indices
		"""
		batch_size = features.size(0)
		# Initialize hidden and cell states using image features
		h = self.init_h(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()  # Shape: (num_layers, batch_size, hidden_size)
		c = self.init_c(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()

		# Sort sequences by length (descending)
		lengths = torch.tensor(lengths, dtype=torch.long)
		lengths, sort_idx = torch.sort(lengths, descending=True)
		captions = captions[sort_idx]

		# Embed and pack sequences
		embeddings = self.embed(captions)  # Shape: (batch_size, max_caption_length, embed_size)
		embeddings = self.dropout(embeddings)
		packed = nn.utils.rnn.pack_padded_sequence(
			embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True
		)

		# LSTM forward
		packed_out, _ = self.lstm(packed, (h[:, sort_idx, :], c[:, sort_idx, :]))
		# Unpack, normalize, and restore order
		lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
		lstm_out = self.layer_norm(lstm_out)
		_, unsort_idx = sort_idx.sort()
		lstm_out = lstm_out[unsort_idx]

		outputs = self.linear(lstm_out)  # Shape: (batch_size, max_caption_length + 1, vocab_size)
		return outputs
