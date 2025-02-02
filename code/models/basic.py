import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
	"""
	Encoder class that uses a pretrained ResNet-50 model to extract features from images, i.e., encode images into a
	fixed-size feature vector suitable for captioning.
	"""

	def __init__(self):
		super(Encoder, self).__init__()
		self.model = models.resnet50(pretrained=True)
		self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the last FC layer (Classification layer)
		self.model.eval()  # Set the model to evaluation mode (don't update weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the encoder
		:param x: Input image tensor of shape (batch_size, 3, 224, 224)
		:return: 1D feature vector of shape (batch_size, 2048)
		"""
		with torch.no_grad():  # No need to compute gradients
			features = self.model(x)
		return features.view(features.size(0), -1)  # Flatten the feature vector


class DecoderLSTM(nn.Module):
	"""
	Decoder class that uses an LSTM to generate captions for images.
	"""

	def __init__(self, feature_size: int, embed_size: int, hidden_size: int, vocab_size: int, dropout: float,
				 num_layers=1) -> None:
		"""
		Constructor for the DecoderLSTM class

		:param feature_size: Size of the image feature vectors
		:param embed_size: Size of the word embeddings
		:param hidden_size: Size of the hidden state of the LSTM
		:param vocab_size: Size of the vocabulary
		:param dropout: Dropout probability
		:param num_layers: Number of layers in the LSTM
		"""
		super(DecoderLSTM, self).__init__()
		self.feature_embed = nn.Linear(feature_size, embed_size)  # Linear layer that transforms the input image features into the embedding size (embed_size)
		self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer converts word indices into dense vectors of a specified size (embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM network that processes the embedded word vectors
		self.linear = nn.Linear(hidden_size, vocab_size)  # The fc layer is a fully connected layer that maps the LSTM outputs to the vocabulary size, producing the final word predictions
		self.dropout = nn.Dropout(dropout)

	def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the decoder
		:param features: Image feature vectors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		embeddings = self.dropout(self.embed(captions))  # Captions are embedded using the embed layer
		lstm_out, _ = self.lstm(
			embeddings)  # The final LSTM outputs are passed through the fc layer to generate the predicted word indices
		outputs = self.linear(lstm_out)

		return outputs
