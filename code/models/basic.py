import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class EncoderResnet(nn.Module):
	"""
	Encoder class that uses a pretrained ResNet-50 model to extract features from images, i.e., encode images into a
	fixed-size feature vector suitable for captioning.
	"""

	def __init__(self, freeze: bool, embed_size: int) -> None:
		super(EncoderResnet, self).__init__()
		self.resnet = models.resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
		self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last FC layer (Classification layer)
		if freeze:
			for param in self.resnet.parameters():
				param.requires_grad = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the encoder
		:param x: Input image tensor of shape (batch_size, 3, 224, 224)
		:return: 1D feature vector of shape (batch_size, feature_dim)
		"""
		features = self.resnet(x)  # shape (batch_size, feature_size, 1, 1)
		return features.view(features.size(0), -1)  # Flatten the feature vector to (batch_size, feature_size)


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
		self.feature_embed = nn.Linear(feature_size, embed_size)  # transforms the input image features into the embedding size
		self.embed = nn.Embedding(vocab_size, embed_size)  # converts word indices into dense vectors of embed_size
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # processes the embedded word vectors
		self.linear = nn.Linear(hidden_size, vocab_size)  # maps the LSTM outputs to the vocabulary scores, producing the final word predictions
		self.dropout = nn.Dropout(dropout)

	def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the decoder
		:param features: Image feature vectors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		features = self.feature_embed(features)
		features = features.unsqueeze(1)
		embeddings = self.embed(captions)
		lstm_out, _ = self.lstm(
			torch.cat((features, embeddings), dim=1)  # concatenate the image features and caption embeddings
		)
		outputs = self.linear(lstm_out)

		return outputs

if __name__ == "__main__":
	m = models.resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
	print(m)