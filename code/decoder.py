import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
	"""
	Decoder class that uses an LSTM to generate captions for images.
	Has optional attention mechanism.
	"""

	def __init__(self, feature_size: int, embed_size: int, hidden_size: int, vocab_size: int, num_layers=1, use_attention=False) -> None:
		"""
		Constructor for the DecoderLSTM class

		:param feature_size: Size of the image feature vectors
		:param embed_size: Size of the word embeddings
		:param hidden_size: Size of the hidden state of the LSTM
		:param vocab_size: Size of the vocabulary
		:param num_layers: Number of layers in the LSTM
		:param use_attention: Whether to use attention mechanism
		"""
		super(DecoderLSTM, self).__init__()
		self.use_attention = use_attention

		self.feature_embed = nn.Linear(feature_size, embed_size)  # Linear layer that transforms the input image features into the embedding size (embed_size)
		self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer converts word indices into dense vectors of a specified size (embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM network that processes the embedded word vectors
		self.fc = nn.Linear(hidden_size, vocab_size)  # The fc layer is a fully connected layer that maps the LSTM outputs to the vocabulary size, producing the final word predictions

		if use_attention:
			self.attention = nn.Linear(hidden_size + embed_size, hidden_size)
			self.attention_combine = nn.Linear(hidden_size + embed_size, embed_size)

	def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the decoder
		:param features: Image feature vectors
		:param captions: Caption word indices
		:return: Predicted word indices
		"""
		embedded_features = self.feature_embed(features).unsqueeze(1)  #  Input image features are first transformed using the feature_embed layer and then unsqueezed to add a time dimension
		embeddings = self.embed(captions)  # Captions are embedded using the embed layer

		if self.use_attention:  # the LSTM outputs are combined with the embedded features to compute attention weights, which are then applied to the LSTM outputs
			h, _ = self.lstm(embeddings)
			attention_weights = torch.softmax(
				self.attention(torch.cat((h, embedded_features.repeat(1, h.size(1), 1)), dim=2)), dim=2
			)
			attention_applied = torch.bmm(attention_weights.permute(0, 2, 1), h)
			embeddings = self.attention_combine(
				torch.cat((attention_applied.squeeze(1), embedded_features.squeeze(1)), dim=1)
			).unsqueeze(1)

		# The final LSTM outputs are passed through the fc layer to generate the predicted word indices
		lstm_out, _ = self.lstm(embeddings)
		outputs = self.fc(lstm_out)

		return outputs
