from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from constants import SOS, EOS
from scripts.dataset.vocabulary import Vocabulary


class Encoder(nn.Module):
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
		self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]) -> torch.Tensor:
		"""
		Forward pass of the decoder

		:param features: Image feature vectors
		:param captions: Caption word indices
		:param lengths: Actual caption lengths excluding padding
		:return: Predicted word indices (batch_size, padded_length, vocab_size)
		"""
		# Calculate new lengths (image feature + actual caption length)
		new_lengths = [l + 1 for l in lengths]  # +1 for the image feature

		# Sort sequences by descending lengths
		new_lengths_tensor = torch.tensor(new_lengths, dtype=torch.long)
		sorted_lengths, sort_idx = torch.sort(new_lengths_tensor, descending=True)
		sorted_captions = captions[sort_idx]
		sorted_features = features[sort_idx]

		# Embed sorted captions
		embeddings = self.embed(sorted_captions)  # (batch_size, padded_seq_len, embed_size)
		embeddings = self.dropout(embeddings)

		# Concatenate image features with embeddings
		image_features = sorted_features.unsqueeze(1)  # (batch_size, 1, embed_size)
		combined = torch.cat([image_features, embeddings], dim=1)  # (batch_size, padded_seq_len + 1, embed_size)

		# Pack sequences (ignore padding)
		packed_combined = nn.utils.rnn.pack_padded_sequence(combined, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)

		# LSTM forward pass
		packed_out, _ = self.lstm(packed_combined)

		# Unpack and restore original order
		lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
		_, unsort_idx = sort_idx.sort()
		lstm_out = lstm_out[unsort_idx]

		# Project to vocabulary
		outputs = self.linear(lstm_out)
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
		self.encoder = Encoder(embed_size, freeze_encoder)
		self.decoder = Decoder(embed_size, hidden_size, vocab_size, dropout, num_layers, padding_idx)

	def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: list[int]) -> torch.Tensor:
		"""
		Forward pass of the ImageCaptioning model

		:param images: Input image tensors
		:param captions: Caption word indices
		:param lengths: Actual caption lengths excluding padding
		:return: Predicted word indices
		"""
		features = self.encoder(images)  # Shape: (batch_size, embed_size)
		outputs = self.decoder(features, captions, lengths)  # Shape: (batch_size, max_caption_length + 1, vocab_size)
		return outputs

	def generate(self, image: torch.Tensor, vocab: Vocabulary, max_length: int = 30, device: torch.device = torch.device("cpu"),
	             temperature: Optional[float] = None, beam_size: int = 1) -> str:
		self.eval()
		with torch.no_grad():
			image = image.to(device)
			features = self.encoder(image)  # Encode the image (1, embed_size)
			if beam_size > 1:
				return self.beam_search(vocab, device, features, max_length, beam_size)
			else:
				return self.temperature_sampling(vocab, device, features, max_length, temperature)

	def temperature_sampling(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int,
	                         temperature: Optional[float]) -> str:
		"""
		Generate a caption using temperature-based sampling if temperature is not None, otherwise use greedy search.

		:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
		:param device: Device to use
		:param features: Encoded image features
		:param max_length: Maximum caption length
		:param temperature: Temperature for sampling (None for greedy search)
		:return: Generated caption as a list of token indices
		"""
		caption = [vocab.to_idx(SOS)]  # Initialize caption with start token
		for _ in range(max_length):
			caption_tensor = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
			current_length = caption_tensor.size(1)
			outputs = self.decoder(features, caption_tensor, [current_length])  # Get predictions (batch_size, seq_len+1, vocab_size)
			logits = outputs[:, -1, :]  # Get last predicted token (batch_size, vocab_size)

			# Choose next token
			if temperature is None or temperature == 0.0:
				# Greedy search
				next_token = torch.argmax(logits, dim=-1).item()
			else:
				# Temperature sampling
				probs = torch.softmax(logits / temperature, dim=-1)
				next_token = torch.multinomial(probs, 1).item()

			# Stop if we predict the end token
			if next_token == vocab.to_idx(EOS):
				break

			caption.append(next_token)
		return vocab.to_text(caption)

	def beam_search(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int, beam_size: int) -> str:
		"""
		Generate a caption using beam search.

		:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
		:param device: Device to use
		:param features: Encoded image features
		:param max_length: Maximum caption length
		:param beam_size: Beam size for beam search
		:return: Generated caption as a list of token indices
		"""
		sos_idx = vocab.to_idx(SOS)
		eos_idx = vocab.to_idx(EOS)
		vocab_size = len(vocab)
		# The image features are replicated beam_size times (one copy per beam/hypothesis)
		features = features.expand(beam_size, -1)  # (beam_size, embed_size)
		# Initialize beam
		beam_scores = torch.zeros(beam_size).to(device)  # log probabilities
		beam_sequences = torch.tensor([[sos_idx]] * beam_size, dtype=torch.long).to(device)  # all beams start with SOS
		completed_sequences = []
		completed_scores = []
		for _ in range(max_length):
			# Use the decoder to predict logits for the next token:
			current_lengths = [seq.size(0) for seq in beam_sequences]
			outputs = self.decoder(features, beam_sequences, current_lengths)  # (beam_size, seq_len, vocab_size)
			# Extract the last token's logits and compute log probabilities
			logits = outputs[:, -1, :]  # (beam_size, vocab_size)
			log_probs = torch.log_softmax(logits, dim=-1)

			# Combine scores
			scores = log_probs + beam_scores.unsqueeze(1)  # (beam_size, vocab_size)
			scores_flat = scores.view(-1)  # (beam_size * vocab_size)

			# Get top candidates
			top_scores, top_indices = torch.topk(scores_flat, k=beam_size)
			beam_indices = top_indices // vocab_size  # Which beam does this come from?
			token_indices = top_indices % vocab_size  # Which token does this predict?

			# Update sequences
			beam_sequences = torch.cat([beam_sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)
			beam_scores = top_scores

			# Check for completed sequences
			eos_mask = token_indices == eos_idx
			if eos_mask.any():
				# If a beam predicts the EOS token, move it to the completed_sequences list and remove it from active beams
				completed_sequences.extend(beam_sequences[eos_mask].tolist())
				completed_scores.extend(beam_scores[eos_mask].tolist())

				# Remove completed sequences from beam
				keep_mask = ~eos_mask
				beam_sequences = beam_sequences[keep_mask]
				beam_scores = beam_scores[keep_mask]
				features = features[keep_mask]

				if beam_sequences.size(0) == 0:
					break  # All sequences completed
		# Select best sequence
		if completed_sequences:
			best_idx = torch.argmax(torch.tensor(completed_scores)).item()
			best_sequence = completed_sequences[best_idx]
		else:
			best_idx = torch.argmax(beam_scores).item()
			best_sequence = beam_sequences[best_idx].tolist()
		return vocab.to_text(best_sequence)
