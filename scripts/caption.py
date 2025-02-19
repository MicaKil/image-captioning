from typing import Optional

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from constants import SOS, EOS
from scripts.dataset.vocabulary import Vocabulary


def gen_caption(model: nn.Module, image: torch.Tensor, vocab: Vocabulary, max_length: int = 30, device: torch.device = torch.device("cpu"),
                temperature: Optional[float] = None, beam_size: int = 1) -> str:
	"""
	Generate a caption for an image using greedy search, temperature-based sampling, or beam search.

	:param model: Trained model for image captioning
	:param image: Preprocessed image tensor (B, C, H, W)
	:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
	:param max_length: Maximum caption length
	:param device: Device to use
	:param temperature: Temperature for sampling (None for greedy search)
	:param beam_size: Beam size for beam search (1 for greedy search/temperature sampling)

	:return: Generated caption string
	"""
	model = model.to(device)
	model.eval()

	with torch.no_grad():
		image = image.to(device)
		features = model.encoder(image)  # Encode the image (1, embed_size)
		if beam_size > 1:
			caption = beam_search(model, vocab, device, features, max_length, beam_size)
		else:
			caption = temperature_sampling(model, vocab, device, features, max_length, temperature)

	# Convert indices to words
	return vocab.to_text(caption)


def temperature_sampling(model: nn.Module, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int,
                         temperature: Optional[float]) -> list[int]:
	"""
	Generate a caption using temperature-based sampling if temperature is not None, otherwise use greedy search.

	:param model: Trained model for image captioning
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
		outputs = model.decoder(features, caption_tensor)  # Get predictions (batch_size, seq_len+1, vocab_size)
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
	return caption


def beam_search(model: nn.Module, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int, beam_size: int) -> list[int]:
	"""
	Generate a caption using beam search.

	:param model: Trained model for image captioning
	:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
	:param device: Device to use
	:param features: Encoded image features
	:param max_length: Maximum caption length
	:param beam_size: Beam size for beam search
	:return: Generated caption as a list of token indices
	"""
	# Beam Search Implementation
	features = features.expand(beam_size, -1)  # (beam_size, embed_size)
	sos_idx = vocab.to_idx(SOS)
	eos_idx = vocab.to_idx(EOS)
	vocab_size = len(vocab)
	# Initialize beam
	beam_scores = torch.zeros(beam_size).to(device)  # log probabilities
	beam_sequences = torch.tensor([[sos_idx]] * beam_size, dtype=torch.long).to(device)
	completed_sequences = []
	completed_scores = []
	for _ in range(max_length):
		outputs = model.decoder(features, beam_sequences)  # (beam_size, seq_len, vocab_size)
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
	return best_sequence


def preprocess_image(img_path: str, transform: v2.Compose) -> torch.Tensor:
	"""
	Preprocess an image for the model.

	:param img_path: Path to the image file
	:param transform: Transform to apply to the image
	:return: Preprocessed image tensor of shape (1, 3, 224, 224)
	"""
	img = Image.open(img_path).convert("RGB")
	img = transform(img)
	img = img.unsqueeze(0)  # Add batch dimension
	return img
