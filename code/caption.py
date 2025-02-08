from typing import Optional

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from constants import SOS, EOS
from dataset.vocabulary import Vocabulary


def gen_caption(model: nn.Module,
				image: torch.Tensor,
				vocab: Vocabulary,
				max_length: int = 30,
				device: torch.device = torch.device("cpu"),
				temperature: Optional[float] = None) -> str:
	"""
	Generate a caption for an image using greedy search or temperature-based sampling.

	:param model: Trained model for image captioning
	:param image: Preprocessed image tensor (1, 3, 224, 224)
	:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
	:param max_length: Maximum caption length
	:param device: Device to use 
	:param temperature: Temperature for sampling (None for greedy search)

	:return: Generated caption string
	"""
	model = model.to(device)
	model.eval()
	image = image.to(device)
	features = model.encoder(image)  # Encode the image
	caption = [vocab.str_to_idx[SOS]]  # Initialize caption with start token

	with torch.no_grad():
		for _ in range(max_length):
			caption_tensor = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
			outputs = model.decoder(features, caption_tensor)  # Get predictions (batch_size, seq_len+1, vocab_size)
			logits = outputs[:, -1, :]  # Get last predicted token (batch_size, vocab_size)

			# Choose next token
			if temperature is not None:
				# Temperature sampling
				probs = torch.softmax(logits / temperature, dim=-1)
				next_token = torch.multinomial(probs, 1).item()
			else:
				# Greedy search
				next_token = torch.argmax(logits, dim=-1).item()

			# Stop if we predict the end token
			if next_token == vocab.str_to_idx[EOS]:
				break

			caption.append(next_token)

	# Convert indices to words
	return vocab.to_text(caption)


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
