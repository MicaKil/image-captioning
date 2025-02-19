from typing import Optional

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

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
		return model.generate(image=image, vocab=vocab, max_length=max_length, temperature=temperature, beam_size=beam_size)


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
