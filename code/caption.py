import os
from typing import Optional

import torch
from PIL import Image
from torchvision.transforms import v2

from constants import SOS, EOS, ROOT, FLICKR8K_IMG_DIR, FLICKR8K_CSV_FILE, TEST_IMG, TEST_IMG_CAPTIONS
from dataset.flickr_dataset import load_captions
from dataset.vocabulary import Vocabulary
from models.basic import ImageCaptioning


def gen_caption(model: ImageCaptioning,
				image: torch.Tensor,
				vocab: Vocabulary,
				max_length: int = 50,
				device: torch.device = torch.device("cpu"),
				temperature: Optional[float] = None
				) -> str:
	"""
	Generate a caption for an image using greedy search or temperature-based sampling.

	:param model: Trained ImageCaptioning model
	:param image: Preprocessed image tensor (1, 3, 224, 224)
	:param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
	:param max_length: Maximum caption length
	:param device: Device to use 
	:param temperature: Temperature for sampling (None for greedy search)

	:return: Generated caption string
	"""
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


if __name__ == "__main__":
	root_dir_ = os.path.join(ROOT, FLICKR8K_IMG_DIR)
	ann_file_ = os.path.join(ROOT, FLICKR8K_CSV_FILE)

	transform_ = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
	])

	img_path_ = os.path.join(ROOT, TEST_IMG)
	img_ = preprocess_image(str(img_path_), transform_)

	df = load_captions(str(ann_file_), save_captions=False)
	vocab_ = Vocabulary(2, df["caption"])

	print(f"Image shape: {img_.size()}")
	print(f"Image captions: {TEST_IMG_CAPTIONS}")

	embed_size_ = 256
	hidden_size_ = 512
	vocab_size_ = len(vocab_)
	dropout_ = 0.5
	num_layers_ = 10

	model_ = ImageCaptioning(embed_size_, hidden_size_, vocab_size_, dropout_, num_layers_)

	print("Generating caption for the first image.")
	caption_ = gen_caption(model_, img_, vocab_)
	print(caption_)
