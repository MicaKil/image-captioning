import argparse
import os.path

import torch
from torchvision.transforms import v2

import scripts.models.basic as basic
from configs.config import logger
from configs.runner_config import TRAIN_PATH
from constants import ROOT, PAD
from scripts import utils
from scripts.caption import gen_caption, preprocess_image
from scripts.models.image_captioning import ImageCaptioner
from scripts.utils import get_vocab

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])


def main():
	parser = argparse.ArgumentParser(description='Generate caption for an image')
	parser.add_argument('image_path', type=str, help='Path to input image')
	# parser.add_argument('--checkpoint', type=str, required=True,
	# 					help='Path to model checkpoint')
	parser.add_argument('--max_length', type=int, default=30,
	                    help='Maximum caption length')
	parser.add_argument('--temperature', type=float, default=None,
	                    help='Temperature for sampling (None for greedy)')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
	                    help='Compute device (cuda/cpu)')
	parser.add_argument('--beam_size', type=int, default=1,
	                    help='Beam size for beam search (1 for greedy/temperature sampling)')
	args = parser.parse_args()

	# Setup device
	device = torch.device(args.device)
	logger.info(f"Using device: {device}")

	try:
		# Load model and vocabulary
		model, vocab = load_model(device)

		# Load and transform image
		image_tensor = preprocess_image(img_path=args.image_path, transform=TRANSFORM)

		# Generate caption
		caption = gen_caption(model=model, image=image_tensor, vocab=vocab, max_length=args.max_length, device=device,
		                      temperature=args.temperature, beam_size=args.beam_size)

		print(f"\nGenerated Caption: {caption}\n")

		utils.show_img(image_tensor, mean=MEAN, std=STD, batch_dim=True)
	except Exception as e:
		logger.error(f"Error generating caption: {str(e)}")
		raise


def load_model(device: torch.device) -> tuple:
	train_dataset = torch.load(str(os.path.join(ROOT, TRAIN_PATH)), weights_only=False)
	vocab = get_vocab(train_dataset)
	encoder = basic.Encoder(512, True)
	decoder = basic.Decoder(512, 1024, len(vocab), 0.1, 3, vocab.to_idx(PAD))
	model = ImageCaptioner(encoder, decoder)
	model.load_state_dict(
		torch.load(os.path.join(ROOT, "checkpoints/basic/last_model_2025-02-14_03-51_4-3635.pt"), weights_only=True)
	)
	model = model.to(device)
	model.eval()
	return model, vocab


if __name__ == '__main__':
	# python main.py datasets/my_pics/2025-02-16_21-32.png --checkpoint checkpoints/basic/last_model_2025-02-14_03-51_4-3635.pt
	print("\nWelcome to the Image Captioning Generator!\n")
	main()
