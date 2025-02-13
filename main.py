import argparse

import torch

from config import logger


def main():
	parser = argparse.ArgumentParser(description='Generate caption for an image')
	parser.add_argument('image_path', type=str, help='Path to input image')
	parser.add_argument('--checkpoint', type=str, required=True,
						help='Path to model checkpoint')
	parser.add_argument('--max_length', type=int, default=30,
						help='Maximum caption length')
	parser.add_argument('--temperature', type=float, default=None,
						help='Temperature for sampling (None for greedy)')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
						help='Compute device (cuda/cpu)')

	args = parser.parse_args()

	# Setup device
	device = torch.device(args.device)
	logger.info(f"Using device: {device}")
