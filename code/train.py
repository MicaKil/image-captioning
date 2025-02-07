import logging
import os.path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption import gen_caption
from constants import ROOT
from dataset.flickr_dataset import FlickerDataset
from dataset.vocabulary import Vocabulary
from utils import time_str

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
		  optimizer: torch.optim, device: torch.device, vocab: Vocabulary, max_epochs: int, patience: Optional[int],
		  checkpoint_dir: str, max_caption_len: int = None, calc_bleu=False) -> None:
	"""
	Training loop for the model.

	:param model: The model to train
	:param train_loader: DataLoader for the training set
	:param val_loader: DataLoader for the validation set
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param device: Device to run the training on
	:param vocab: Vocabulary object
	:param max_epochs: Maximum number of epochs to train
	:param patience: Number of epochs to wait for improvement before early stopping
	:param checkpoint_dir: Directory to save the best model
	:param max_caption_len: Maximum length of the generated captions
	:param calc_bleu:
	:return:
	"""

	logger.info(f"Start training for {max_epochs} epochs.")

	model = model.to(device)
	best_bleu_score = -np.inf
	best_val_loss = np.inf
	epochs_no_improve = 0
	smoothing = SmoothingFunction().method1

	for epoch in range(max_epochs):
		train_loss = train_load(criterion, device, epoch, max_epochs, model, optimizer, train_loader)
		val_loss, blue_score = eval_load(criterion, device, epoch, max_epochs, model, val_loader, vocab,
										 max_caption_len, smoothing, calc_bleu)

		# Calculate epoch metrics
		train_loss /= len(train_loader.dataset)
		val_loss /= len(val_loader.dataset)
		logger.info(f"Epoch {epoch + 1} | Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

		# Early stopping and checkpointing
		if calc_bleu:
			if blue_score > best_bleu_score:
				best_bleu_score = blue_score
				torch.save(model.state_dict(), os.path.join(ROOT, f"{checkpoint_dir}/best_bleu_{time_str()}.pt"))
				logger.info(f"New best BLEU score: {best_bleu_score:.4f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_no_improve = 0
			torch.save(model.state_dict(), os.path.join(ROOT, f"{checkpoint_dir}/best_val_{time_str()}.pt"))
			logger.info(f"New best validation loss: {best_val_loss:.4f}")
		else:
			if patience is not None:
				epochs_no_improve += 1
				if epochs_no_improve >= patience:
					logger.info(f"Early stopping after {epoch + 1} epochs")
					break


def train_load(criterion: nn.Module,
			   device: torch.device,
			   epoch: int,
			   max_epochs: int,
			   model: nn.Module,
			   optimizer: torch.optim,
			   train_loader: DataLoader):
	"""
	Trains the model on the training set for one epoch

	:param criterion: Loss function
	:param device: Device to run the training on
	:param epoch: Current epoch
	:param max_epochs: Maximum number of epochs to train
	:param model: Model to train
	:param optimizer: Optimizer for training
	:param train_loader: DataLoader for the training set
	:return: Total training loss for the epoch
	"""

	model.train()
	train_loss = 0.0
	batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
	for images, captions, images_id in batch_progress:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		loss = forward_pass(captions, criterion, images, model)

		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
		optimizer.step()
		# After optimizer.step():
		for name, param in model.named_parameters():
			if param.grad is not None:
				print(f"{name} gradient norm: {param.grad.norm()}")
			else:
				print(f"{name} has no gradients!")

		train_loss += loss.item() * images.size(0)
		batch_progress.set_postfix({"loss": loss.item()})
	return train_loss


def eval_load(criterion: nn.Module, device: torch.device, epoch: int, max_epochs: int, model: nn.Module,
			  val_loader: DataLoader, vocab: Vocabulary, max_caption_len: int = None, smoothing: Any = None,
			  calc_bleu=False) -> tuple:
	"""
	Evaluates the model on the validation set for one epoch

	:param criterion: Loss function
	:param device: Device to run the evaluation on
	:param epoch: Current epoch
	:param max_epochs: Maximum number of epochs to train
	:param model: Model to evaluate
	:param val_loader: DataLoader for the validation set
	:param vocab: Vocabulary object
	:param max_caption_len: Maximum length of the generated captions
	:param smoothing: Smoothing function for BLEU score
	:param calc_bleu:
	:return: Total validation loss for the epoch and BLEU score
	"""

	model.eval()
	val_loss = 0.0
	all_references = []  # List of lists of reference captions
	all_hypotheses = []  # List of generated captions
	df = val_loader.dataset.df if isinstance(val_loader.dataset, FlickerDataset) else val_loader.dataset.dataset.df

	with torch.no_grad():
		batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]")
		for images, captions, images_id in batch_progress:
			images = images.to(device)
			captions = captions.to(device)

			# Forward pass
			val_loss = forward_pass(captions, criterion, images, model)
			val_loss += val_loss.item() * images.size(0)
			batch_progress.set_postfix({"loss": val_loss.item()})
			# Generate captions for BLEU
			if calc_bleu:
				captions_for_bleu(all_hypotheses, all_references, device, df, images, images_id, max_caption_len, model,
								  vocab)
	if not calc_bleu:
		return val_loss, None

	blue_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing)
	return val_loss, blue_score


def forward_pass(captions, criterion, images, model):
	"""
	Performs a forward pass through the model.

	:param captions: Batch of captions.
	:param criterion: Loss function.
	:param images: Batch of images.
	:param model: The model to perform the forward pass.
	:return: Loss value.
	"""
	# Image size: (batch_size, 3, 224, 224) Captions size: (batch_size, max_caption_length)
	# print(f"\nImages size: {images.size()}")
	# print(f"Captions size: {captions.size()} || {captions[:, :-1].size()}")

	# Teacher forcing
	# captions[:, :-1] slices the captions tensor along the time (sequence) dimension to remove the last token from each caption.
	# Why? In many sequence-to-sequence tasks, you feed the model the caption without its final token so that the model learns to predict the next word at every time step.
	outputs = model(images, captions[:, :-1])  # Output size: (batch_size, max_caption_length, vocab_size)
	# print(f"Outputs size: {outputs.size()}")
	# During evaluation:
	generated = model(images, captions[:, :-1])
	print("Model output example:", generated[0].argmax(dim=-1))
	print("Target example:", captions[0, 1:])
	print("Output min/max:", outputs.min(), outputs.max())
	print("Target min/max:", captions[:, 1:].min(), captions[:, 1:].max())

	# This line removes the first time step from the outputs tensor by taking all elements starting from index 1 along the time dimension.
	# Why?
	# - The model may output an initial prediction (often corresponding to an artificial or default starting state) that doesn’t correspond to a target token.
	# - This adjustment aligns the output sequence with the target sequence, which starts at the first true word (excluding the start-of-sequence token).
	# Now, outputs is modified so that its time dimension matches the target tokens, typically of shape (batch_size, sequence_length - 1, vocab_size).
	outputs = outputs[:, 1:, :]
	# print(f"Outputs size: {outputs.size()}")

	loss = criterion(
		# The outputs tensor is reshaped from a 3D tensor (batch_size, sequence_length - 1, vocab_size) to a 2D tensor (batch_size * (sequence_length - 1), vocab_size).
		# This flattening is required because loss functions like cross-entropy expect a 2D input
		outputs.reshape(-1, outputs.size(-1)), # (B * seq_len, vocab_size)
		# captions[:, 1:] slices the original captions tensor to remove the first token, which is usually the Start-Of-Sequence (SOS) token.
		# This ensures that the target tokens are the ones the model is expected to predict.
		# captions[:, 1:].reshape(-1): The target tokens are reshaped from (batch_size, sequence_length - 1) to a 1D tensor of size (batch_size * (sequence_length - 1)).
		captions[:, 1:].reshape(-1)  # Exclude first token (SOS token) # (B * seq_len, vocab_size)
	)
	return loss


def captions_for_bleu(all_hypotheses, all_references, device, df, images, images_id, max_caption_len, model, vocab):
	"""
	Generates captions for BLEU score calculation.

	:param all_hypotheses: List to store generated captions.
	:param all_references: List to store reference captions.
	:param device: Device to run the generation on (CPU or GPU).
	:param df: DataFrame containing image IDs and captions.
	:param images: Batch of images.
	:param images_id: List of image IDs.
	:param max_caption_len: Maximum length of the generated captions.
	:param model: The model to generate captions.
	:param vocab: Vocabulary object.
	:return: None
	"""
	generated_captions = []
	for i in range(images.size(0)):
		image = images[i].unsqueeze(0)
		caption = gen_caption(model, image, vocab, max_caption_len, device)
		generated_captions.append(caption.split())
	# Process ground truth captions
	references = []
	for i in range(images.size(0)):
		captions = df[df["image_id"] == images_id[i]]["caption"].values
		references.append([caption.split() for caption in captions])
	all_hypotheses.extend(generated_captions)
	all_references.extend(references)
