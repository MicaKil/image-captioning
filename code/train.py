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


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, vocab: Vocabulary,
		  max_epochs: int, criterion: nn.Module, optimizer: torch.optim, checkpoint_dir: str, clip_grad: bool = False,
		  grad_max_norm: float = None, patience: int = None, calc_bleu: bool = False,
		  max_caption_len: int = None) -> None:
	"""
	Training loop for the model.

	:param grad_max_norm:
	:param model: The model to train
	:param train_loader: DataLoader for the training set
	:param val_loader: DataLoader for the validation set
	:param device: Device to run the training on
	:param vocab: Vocabulary object
	:param max_epochs: Maximum number of epochs to train
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param clip_grad:
	:param patience: Number of epochs to wait for improvement before early stopping
	:param checkpoint_dir: Directory to save the best model
	:param max_caption_len: Maximum length of the generated captions
	:param calc_bleu:
	:return:
	"""
	model_name = model.__class__.__name__
	model_params = sum(p.numel() for p in model.parameters())

	logger.info(f"Start training model {model_name} (Parameters: {model_params}) for {max_epochs} epochs")

	model = model.to(device)
	best_bleu_score = -np.inf
	best_val_loss = np.inf
	epochs_no_improve = 0
	smoothing = SmoothingFunction().method1

	for epoch in range(max_epochs):
		train_loss = train_load(model, train_loader, device, epoch, max_epochs, criterion, optimizer, clip_grad,
								grad_max_norm)
		val_loss, blue_score = eval_load(criterion, device, epoch, max_epochs, model, val_loader, vocab, calc_bleu,
										 max_caption_len, smoothing)

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


def train_load(model: nn.Module, train_loader: DataLoader, device: torch.device, epoch: int, max_epochs: int,
			   criterion: nn.Module, optimizer: torch.optim, clip_grad: bool, grad_max_norm: Optional[int]) -> float:
	"""
	Trains the model on the training set for one epoch

	:param grad_max_norm:
	:param model: Model to train
	:param train_loader: DataLoader for the training set
	:param device: Device to run the training on
	:param epoch: Current epoch
	:param max_epochs: Maximum number of epochs to train
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param clip_grad:
	:return: Total training loss for the epoch
	"""

	model.train()
	train_loss = 0.0
	batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
	for images, captions, images_id in batch_progress:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		loss = forward_pass(model, images, captions, criterion)

		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		if clip_grad:
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)  # Gradient clipping
		optimizer.step()

		train_loss += loss.item() * images.size(0)
		batch_progress.set_postfix({"loss": loss.item()})
	return train_loss


def eval_load(criterion: nn.Module, device: torch.device, epoch: int, max_epochs: int, model: nn.Module,
			  val_loader: DataLoader, vocab: Vocabulary, calc_bleu: Optional[bool], max_caption_len: Optional[int],
			  smoothing: Any) -> tuple:
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
	df = val_loader.dataset.df if isinstance(val_loader.dataset, FlickerDataset) else val_loader.dataset.dataset.df
	if calc_bleu:
		all_references = []  # List of lists of reference captions
		all_hypotheses = []  # List of generated captions

	with torch.no_grad():
		batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]")
		for images, captions, images_id in batch_progress:
			images = images.to(device)
			captions = captions.to(device)

			# Forward pass
			loss = forward_pass(model, images, captions, criterion)

			val_loss += loss.item() * images.size(0)
			batch_progress.set_postfix({"loss": loss.item()})
			# Generate captions for BLEU
			if calc_bleu:
				captions_for_bleu(all_hypotheses, all_references, device, df, images, images_id, max_caption_len, model,
								  vocab)
	if not calc_bleu:
		return val_loss, None

	blue_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing)
	return val_loss, blue_score


def forward_pass(model, images, captions, criterion):
	"""
	Performs a forward pass through the model.

	:param images: Batch of images. Shape: (batch_size, 3, 224, 224)
	:param captions: Batch of captions. Shape (batch_size, max_caption_length)
	:param model: The model to perform the forward pass.
	:param criterion: Loss function.
	:return: Loss value.
	"""
	outputs = model(images, captions[:, :-1])  # Output shape: (batch_size, max_caption_length, vocab_size)
	outputs = outputs[:, 1:, :]  # Remove the <SOS> token | Shape: (batch_size, max_caption_length - 1, vocab_size)
	loss = criterion(
		outputs.reshape(-1, outputs.size(-1)),
		captions[:, 1:].reshape(-1)
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
