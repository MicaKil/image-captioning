import logging
import os.path
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption import gen_caption
from constants import ROOT, PAD
from dataset.flickr_dataset import FlickerDataset
from dataset.vocabulary import Vocabulary
from utils import time_str

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, vocab: Vocabulary,
		  max_epochs: int, criterion: nn.Module, optimizer: torch.optim, checkpoint_dir: Optional[str],
		  clip_grad: bool = False, grad_max_norm: float = None, patience: int = None, calc_bleu: bool = False,
		  max_caption_len: int = None) -> None:
	"""
	Training loop for the model.

	:param model: The model to train
	:param train_loader: DataLoader for the training set
	:param val_loader: DataLoader for the validation set
	:param device: Device to run the training on
	:param vocab: Vocabulary of the dataset
	:param max_epochs: Maximum number of epochs to train
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param checkpoint_dir: Directory to save the best model
	:param clip_grad: Whether to clip gradients
	:param grad_max_norm: Maximum norm for gradient clipping
	:param patience: Number of epochs to wait for improvement before early stopping
	:param calc_bleu: Whether to calculate BLEU score
	:param max_caption_len: Maximum length of the generated captions
	:return:
	"""

	logger.info(
		f"Start training model {model.__class__.__name__} (Parameters: {sum(p.numel() for p in model.parameters())}) for {max_epochs} epochs"
	)

	model = model.to(device)
	best_bleu_score = -np.inf
	best_val_loss = np.inf
	epochs_no_improve = 0

	for epoch in range(max_epochs):
		avg_train_loss = train_load(model, train_loader, vocab, device, epoch, max_epochs, criterion, optimizer,
									clip_grad, grad_max_norm)
		avg_val_loss, blue_score = eval_load(model, val_loader, vocab, device, epoch, max_epochs, criterion, calc_bleu,
											 max_caption_len, SmoothingFunction().method1)

		logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

		# Early stopping and checkpointing
		if calc_bleu:
			if blue_score > best_bleu_score:
				best_bleu_score = blue_score
				if checkpoint_dir is not None:
					torch.save(model.state_dict(), os.path.join(ROOT, f"{checkpoint_dir}/best_bleu_{time_str()}.pt"))
				logger.info(f"New best BLEU score: {best_bleu_score:.4f}")
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			epochs_no_improve = 0
			if checkpoint_dir is not None:
				torch.save(model.state_dict(), os.path.join(ROOT, f"{checkpoint_dir}/best_val_{time_str()}.pt"))
			logger.info(f"New best validation loss: {best_val_loss:.4f}")
		else:
			if patience is not None:
				epochs_no_improve += 1
				if epochs_no_improve >= patience:
					logger.info(f"Early stopping after {epoch + 1} epochs")
					break


def train_load(model: nn.Module, train_loader: DataLoader, vocab: Vocabulary, device: torch.device, epoch: int,
			   max_epochs: int, criterion: nn.Module, optimizer: torch.optim, clip_grad: bool,
			   grad_max_norm: Optional[int]) -> float:
	"""
	Trains the model on the training set for one epoch

	:param model: Model to train
	:param vocab: Vocabulary of the dataset
	:param train_loader: DataLoader for the training set
	:param device: Device to run the training on
	:param epoch: Current epoch
	:param max_epochs: Maximum number of epochs to train
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param clip_grad: Whether to clip gradients
	:param grad_max_norm: Maximum norm for gradient clipping
	:return: Total training loss for the epoch
	"""
	model.train()
	train_loss = 0.
	total_tokens = 0

	batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
	for images, captions, images_id in batch_progress:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		loss, num_tokens = forward_pass(model, images, captions, criterion, vocab.to_idx(PAD))
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		if clip_grad:
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)  # Gradient clipping
		optimizer.step()

		train_loss += loss.item()
		total_tokens += num_tokens
		batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

	avg_loss = train_loss / total_tokens if total_tokens > 0 else 0
	return avg_loss


def eval_load(model: nn.Module, val_loader: DataLoader, vocab: Vocabulary, device: torch.device, epoch: int,
			  max_epochs: int, criterion: nn.Module, calc_bleu: Optional[bool], max_caption_len: Optional[int],
			  smoothing: Any) -> tuple:
	"""
	Evaluates the model on the validation set for one epoch

	:param model: Model to evaluate
	:param val_loader: DataLoader for the validation set
	:param vocab: Vocabulary of the dataset
	:param device: Device to run the evaluation on
	:param epoch: Current epoch
	:param max_epochs: Maximum number of epochs to train
	:param criterion: Loss function
	:param calc_bleu: Whether to calculate BLEU score
	:param max_caption_len: Maximum length of the generated captions
	:param smoothing: Smoothing function for BLEU score
	:return: Average validation loss and BLEU score (if calc_bleu is True)
	"""
	model.eval()
	val_loss = 0.0
	total_tokens = 0
	df = val_loader.dataset.df if isinstance(val_loader.dataset, FlickerDataset) else val_loader.dataset.dataset.df
	if calc_bleu:
		all_references = []  # List of lists of reference captions
		all_hypothesis = []  # List of generated captions (hypotheses)

	with torch.no_grad():
		batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]")
		for images, captions, images_id in batch_progress:
			images = images.to(device)
			captions = captions.to(device)

			loss, num_tokens = forward_pass(model, images, captions, criterion, vocab.to_idx(PAD))  # Forward pass

			val_loss += loss.item()
			total_tokens += num_tokens
			batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})
			# Generate captions for BLEU
			if calc_bleu:
				generated, ref = captions_for_bleu(device, model, vocab, df, images, images_id, max_caption_len)
				all_hypothesis.extend(generated)
				all_references.extend(ref)

	avg_loss = val_loss / total_tokens if total_tokens > 0 else 0
	if not calc_bleu:
		return avg_loss, None

	return avg_loss, corpus_bleu(all_references, all_hypothesis, smoothing_function=smoothing)


def forward_pass(model: nn.Module, images: torch.Tensor, captions: torch.Tensor, criterion: nn.Module,
				 pad_idx: int) -> tuple:
	"""
	Performs a forward pass through the model.

	:param model: The model to perform the forward pass.
	:param images: Batch of images. Shape: (batch_size, 3, 224, 224)
	:param captions: Batch of captions. Shape (batch_size, seq_len)
	:param criterion: Loss function.
	:param pad_idx: Index of the padding token in the vocabulary.
	:return: The loss and the number of tokens.
	"""
	outputs = model(images, captions[:, :-1])  # Shape: (batch_size, seq_len, vocab_size)
	outputs = outputs[:, 1:, :]  # Remove the <SOS> token | Shape: (batch_size, seq_len - 1, vocab_size)
	targets = captions[:, 1:]  # Remove the <SOS> token | Shape: (batch_size, seq_len - 1)

	mask = (targets != pad_idx)
	num_tokens = mask.sum().item()
	loss = criterion(
		outputs.reshape(-1, outputs.size(-1)),  # Shape: (batch_size * (seq_len - 1), vocab_size)
		targets.reshape(-1)  # Shape: (batch_size * (seq_len - 1))
	)
	return loss, num_tokens


def captions_for_bleu(device: torch.device, model: nn.Module, vocab: Vocabulary, df: pd.DataFrame, images: torch.Tensor,
					  images_id: list, max_caption_len: int) -> tuple:
	"""
	Generates captions for BLEU score calculation in a batch.

	:param device: Device to run the generation on (CPU or GPU).
	:param model: The model to generate captions.
	:param df: DataFrame containing image IDs and captions.
	:param images: Batch of images.
	:param images_id: List of image IDs.
	:param max_caption_len: Maximum length of the generated captions.
	:param vocab: Vocabulary object.
	:return: List of generated captions and list of reference captions.
	"""
	generated = []
	for image in images:
		image = image.unsqueeze(0).to(device)
		caption = gen_caption(model, image, vocab, max_caption_len, device)
		generated.append(caption.split())
	# Process ground truth captions
	references = []
	for image_id in images_id:
		captions = df[df["image_id"] == image_id]["caption"].values
		references.append([caption.split() for caption in captions])
	return generated, references
