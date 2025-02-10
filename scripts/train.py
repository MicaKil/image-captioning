import logging
import os.path
import time
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import ROOT, PAD
from scripts.caption import gen_caption
from scripts.dataset.flickr_dataset import FlickerDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.utils import time_str

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

config = wandb.config

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
		  criterion: nn.Module, optimizer: torch.optim, checkpoint_dir: Optional[str]) -> str:
	"""
	Training loop for the model.

	:param model: The model to train
	:param train_loader: DataLoader for the training set
	:param val_loader: DataLoader for the validation set
	:param device: Device to run the training on
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:param checkpoint_dir: Directory to save the best model
	:return: Path to the best model
	"""
	wandb.watch(model, criterion=criterion, log="all", log_freq=100)

	logger.info(
		f"Start training model {model.__class__.__name__} (Parameters: {sum(p.numel() for p in model.parameters())}) for {config["max_epochs"]} epochs")
	start_time = time.time()

	best_bleu_score = -np.inf
	best_val_loss = np.inf
	best_model = None
	epochs_no_improve = 0

	model = model.to(device)
	for epoch in range(config["max_epochs"]):
		avg_train_loss = train_load(model, train_loader, device, epoch, criterion, optimizer)
		avg_val_loss, blue_score = eval_load(model, val_loader, device, epoch, criterion, SmoothingFunction().method1)

		logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
		wandb.log({
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"val_loss": avg_val_loss
		})
		# Early stopping and checkpointing
		time_ = time_str()
		if config["max_caption_len"] is not None:
			wandb.log({"val_BLEU-4": blue_score})
			if blue_score > best_bleu_score:
				best_bleu_score = blue_score
				if checkpoint_dir is not None:
					best_model = os.path.join(ROOT, f"{checkpoint_dir}/best_bleu4_{time_}.pt")
					torch.save(model.state_dict(), best_model)
				logger.info(f"New best BLEU-4 score: {best_bleu_score:.4f}")
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			epochs_no_improve = 0
			if checkpoint_dir is not None:
				best_model = os.path.join(ROOT, f"{checkpoint_dir}/best_val_{time_}.pt")
				torch.save(model.state_dict(), best_model)
			logger.info(f"New best validation loss: {best_val_loss:.4f}")
		else:
			if config["patience"] is not None:
				epochs_no_improve += 1
				if epochs_no_improve >= config["patience"]:
					logger.info(f"Early stopping after {epoch + 1} epochs")
					break
	# Log training time
	train_time = time.time() - start_time
	logger.info(f"Training completed in {train_time:.2f} seconds")
	wandb.log({"train_time": train_time})
	# Log best model
	if best_model is not None:
		wandb.log_model(path=best_model)
	return best_model


def train_load(model: nn.Module, train_loader: DataLoader, device: torch.device, epoch: int, criterion: nn.Module,
			   optimizer: torch.optim) -> float:
	"""
	Trains the model on the training set for one epoch

	:param model: Model to train
	:param train_loader: DataLoader for the training set
	:param device: Device to run the training on
	:param epoch: Current epoch
	:param criterion: Loss function
	:param optimizer: Optimizer for training
	:return: Total training loss for the epoch
	"""
	train_loss = 0.
	total_tokens = 0
	vocab = train_loader.dataset.vocab if isinstance(train_loader.dataset,
													 FlickerDataset) else train_loader.dataset.dataset.vocab
	pad_idx = vocab.to_idx(PAD)

	model.train()
	batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config["max_epochs"]} [Train]")
	for images, captions, images_id in batch_progress:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		loss, num_tokens = forward_pass(model, images, captions, criterion, pad_idx)
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		if config["gradient_clip"] is not None:
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])  # Gradient clipping
		optimizer.step()

		train_loss += loss.item()
		total_tokens += num_tokens
		batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

	avg_loss = train_loss / total_tokens if total_tokens > 0 else 0
	return avg_loss


def eval_load(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, criterion: nn.Module,
			  smoothing: Any) -> tuple:
	"""
	Evaluates the model on the validation set for one epoch

	:param model: Model to evaluate
	:param val_loader: DataLoader for the validation set
	:param device: Device to run the evaluation on
	:param epoch: Current epoch
	:param criterion: Loss function
	:param smoothing: Smoothing function for BLEU score
	:return: Average validation loss and BLEU score (if calc_bleu is True)
	"""
	val_loss = 0.0
	total_tokens = 0
	df = val_loader.dataset.df if isinstance(val_loader.dataset, FlickerDataset) else val_loader.dataset.dataset.df
	vocab = val_loader.dataset.vocab if isinstance(val_loader.dataset,
												   FlickerDataset) else val_loader.dataset.dataset.vocab
	pad_idx = vocab.to_idx(PAD)
	all_references = []  # List of lists of reference captions
	all_hypothesis = []  # List of generated captions (hypotheses)

	model.eval()
	with torch.no_grad():
		batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config["max_epochs"]} [Val]")
		for images, captions, images_id in batch_progress:
			images = images.to(device)
			captions = captions.to(device)

			loss, num_tokens = forward_pass(model, images, captions, criterion, pad_idx)  # Forward pass

			val_loss += loss.item()
			total_tokens += num_tokens
			batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

			if config["max_caption_len"] is not None:
				# Generate captions
				generated = gen_captions(model, vocab, device, images)
				all_hypothesis.extend(generated)
				# Process ground truth captions
				references = get_references(df, images_id)
				all_references.extend(references)

	avg_loss = val_loss / total_tokens if total_tokens > 0 else 0
	if config["max_caption_len"] is None:
		return avg_loss, None

	return avg_loss, corpus_bleu(all_references, all_hypothesis, smoothing_function=smoothing)


def get_references(df: pd.DataFrame, image_ids: list) -> list:
	"""
	Get references for a list of image ids
	:param df: DataFrame containing image ids and its captions
	:param image_ids: List of image ids to get references for
	:return: List of references for each image id
	"""
	references = []
	for image_id in image_ids:
		captions = df[df["image_id"] == image_id]["caption"].values
		references.append([caption.split() for caption in captions])
	return references


def gen_captions(model: nn.Module, vocab: Vocabulary, device: torch.device, images: list) -> list:
	"""
	Generate captions for a list of images
	:param model: Model to use for caption generation
	:param device: Device to use
	:param images: List of images to generate captions for
	:param vocab: Vocabulary of the dataset
	:return: List of generated captions
	"""
	generated = []
	for image in images:
		image = image.unsqueeze(0).to(device)
		caption = gen_caption(model, image, vocab, config["max_caption_len"], device)
		generated.append(caption.split())
	return generated


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
