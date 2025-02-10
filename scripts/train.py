import logging
import os.path
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import ROOT, PAD
from scripts.dataset.flickr_dataset import FlickerDataset
from scripts.utils import time_str

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


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
	config = wandb.config
	wandb.watch(model, criterion=criterion, log="all", log_freq=100)

	logger.info(
		f"Start training model {model.__class__.__name__} for {config["max_epochs"]} {"epoch" if config["max_epochs"] == 1 else "epochs"}"
	)
	start_time = time.time()

	best_val_loss = np.inf
	best_model = None
	epochs_no_improve = 0

	model = model.to(device)
	for epoch in range(config["max_epochs"]):
		avg_train_loss = train_load(model, train_loader, device, epoch, criterion, optimizer)
		avg_val_loss = eval_load(model, val_loader, device, epoch, criterion)

		logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
		wandb.log({
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"val_loss": avg_val_loss
		})

		# Early stopping and checkpointing
		time_ = time_str()
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
					wandb.log({"early_stopping": True})
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
	config = wandb.config

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

	return train_loss / total_tokens if total_tokens > 0 else 0


def eval_load(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int,
			  criterion: nn.Module) -> float:
	"""
	Evaluates the model on the validation set for one epoch

	:param model: Model to evaluate
	:param val_loader: DataLoader for the validation set
	:param device: Device to run the evaluation on
	:param epoch: Current epoch
	:param criterion: Loss function
	:return: Average validation loss and BLEU score (if calc_bleu is True)
	"""
	config = wandb.config

	val_loss = 0.0
	total_tokens = 0
	vocab = val_loader.dataset.vocab if isinstance(val_loader.dataset,
												   FlickerDataset) else val_loader.dataset.dataset.vocab

	model.eval()
	with torch.no_grad():
		batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config["max_epochs"]} [Val]")
		for images, captions, images_id in batch_progress:
			images = images.to(device)
			captions = captions.to(device)
			loss, num_tokens = forward_pass(model, images, captions, criterion, vocab.to_idx(PAD))  # Forward pass
			val_loss += loss.item()
			total_tokens += num_tokens
			batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

	return val_loss / total_tokens if total_tokens > 0 else 0


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
