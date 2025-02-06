import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def train(model: nn.Module,
		  train_loader: DataLoader,
		  val_loader: DataLoader,
		  criterion: nn.Module,
		  optimizer: nn.Module,
		  device: torch.device,
		  max_epochs: int,
		  patience: int,
		  checkpoint_path: str
		  ) -> None:
	"""
	Training loop for the model.

	:param model:
	:param train_loader:
	:param val_loader:
	:param criterion:
	:param optimizer:
	:param device:
	:param max_epochs:
	:param patience:
	:param checkpoint_path:
	:return:
	"""

	logger.info(f"Starting training for {max_epochs} epochs.")

	model = model.to(device)
	best_val_loss = np.inf
	epochs_no_improve = 0

	for epoch in range(max_epochs):
		train_loss = train_load(criterion, device, epoch, max_epochs, model, optimizer, train_loader)
		val_loss = eval_load(criterion, device, epoch, max_epochs, model, val_loader)

		# Calculate epoch metrics
		train_loss = train_loss / len(train_loader.dataset)
		val_loss = val_loss / len(val_loader.dataset)
		logger.info(f"Epoch {epoch + 1} | Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

		# Early stopping and checkpointing
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_no_improve = 0
			torch.save(model.state_dict(), checkpoint_path)
			logger.info(f"Checkpoint saved to {checkpoint_path}")
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				logger.info(f"Early stopping after {epoch + 1} epochs")
				break


def train_load(criterion, device, epoch, max_epochs, model, optimizer, train_loader):
	"""
	Trains for one epoch

	:param criterion:
	:param device:
	:param epoch:
	:param max_epochs:
	:param model:
	:param optimizer:
	:param train_loader:
	:return:
	"""

	model.train()
	train_loss = 0.0
	batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
	for images, captions in batch:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		outputs = model(images, captions[:, :-1])  # Exclude last token
		loss = criterion(
			outputs.view(-1, outputs.size(-1)),
			captions[:, 1:].reshape(-1)  # Exclude first token
		)

		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
		optimizer.step()

		train_loss += loss.item() * images.size(0)
		batch.set_postfix({"loss": loss.item()})
	return train_loss


def eval_load(criterion, device, epoch, max_epochs, model, val_loader):
	# Validation phase
	model.eval()
	val_loss = 0.0
	with torch.no_grad():
		for images, captions in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]"):
			images = images.to(device)
			captions = captions.to(device)

			outputs = model(images, captions[:, :-1])
			loss = criterion(
				outputs.view(-1, outputs.size(-1)),
				captions[:, 1:].reshape(-1)
			)
			val_loss += loss.item() * images.size(0)
	return val_loss
