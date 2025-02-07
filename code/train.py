import logging

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption import gen_caption
from dataset.vocabulary import Vocabulary
from utils import time_str

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def train(model: nn.Module,
		  train_loader: DataLoader,
		  val_loader: DataLoader,
		  criterion: nn.Module,
		  optimizer: nn.Module,
		  device: torch.device,
		  vocab: Vocabulary,
		  max_epochs: int,
		  patience: int,
		  checkpoint_dir: str,
		  max_caption_len) -> None:
	"""
	Training loop for the model.

	:param model:
	:param train_loader:
	:param val_loader:
	:param criterion:
	:param optimizer:
	:param device:
	:param vocab:
	:param max_epochs:
	:param patience:
	:param checkpoint_dir:
	:param max_caption_len:
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
										 max_caption_len, smoothing)

		# Calculate epoch metrics
		train_loss = train_loss / len(train_loader.dataset)
		val_loss = val_loss / len(val_loader.dataset)
		logger.info(f"Epoch {epoch + 1} | Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

		# Early stopping and checkpointing
		if blue_score > best_bleu_score:
			best_bleu_score = blue_score
			torch.save(model.state_dict(), f"{checkpoint_dir}/best_bleu_{time_str()}.pt")
			logger.info(f"New best BLEU score: {best_bleu_score:.4f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_no_improve = 0
			torch.save(model.state_dict(), f"{checkpoint_dir}/best_val_loss_{time_str()}.pt")
			logger.info(f"New best validation loss: {best_val_loss:.4f}")
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				logger.info(f"Early stopping after {epoch + 1} epochs")
				break


def train_load(criterion: nn.Module,
			   device: torch.device,
			   epoch: int,
			   max_epochs: int,
			   model: nn.Module,
			   optimizer: nn.Module,
			   train_loader: DataLoader):
	"""
	Trains the model on the training set for one epoch

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
	for images, captions, images_id in batch:
		images = images.to(device)
		captions = captions.to(device)

		# Forward pass
		loss = forward_pass(captions, criterion, images, model)

		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
		optimizer.step()

		train_loss += loss.item() * images.size(0)
		batch.set_postfix({"loss": loss.item()})
	return train_loss


def eval_load(criterion, device, epoch, max_epochs, model, val_loader, vocab, max_caption_len, smoothing):
	"""
	Evaluates the model on the validation set for one epoch

	:param criterion:
	:param device:
	:param epoch:
	:param max_epochs:
	:param model:
	:param vocab:
	:param val_loader:
	:param max_caption_len:
	:param smoothing:
	:return:
	"""

	model.eval()
	val_loss = 0.0
	all_references = []  # List of lists of reference captions
	all_hypotheses = []  # List of generated captions
	df = val_loader.dataset.df  # DataFrame containing image IDs and captions

	with torch.no_grad():
		for images, captions, images_id in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]"):
			images = images.to(device)
			captions = captions.to(device)

			# Forward pass
			val_loss = forward_pass(captions, criterion, images, model)

			# Generate captions for BLEU
			captions_for_bleu(all_hypotheses, all_references, device, df, images, images_id, max_caption_len, model,
							  vocab)

	val_loss /= len(val_loader.dataset)
	blue_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing)

	return val_loss, blue_score


def captions_for_bleu(all_hypotheses, all_references, device, df, images, images_id, max_caption_len, model, vocab):
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


def forward_pass(captions, criterion, images, model):
	outputs = model(images, captions[:, :-1])  # Exclude last token
	loss = criterion(
		outputs.view(-1, outputs.size(-1)),
		captions[:, 1:].reshape(-1)  # Exclude first token
	)
	return loss
