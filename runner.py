import os.path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, Subset
from wandb.sdk.wandb_run import Run

import wandb
from config import logger
from constants import ROOT, FLICKR8K_CSV_FILE, FLICKR8K_IMG_DIR, CHECKPOINT_DIR, PAD, FLICKR8K_DIR
from runner_config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY, RUN_CONFIG, RUN_TAGS, PROJECT
from scripts.dataset.flickr_dataloader import FlickrDataLoader
from scripts.dataset.flickr_dataset import FlickrDataset, load_captions
from scripts.dataset.vocabulary import Vocabulary
from scripts.models.basic import ImageCaptioning
from scripts.test import test
from scripts.train import train
from scripts.utils import date_str, get_vocab


def run(run_config: dict, run_tags: list, create_dataset: bool, train_model: bool, test_model: bool,
		model_path: Optional[str], save_results: bool):
	"""
	Run the training and testing pipeline
	:param run_config: A dictionary the wandb run configuration
	:param run_tags: A list of tags to be added to the wandb run
	:param create_dataset: Whether to create a new dataset or load an existing one. Saves the dataset to disk if a new one is created
	:param train_model: Whether to train the model
	:param test_model: Whether to test the model
	:param model_path: Path to the model to be loaded
	:param save_results: Whether to save the test results
	:return:
	"""
	date = date_str()
	init_wandb_run(project=PROJECT, tags=run_tags, config=run_config)
	config = wandb.config

	# create or load dataset
	if create_dataset:
		ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
		img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
		df_captions = load_captions(ann_file)
		vocab = Vocabulary(config["vocab"]["freq_threshold"], df_captions["caption"])
		full_dataset = FlickrDataset(img_dir, df_captions, vocab, transform=TRANSFORM)

		total_size = len(full_dataset)
		train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
		val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
		test_size = total_size - train_size - val_size
		train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

		save_datasets(full_dataset, test_dataset, train_dataset, val_dataset, date)  # save new datasets
		log_datasets(date)  # log datasets to wandb
	else:
		train_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s80_2025-02-10.pt"),
								   weights_only=False)
		val_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s10_2025-02-10.pt"),
								 weights_only=False)
		test_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/test_dataset_s10_2025-02-10.pt"),
								  weights_only=False)

	# create or load model
	vocab = get_vocab(train_dataset)
	pad_idx = vocab.to_idx(PAD)

	model = ImageCaptioning(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"],
							config["num_layers"], pad_idx, config["freeze_encoder"])
	if model_path is not None:
		logger.info(f"Loading model from {model_path}")
		model.load_state_dict(torch.load(os.path.join(ROOT, model_path), weights_only=True))

	print(f"Model:\n{model}")
	print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

	if train_model:
		train_dataloader = FlickrDataLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		val_dataloader = FlickrDataLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
		optimizer = Adam([
			{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
			{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
		])
		scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"],
									  patience=config["scheduler"]["patience"])
		train(model, train_dataloader, val_dataloader, DEVICE, criterion, optimizer, scheduler, CHECKPOINT_DIR)

	if test_model:
		test_dataloader = FlickrDataLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		test(model, test_dataloader, DEVICE, save_results)

	wandb.finish()


def init_wandb_run(project: str, tags: list, config: dict) -> Run:
	"""
	Initialize wandb run
	:param project: Project name
	:param tags: List of tags
	:param config: Configuration dictionary
	:return: Wandb run
	"""
	wandb_run = wandb.init(project=project, tags=tags, config=config)
	wandb_run.define_metric("train_loss", summary="min")
	wandb_run.define_metric("val_loss", summary="min")
	wandb_run.define_metric("val_BLEU-4", summary="max")
	wandb_run.define_metric("test_BLEU-1", summary="max")
	wandb_run.define_metric("test_BLEU-2", summary="max")
	wandb_run.define_metric("test_BLEU-4", summary="max")
	wandb_run.define_metric("test_CIDEr", summary="max")
	return wandb_run


def log_datasets(date: str):
	"""
	Log datasets to wandb
	:param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
	:return:
	"""
	config = wandb.config
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset",
					   metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt")
	)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset",
					   metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/train_dataset_{date}_s{config["dataset"]["split"]["train"]}.pt")
	)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset",
					   metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/val_dataset_{date}_s{config["dataset"]["split"]["val"]}.pt")
	)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset",
					   metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/test_dataset_{date}_s{config["dataset"]["split"]["test"]}.pt")
	)


def log_dataset(artifact: wandb.Artifact, dataset_path: str):
	"""
	Log dataset to wandb
	:param artifact: Wandb artifact
	:param dataset_path: Path to the dataset
	"""
	artifact.add_file(dataset_path)
	wandb.log_artifact(artifact)


def save_datasets(full_dataset: FlickrDataset, test_dataset: Union[FlickrDataset | Subset],
				  train_dataset: Union[FlickrDataset | Subset], val_dataset: Union[FlickrDataset | Subset], date: str):
	"""
	Save datasets to disk
	:param full_dataset: Complete dataset
	:param test_dataset: Test dataset or subset
	:param train_dataset: Training dataset or subset
	:param val_dataset: Validation dataset or subset
	:param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
	:return:
	"""
	config = wandb.config
	torch.save(full_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt"))
	torch.save(
		train_dataset,
		os.path.join(ROOT, f"{FLICKR8K_DIR}/train_dataset_{date}_s{config["dataset"]["split"]["train"]}.pt")
	)
	torch.save(
		val_dataset,
		os.path.join(ROOT, f"{FLICKR8K_DIR}/val_dataset_{date}_s{config["dataset"]["split"]["val"]}.pt")
	)
	torch.save(
		test_dataset,
		os.path.join(ROOT, f"{FLICKR8K_DIR}/test_dataset_{date}_s{config["dataset"]["split"]["test"]}.pt")
	)


if __name__ == "__main__":
	# model_path_ = os.path.join(ROOT, f"{CHECKPOINT_DIR}/best_val_2025-02-09_23-07.pt")
	wandb.teardown()
	run(run_config=RUN_CONFIG, run_tags=RUN_TAGS, create_dataset=True, train_model=True, test_model=True,
		model_path=None, save_results=True)
