import os.path
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from wandb.sdk.wandb_run import Run

import scripts.models.basic as basic
import wandb
from config import logger
from constants import ROOT, FLICKR8K_IMG_DIR, CHECKPOINT_DIR, PAD, FLICKR8K_DIR, FLICKR8K_ANN_FILE, BASIC_RESULTS, TRAIN_CSV, VAL_CSV, TEST_CSV
from runner_config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY, RUN_CONFIG, PROJECT, RUN_TAGS
from scripts.dataset.flickr_dataloader import FlickrDataLoader
from scripts.dataset.flickr_dataset import FlickrDataset, split_dataframe, load_captions
from scripts.dataset.vocabulary import Vocabulary
from scripts.models.image_captioning import ImageCaptioning
from scripts.test import test
from scripts.train import train
from scripts.utils import date_str


def run(run_config: dict, use_wandb: bool, run_tags: list, create_ds: bool, save_ds: bool, train_model: bool, test_model: bool,
        saved_model: Optional[tuple[str, str]], save_dir: Optional[str]):
	"""
	Run the training and testing pipeline
	:param run_config: A dictionary the wandb run configuration
	:param use_wandb:
	:param run_tags: A list of tags to be added to the wandb run
	:param create_ds: Whether to create a new dataset or load an existing one. Saves the dataset to disk if a new one is created
	:param save_ds:
	:param train_model: Whether to train the model
	:param test_model: Whether to test the model
	:param saved_model: Tuple containing the model path and the model tag. If not None, the model is loaded from the path.
	:param save_dir: If not None, the test results are saved to this directory
	:return:
	"""
	date = date_str()
	if use_wandb:
		init_wandb_run(project=PROJECT, tags=run_tags, config=run_config)
		config = wandb.config
	else:
		config = run_config

	# create or load dataset
	img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
	if create_ds:
		df_captions = load_captions(str(os.path.join(ROOT, FLICKR8K_ANN_FILE)), True)
		total_size = len(df_captions["image_id"].unique())
		train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
		val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
		test_size = total_size - train_size - val_size
		train_df, val_df, test_df = split_dataframe(df_captions, [train_size, val_size, test_size])
	else:
		train_df = pd.read_csv(str(os.path.join(ROOT, TRAIN_CSV)))
		val_df = pd.read_csv(str(os.path.join(ROOT, VAL_CSV)))
		test_df = pd.read_csv(str(os.path.join(ROOT, TEST_CSV)))

	vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])
	train_dataset = FlickrDataset(img_dir, train_df, vocab, transform=TRANSFORM)
	val_dataset = FlickrDataset(img_dir, val_df, vocab, transform=TRANSFORM)
	test_dataset = FlickrDataset(img_dir, test_df, vocab, transform=TRANSFORM)

	if save_ds:
		save_df(config, date, test_df, train_df, val_df)
		save_datasets(None, train_dataset, val_dataset, test_dataset, date, config)  # save new datasets
		if use_wandb:
			log_datasets(date, False)

	# Initialize model
	pad_idx = vocab.to_idx(PAD)
	encoder = basic.Encoder(config["embed_size"], config["freeze_encoder"])
	decoder = basic.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
	model = ImageCaptioning(encoder, decoder)

	if saved_model is not None:
		# Load model from saved model
		logger.info(f"Loading model from {saved_model}")
		model.load_state_dict(torch.load(os.path.join(ROOT, saved_model[0]), weights_only=True))
		if test_model:
			test_dataloader = FlickrDataLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
			test(model, test_dataloader, DEVICE, save_dir, saved_model[1], use_wandb, run_config)
		if use_wandb:
			wandb.finish()
		return

	parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if use_wandb:
		wandb.run.summary["trainable_parameters"] = parameter_count
	logger.info(f"Number of trainable parameters: {parameter_count}")

	if train_model:
		train_dataloader = FlickrDataLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		val_dataloader = FlickrDataLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
		optimizer = Adam([
			{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
			{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
		])
		scheduler = None
		if config["scheduler"] is not None:
			scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"])
		best_val_model, best_val_info, _ = train(model, train_dataloader, val_dataloader, DEVICE, criterion, optimizer, scheduler, CHECKPOINT_DIR,
		                                         use_wandb, run_config)
		if test_model:
			test_dataloader = FlickrDataLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
			test(model, test_dataloader, DEVICE, save_dir, "last-model", use_wandb, run_config)
			if use_wandb:
				wandb.finish()
			if best_val_model is not None:
				init_wandb_run(project=PROJECT, tags=run_tags, config=run_config)
				best = ImageCaptioning(
					basic.Encoder(config["embed_size"], config["freeze_encoder"]),
					basic.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
				)
				best.load_state_dict(torch.load(best_val_model, weights_only=True))
				test(best, test_dataloader, DEVICE, save_dir, "best-model", use_wandb, run_config)
				if use_wandb:
					wandb.log(best_val_info)
					wandb.log_model(path=best_val_model)

	if use_wandb:
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
	wandb_run.define_metric("epoch", summary="max")
	return wandb_run


def save_df(config, date, test_df, train_df, val_df):
	train_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"train_{date}_{config['dataset']['split']['train']}.csv")), header=["image_id", "caption"],
	                index=False)
	val_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"val_{date}_{config['dataset']['split']['val']}.csv")), header=["image_id", "caption"],
	              index=False)
	test_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"test_{date}_{config['dataset']['split']['test']}.csv")), header=["image_id", "caption"],
	               index=False)


def save_datasets(full_dataset: Optional[FlickrDataset], train_dataset: Union[FlickrDataset | Subset], val_dataset: Union[FlickrDataset | Subset],
                  test_dataset: Union[FlickrDataset | Subset], date: str, config: dict):
	"""
	Save datasets to disk
	:param config:
	:param full_dataset: Complete dataset
	:param test_dataset: Test dataset or subset
	:param train_dataset: Training dataset or subset
	:param val_dataset: Validation dataset or subset
	:param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
	:return:
	"""
	if full_dataset is not None:
		torch.save(full_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt"))
	torch.save(train_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/train_{date}_{config["dataset"]["split"]["train"]}.pt"))
	torch.save(val_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/val_{date}_{config["dataset"]["split"]["val"]}.pt"))
	torch.save(test_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/test_{date}_{config["dataset"]["split"]["test"]}.pt"))


def log_datasets(date: str, has_full_ds: bool):
	"""
	Log datasets to wandb
	:param has_full_ds:
	:param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
	:return:
	"""
	config = wandb.config
	if has_full_ds:
		log_dataset(
			wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
			os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt")
		)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/train_{date}_{config["dataset"]["split"]["train"]}.pt")
	)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/val_{date}_{config["dataset"]["split"]["val"]}.pt")
	)
	log_dataset(
		wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
		os.path.join(ROOT, f"{FLICKR8K_DIR}/test_{date}_{config["dataset"]["split"]["test"]}.pt")
	)


def log_dataset(artifact: wandb.Artifact, dataset_path: str):
	"""
	Log dataset to wandb
	:param artifact: Wandb artifact
	:param dataset_path: Path to the dataset
	"""
	artifact.add_file(dataset_path)
	wandb.log_artifact(artifact)


def state_dicts_equal(state_a, state_b) -> bool:
	# Check keys match
	if state_a.keys() != state_b.keys():
		return False

	# Check tensor values match
	for key in state_a:
		if not torch.equal(state_a[key], state_b[key]):
			return False

	return True


if __name__ == "__main__":
	wandb.teardown()
	run(run_config=RUN_CONFIG, use_wandb=True, run_tags=RUN_TAGS, create_ds=False, save_ds=False, train_model=True, test_model=True, saved_model=None,
	    save_dir=BASIC_RESULTS)
