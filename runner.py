import os.path

import torch
from torch.utils.data import random_split
from wandb.sdk.wandb_run import Run

import wandb
from config import logger
from constants import ROOT, FLICKR8K_CSV_FILE, FLICKR8K_IMG_DIR, CHECKPOINT_DIR, PAD, FLICKR8K_DIR, PROJECT
from runner_config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY, RUN_CONFIG, RUN_TAGS
from scripts.dataset.flickr_dataloader import FlickerDataLoader
from scripts.dataset.flickr_dataset import FlickerDataset, load_captions
from scripts.dataset.vocabulary import Vocabulary
from scripts.models.basic import ImageCaptioning
from scripts.test import test
from scripts.train import train
from scripts.utils import date_str


def run(run_config: dict, run_tags: list, create_dataset=False, train_model=True, test_model=True,
		model_path: str = None, save_results=True):
	date = date_str()
	init_wandb_run(project=PROJECT, tags=run_tags, config=run_config)
	config = wandb.config

	# create or load dataset
	if create_dataset:
		ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
		img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
		df_captions = load_captions(ann_file)
		vocab = Vocabulary(config["vocab"]["freq_threshold"], df_captions["caption"])
		full_dataset = FlickerDataset(img_dir, df_captions, vocab, transform=TRANSFORM)

		torch.save(full_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt"))

		total_size = len(full_dataset)
		train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
		val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
		test_size = total_size - train_size - val_size

		train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

		# save new datasets
		torch.save(train_dataset,
				   os.path.join(ROOT, f"{FLICKR8K_DIR}/train_dataset_s{config["dataset"]["split"]["train"]}_{date}.pt"))
		torch.save(val_dataset,
				   os.path.join(ROOT, f"{FLICKR8K_DIR}/val_dataset_s{config["dataset"]["split"]["val"]}_{date}.pt"))
		torch.save(test_dataset,
				   os.path.join(ROOT, f"{FLICKR8K_DIR}/test_dataset_s{config["dataset"]["split"]["test"]}_{date}.pt"))

		# log datasets to wandb
		log_dataset(wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset",
								   metadata={"version": config["dataset"]["version"]}),
					os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt"))
		log_dataset(wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset",
								   metadata={"version": config["dataset"]["version"]}),
					os.path.join(ROOT,
								 f"{FLICKR8K_DIR}/train_dataset_s{config["dataset"]["split"]["train"]}_{date}.pt"))
		log_dataset(wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset",
								   metadata={"version": config["dataset"]["version"]}),
					os.path.join(ROOT, f"{FLICKR8K_DIR}/val_dataset_s{config["dataset"]["split"]["val"]}_{date}.pt"))
		log_dataset(wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset",
								   metadata={"version": config["dataset"]["version"]}),
					os.path.join(ROOT, f"{FLICKR8K_DIR}/test_dataset_s{config["dataset"]["split"]["test"]}_{date}.pt"))
	else:
		train_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s80_2025-02-10.pt"),
								   weights_only=False)
		val_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s10_2025-02-10.pt"),
								 weights_only=False)
		test_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/test_dataset_s10_2025-02-10.pt"),
								  weights_only=False)

	# create or load model
	vocab = train_dataset.vocab
	pad_idx = vocab.to_idx(PAD)

	model = ImageCaptioning(config["embed_size"], config["hidden_size"], config["vocab_size"], config["dropout"],
							config["num_layers"], pad_idx, config["freeze_encoder"])
	if model_path is not None:
		logger.info(f"Loading model from {model_path}")
		model.load_state_dict(torch.load(os.path.join(ROOT, model_path), weights_only=True))

	print(f"Model:\n{model}")
	print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

	if train_model:
		train_dataloader = FlickerDataLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		val_dataloader = FlickerDataLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
		optimizer = torch.optim.Adam([
			{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
			{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
		])
		train(model, train_dataloader, val_dataloader, DEVICE, criterion, optimizer, CHECKPOINT_DIR)

	if test_model:
		test_dataloader = FlickerDataLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		test(model, test_dataloader, DEVICE, save_results)

	wandb.finish()


def init_wandb_run(project, tags, config) -> Run:
	"""
	Initialize wandb run
	:return: Wandb run
	"""
	wandb_run = wandb.init(
		project=project,
		tags=tags,
		config=config
	)
	wandb_run.define_metric("train_loss", summary="min")
	wandb_run.define_metric("val_loss", summary="min")
	wandb_run.define_metric("val_BLEU-4", summary="max")
	wandb_run.define_metric("test_BLEU-1", summary="max")
	wandb_run.define_metric("test_BLEU-2", summary="max")
	wandb_run.define_metric("test_BLEU-4", summary="max")
	wandb_run.define_metric("test_CIDEr", summary="max")
	return wandb_run


def log_dataset(artifact: wandb.Artifact, dataset_path: str):
	"""
	Log dataset to wandb
	:param artifact: Wandb artifact
	:param dataset_path: Path to the dataset
	"""
	artifact.add_file(dataset_path)
	wandb.log_artifact(artifact)


if __name__ == "__main__":
	# model_path_ = os.path.join(ROOT, f"{CHECKPOINT_DIR}/best_val_2025-02-09_23-07.pt")
	wandb.teardown()
	run(run_config=RUN_CONFIG, run_tags=RUN_TAGS, create_dataset=False, train_model=True, test_model=True,
		model_path=None, save_results=True)
