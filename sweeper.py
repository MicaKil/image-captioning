import os.path

import torch

import wandb
from constants import ROOT, PAD, CHECKPOINT_DIR, FLICKR8K_DIR
from scripts.dataset.flickr_dataloader import FlickerDataLoader
from scripts.dataset.flickr_dataset import FlickerDataset
from scripts.models.basic import ImageCaptioning
from scripts.test import test
from scripts.train import train
from sweeper_config import DEFAULT_CONFIG, SWEEP_CONFIG, PROJECT, SWEEP_TAGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_config = DEFAULT_CONFIG
sweeper_config = SWEEP_CONFIG

def run_sweep():
	num_workers = 4
	shuffle = True
	pin_memory = True
	save_results = True

	# Initialize wandb run
	wandb_run = wandb.init(project=PROJECT, config=default_config, tags=SWEEP_TAGS)
	wandb_run.define_metric("train_loss", summary="min")
	wandb_run.define_metric("val_loss", summary="min")
	config = wandb_run.config

	# Load datasets
	train_dataset: FlickerDataset = torch.load(os.path.join(ROOT, f"{FLICKR8K_DIR}/train_dataset_s-80_2025-02-10.pt"),
											   weights_only=False)
	val_dataset: FlickerDataset = torch.load(os.path.join(ROOT, f"{FLICKR8K_DIR}/val_dataset_s-10_2025-02-10.pt"),
											 weights_only=False)
	test_dataset: FlickerDataset = torch.load(os.path.join(ROOT, f"{FLICKR8K_DIR}/test_dataset_s-10_2025-02-10.pt"),
											  weights_only=False)
	vocab = train_dataset.dataset.vocab
	pad_idx = vocab.to_idx(PAD)

	# Initialize dataloaders
	train_dataloader = FlickerDataLoader(train_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
	val_dataloader = FlickerDataLoader(val_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
	test_dataloader = FlickerDataLoader(test_dataset, config["batch_size"], num_workers, shuffle, pin_memory)

	# Initialize model and optimizer
	model = ImageCaptioning(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"],
							config["num_layers"], pad_idx, config["freeze_encoder"])

	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
	optimizer = torch.optim.Adam([
		{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
		{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
	])

	print(f"Model:\n{model}")
	print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

	train(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, CHECKPOINT_DIR)
	test(model, test_dataloader, device, save_results)


if __name__ == "__main__":
	sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT)
	print(f"Sweep id: {sweep_id}")
	wandb.agent(sweep_id=sweep_id, function=run_sweep, count=2)
