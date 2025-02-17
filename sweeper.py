import os.path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from constants import ROOT, PAD, CHECKPOINT_DIR, BASIC_RESULTS
from runner import init_wandb_run
from scripts.dataset.flickr_dataloader import FlickrDataLoader
from scripts.models.basic import ImageCaptioning
from scripts.test import test
from scripts.train import train
from scripts.utils import get_vocab
from sweeper_config import DEFAULT_CONFIG, SWEEP_CONFIG, PROJECT, SWEEP_TAGS, TRAIN_PATH, VAL_PATH, TEST_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_config = DEFAULT_CONFIG
sweep_tags = SWEEP_TAGS

def run_sweep():
	num_workers = 4
	shuffle = True
	pin_memory = True
	eval_bleu4 = False
	eval_bleu4_step = 10

	# Initialize wandb run
	init_wandb_run(project=PROJECT, tags=sweep_tags, config=default_config)
	config = wandb.config

	# Load datasets
	train_dataset = torch.load(str(os.path.join(ROOT, TRAIN_PATH)), weights_only=False)
	val_dataset = torch.load(str(os.path.join(ROOT, VAL_PATH)), weights_only=False)
	test_dataset = torch.load(str(os.path.join(ROOT, TEST_PATH)), weights_only=False)
	vocab = get_vocab(train_dataset)
	pad_idx = vocab.to_idx(PAD)

	# Initialize dataloaders
	train_dataloader = FlickrDataLoader(train_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
	val_dataloader = FlickrDataLoader(val_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
	test_dataloader = FlickrDataLoader(test_dataset, config["batch_size"], num_workers, shuffle, pin_memory)

	# Initialize model and optimizer
	model = ImageCaptioning(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"],
							config["num_layers"], pad_idx, config["freeze_encoder"])

	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
	optimizer = torch.optim.Adam([
		{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
		{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
	])
	scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"],
								  patience=config["scheduler"]["patience"])

	print(f"Model:\n{model}")
	print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

	best_val_model, _, _ = train(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler,
								 CHECKPOINT_DIR, eval_bleu4, eval_bleu4_step)
	# test last model
	test(model, test_dataloader, device, BASIC_RESULTS, "last-model")
	# test model with the best validation loss
	best = ImageCaptioning(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"],
						   config["num_layers"], pad_idx, config["freeze_encoder"])
	best.load_state_dict(torch.load(best_val_model, weights_only=True))
	test(best, test_dataloader, device, BASIC_RESULTS, "best-model")


if __name__ == "__main__":
	sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT)
	print(f"Sweep id: {sweep_id}")
	wandb.agent(sweep_id=sweep_id, function=run_sweep, count=2)
