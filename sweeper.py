import os.path

import torch

import wandb
from constants import ROOT, PAD, CHECKPOINT_DIR, PROJECT
from scripts.dataset.flickr_dataloader import FlickerDataLoader
from scripts.dataset.flickr_dataset import FlickerDataset
from scripts.models.basic import ImageCaptioning
from scripts.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_config = {
	"encoder": "resnet50",
	"decoder": "LSTM",
	"batch_size": 16,
	"embed_size": 128,
	"hidden_size": 256,
	"num_layers": 1,
	"dropout": 0.2,
	"freeze_encoder": True,
	"encoder_lr": 1e-4,
	"decoder_lr": 1e-4,
	"criterion": "CrossEntropyLoss",
	"optimizer": "Adam",
	"max_epochs": 1,
	"patience": None,
	"gradient_clip": None,
	"dataset": "flickr8k",
	"dataset_version": "2025-02-10",
	"vocab_threshold": 3,
	"vocab_size": 4107,
}

sweep_config = {
	"project": PROJECT,
	"method": "bayes",
	"metric": {
		"name": "val_loss",
		"goal": "minimize"
	},
	"parameters": {
		# architecture
		"hidden_size": {
			"value": 512
		},
		"embed_size": {
			"value": 256
		},
		# regularisation
		"dropout": {
			"values": [0.4, 0.5]
		},
		"batch_size": {
			"value": 32
		}
	}
}



def run_sweep():
	wandb_run = wandb.init(project=PROJECT, config=default_config, job_type="train",
						   tags=["basic", "flickr8k", "config-test"])
	wandb_run.define_metric("train_loss", summary="min")
	wandb_run.define_metric("val_loss", summary="min")
	config = wandb_run.config

	train_dataset: FlickerDataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s-80_2025-02-10.pt"),
											   weights_only=False)
	val_dataset: FlickerDataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s-10_2025-02-10.pt"),
											 weights_only=False)
	vocab = train_dataset.dataset.vocab
	pad_idx = vocab.to_idx(PAD)
	model = ImageCaptioning(config["embed_size"], config["hidden_size"], config["vocab_size"], config["dropout"],
							config["num_layers"], pad_idx, config["freeze_encoder"])

	train_dataloader = FlickerDataLoader(train_dataset, config["batch_size"], 4, True, True)
	val_dataloader = FlickerDataLoader(val_dataset, config["batch_size"], 4, True, True)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
	optimizer = torch.optim.Adam([
		{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
		{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
	])

	train(model, train_dataloader, val_dataloader, device, criterion, optimizer, CHECKPOINT_DIR, 30)


if __name__ == "__main__":
	sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT)
	print(f"Sweep id: {sweep_id}")
	wandb.agent(sweep_id=sweep_id, function=run_sweep, count=2)
