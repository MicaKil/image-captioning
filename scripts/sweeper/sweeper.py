import os.path

import pandas as pd
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import logger
from constants import ROOT, PAD, CHECKPOINT_DIR, FLICKR_VAL_CSV, FLICKR_TRAIN_CSV, FLICKR_TEST_CSV, FLICKR8K_IMG_DIR, RESULTS_DIR
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.dataset import CaptionDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.runner import init_wandb_run, get_model
from scripts.runner.config import TRANSFORM
from scripts.sweeper.sweeper_config import DEFAULT_CONFIG, SWEEP_CONFIG, PROJECT, TAGS
from scripts.test import test
from scripts.train import train

default_config = DEFAULT_CONFIG
sweep_tags = TAGS


def run_sweep():
    num_workers = 4
    shuffle = True
    pin_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb run
    init_wandb_run(project=PROJECT, tags=sweep_tags, config=default_config)
    config = wandb.config

    # Load datasets
    img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
    train_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_TRAIN_CSV)))
    val_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_VAL_CSV)))
    test_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_TEST_CSV)))
    vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])
    train_dataset = CaptionDataset(img_dir, train_df, vocab, transform=TRANSFORM)
    val_dataset = CaptionDataset(img_dir, val_df, vocab, transform=TRANSFORM)
    test_dataset = CaptionDataset(img_dir, test_df, vocab, transform=TRANSFORM)

    vocab = train_dataset.vocab
    pad_idx = vocab.to_idx(PAD)

    train_dataloader = CaptionLoader(train_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
    val_dataloader = CaptionLoader(val_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
    test_dataloader = CaptionLoader(test_dataset, config["batch_size"], num_workers, shuffle, pin_memory)

    model = get_model(config, vocab, pad_idx)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
        {"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
    ])
    scheduler = None
    if config["scheduler"] is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"])

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary["trainable_parameters"] = parameter_count
    logger.info(f"Number of trainable parameters: {parameter_count}")

    train(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, CHECKPOINT_DIR + config["model"], True, config,
          resume_checkpoint)
    test(model, test_dataloader, device, RESULTS_DIR + config["model"], "last-model", True, config)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT)
    print(f"Sweep id: {sweep_id}")
    wandb.agent(sweep_id=sweep_id, function=run_sweep, count=20)
