import os.path

import numpy as np
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import logger
from configs.runner_config import TRANSFORM
from constants import ROOT, PATH_ALVARITO, PAD, UNK, SOS
from scripts import test
from scripts.caption import gen_caption, preprocess_image
from scripts.dataset.vocabulary import Vocabulary
from scripts.utils import time_str, get_vocab, get_dataset


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler, checkpoint_dir: str, use_wandb: bool, run_config: dict) -> tuple[str | None, dict, str | None]:
    """
    Training loop for the model.

    :param model: The model to train
    :param train_loader: DataLoader for the training set
    :param val_loader: DataLoader for the validation set
    :param device: Device to run the training on
    :param criterion: Loss function
    :param optimizer: Optimizer for training
    :param scheduler: Learning rate scheduler
    :param checkpoint_dir: Directory to save the best model
    :param use_wandb: Whether to use Weights & Biases for logging
    :param run_config: Configuration for the run
    :return: Path to the best model
    """
    if use_wandb:
        config = wandb.config
        wandb.watch(model, criterion=criterion, log="all")
    else:
        config = run_config

    if config["validation"]["bleu4"] and (config["validation"]["bleu4_step"] is None or config["validation"]["bleu4_step"] < 1):
        raise ValueError("eval_bleu4_step must be greater than 0 if eval_bleu4 is True")

    logger.info(f"Start training model {model.__class__.__name__} for {config["max_epochs"]} {"epoch" if config["max_epochs"] == 1 else "epochs"}")

    avg_val_loss = -1
    metric = dict()
    cur_lr = (config["encoder_lr"], config["decoder_lr"])
    best_val_loss = np.inf
    best_val_info = None
    best_val_model = None
    last_model_pth = None
    epochs_no_improve = 0

    model = model.to(device)
    for epoch in range(config["max_epochs"]):
        avg_train_loss = train_load(model, train_loader, device, epoch, criterion, optimizer, use_wandb, run_config)
        avg_val_loss, val_bleu4 = eval_load(model, val_loader, device, epoch, criterion, use_wandb, run_config)
        if val_bleu4 is not None:
            logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val BLEU-4 = {val_bleu4:.4f}")
        else:
            logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        metric = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        if config["validation"]["bleu4"]:
            metric["val_BLEU-4"] = val_bleu4
        if use_wandb:
            wandb.log(metric)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            cur_lr = scheduler.get_last_lr()
            if use_wandb:
                wandb.log({"encoder_lr": cur_lr[0], "decoder_lr": cur_lr[1]})

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_val_info, best_val_model = checkpoint(model, checkpoint_dir, best_val_loss, best_val_model, cur_lr, epoch)
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if config["patience"] is not None:
                if epochs_no_improve >= config["patience"]:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            if scheduler is not None:
                if (epochs_no_improve - 1) > 0 and (epochs_no_improve - 1) % config["scheduler"]["patience"] == 0:
                    logger.info(f"Reducing learning rate. Encoder LR: {cur_lr[0]}, Decoder LR: {cur_lr[1]}")

    # Log last model
    if avg_val_loss != -1:
        last_model_pth = os.path.join(ROOT, f"{checkpoint_dir}/last_model_{time_str()}_{str(round(avg_val_loss, 4)).replace('.', '-')}.pt")
        torch.save(model.state_dict(), last_model_pth)
        if use_wandb:
            wandb.log_model(path=last_model_pth)
    logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")

    # check best_val_model != last_model
    if best_val_info["epoch"] == metric["epoch"]:
        # if they are the same, then the best model is the last model
        logger.info("Best model is the last model")
        best_val_model = None
        best_val_info = None
    return best_val_model, best_val_info, last_model_pth


def train_load(model: nn.Module, train_loader: DataLoader, device: torch.device, epoch: int, criterion: nn.Module, optimizer: torch.optim, use_wandb,
               run_config) -> float:
    """
    Trains the model on the training set for one epoch

    :param model: Model to train
    :param train_loader: DataLoader for the training set
    :param device: Device to run the training on
    :param epoch: Current epoch
    :param criterion: Loss function
    :param optimizer: Optimizer for training
    :param use_wandb: Whether to use Weights & Biases for logging
    :param run_config: Configuration for the run
    :return: Total training loss for the epoch
    """
    if use_wandb:
        config = wandb.config
    else:
        config = run_config

    train_loss = 0.
    total_tokens = 0
    vocab = get_vocab(train_loader)

    batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config["max_epochs"]} [Train]")
    model.train()
    for images, captions, images_id in batch_progress:
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        loss, num_tokens = forward_pass(model, images, captions, criterion, vocab)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config["gradient_clip"] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])
        optimizer.step()

        train_loss += loss.item()
        total_tokens += num_tokens
        batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

    return train_loss / total_tokens if total_tokens > 0 else 0


def eval_load(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, criterion: nn.Module, use_wandb,
              run_config) -> tuple[float | int, float | None]:
    """
    Evaluates the model on the validation set for one epoch

    :param model: Model to evaluate
    :param val_loader: DataLoader for the validation set
    :param device: Device to run the evaluation on
    :param epoch: Current epoch
    :param criterion: Loss function
    :param use_wandb: Whether to use Weights & Biases for logging
    :param run_config: Configuration for the run
    :return: Average validation loss and BLEU score (if calc_bleu is True)
    """
    if use_wandb:
        config = wandb.config
    else:
        config = run_config

    # for BLEU score
    val_ble4 = None
    all_hypotheses = []
    all_references = []
    df = get_dataset(val_loader).df
    smoothing = SmoothingFunction().method1
    calc_bleu4 = config["validation"]["bleu4"] and (epoch == 0 or (epoch + 1) % config["validation"]["bleu4_step"] == 0)

    val_loss = 0.0
    total_tokens = 0
    vocab = get_vocab(val_loader)
    model.eval()
    with torch.no_grad():
        batch_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config["max_epochs"]} [Val]")
        for images, captions, images_id in batch_progress:
            images = images.to(device)
            captions = captions.to(device)
            loss, num_tokens = forward_pass(model, images, captions, criterion, vocab)  # Forward pass
            val_loss += loss.item()
            total_tokens += num_tokens

            if calc_bleu4:
                generated = test.gen_captions(model, vocab, device, images, use_wandb, run_config)
                all_hypotheses.extend(generated)
                references = test.get_references(df, images_id)
                all_references.extend(references)

            batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

    if calc_bleu4:
        val_ble4 = test.get_bleu4_score(all_hypotheses, all_references, smoothing)
        if use_wandb:
            wandb.log({"val_BLEU-4": val_ble4})

    sample_caption(config, device, model, vocab)
    return val_loss / total_tokens if total_tokens > 0 else 0, val_ble4


def forward_pass(model: nn.Module, images: torch.Tensor, captions: torch.Tensor, criterion: nn.Module, vocab: Vocabulary) -> tuple[torch.Tensor, int]:
    """
    Performs a forward pass through the model.
    :param model: The model to perform the forward pass.
    :param images: Batch of images. Shape: (batch_size, 3, 224, 224)
    :param captions: Batch of captions. Shape (batch_size, seq_len)
    :param criterion: Loss function.
    :param vocab: Vocabulary of the training set.
    :return: The loss and the number of tokens.
    """
    outputs = model(images, captions[:, :-1])  # Shape: (batch_size, seq_len, vocab_size)
    targets = captions[:, 1:]  # Remove the <SOS> token | Shape: (batch_size, seq_len - 1)

    # Calculate loss per token (without reduction)
    per_token_loss = model.calc_loss(outputs, targets, criterion)  # (batch*(seq_len-1))

    # Create mask: 1 for valid tokens, 0 for banned tokens
    targets_flat = targets.reshape(-1)
    mask = torch.ones_like(targets_flat, dtype=torch.bool)
    banned_indices = [vocab.to_idx(PAD), vocab.to_idx(UNK), vocab.to_idx(SOS)]
    for banned_idx in banned_indices:
        mask &= (targets_flat != banned_idx)

    # Apply mask and compute mean loss
    valid_losses = per_token_loss[mask]
    num_valid = mask.sum().item()

    if num_valid > 0:
        loss = valid_losses.mean()
    else:
        loss = torch.tensor(0.0, device=images.device)  # Avoid division by zero

    return loss, num_valid


def sample_caption(config: dict, device: torch.device, model: nn.Module, vocab: Vocabulary) -> None:
    """
    Generate a sample caption
    :param config: Run configuration
    :param device: Device to run the model
    :param model: Model to generate the caption
    :param vocab: Vocabulary of the training set
    :return: Prints the sample caption
    """
    img = preprocess_image(str(os.path.join(ROOT, PATH_ALVARITO)), TRANSFORM)
    caption = gen_caption(model, img, vocab, config["max_caption_len"], device, config["temperature"], config["beam_size"])
    logger.info(f"Sample caption: {caption}")


def checkpoint(model: nn.Module, checkpoint_dir: str, best_val_loss: float, best_val_model: str, cur_lr: tuple, epoch: int):
    """
    Checkpoint the model
    :param model: Current model
    :param checkpoint_dir: Path to directory to save the model
    :param best_val_loss: Current best validation loss
    :param best_val_model: Current best model path
    :param cur_lr: Current learning rate
    :param epoch: Current epoch
    :return:
    """
    # delete previous best model
    if best_val_model is not None:
        os.remove(best_val_model)
    # save new best model
    best_val_model = os.path.join(ROOT, f"{checkpoint_dir}/best_val_{time_str()}_{str(round(best_val_loss, 4)).replace(".", "-")}.pt")
    best_val_info = {"epoch": epoch + 1, "val_loss": best_val_loss, "encoder_lr": cur_lr[0], "decoder_lr": cur_lr[1]}
    torch.save(model.state_dict(), best_val_model)
    return best_val_info, best_val_model
