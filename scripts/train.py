import os.path

import numpy as np
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

import scripts.metrics as metrics
from config.config import logger
from constants import ROOT, PATH_ALVARITO, PAD, UNK, SOS
from scripts.caption import gen_caption, preprocess_image
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.vocabulary import Vocabulary
from scripts.runner.config import TRANSFORM
from scripts.scheduler import SchedulerWrapper
from scripts.utils import time_str, get_config


def train(model: nn.Module, train_loader: CaptionLoader, val_loader: CaptionLoader, device: torch.device, criterion: nn.Module,
          optimizer: torch.optim, scheduler: SchedulerWrapper, checkpoint_dir: str, use_wandb: bool, run_config: dict,
          resume_checkpoint: str) -> tuple:
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
    :param resume_checkpoint: Path to a checkpoint to resume training from
    :return: Path to the best model
    """
    config = get_config(run_config, use_wandb)
    if use_wandb:
        wandb.watch(model, criterion=criterion, log="all")

    if config["eval_bleu4"] and (config["eval_bleu4"]["step"] is None or config["eval_bleu4"]["step"] < 1):
        raise ValueError("The evaluation step for BLEU-4 must be greater than 0 if enabled")

    logger.info(f"Start training model {model.__class__.__name__} for {config["max_epochs"]} {"epoch" if config["max_epochs"] == 1 else "epochs"}")

    cur_lr = (config["encoder_lr"], config["decoder_lr"])
    best_val_loss = np.inf
    best_pth = None
    best_state = None
    last_path = None
    last_state = dict()
    epochs_no_improve = 0

    start_epoch = 0
    if resume_checkpoint:
        best_val_loss, epochs_no_improve, start_epoch = resume(model, device, optimizer, scheduler, resume_checkpoint)

    model = model.to(device)
    for epoch in range(start_epoch, config["max_epochs"]):
        avg_train_loss = train_load(model, train_loader, device, epoch, criterion, optimizer, use_wandb, run_config)
        avg_val_loss, val_bleu4 = eval_load(model, val_loader, device, epoch, criterion, use_wandb, run_config)

        if val_bleu4 is not None:
            logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val BLEU-4 = {val_bleu4:.4f}")
        else:
            logger.info(f"Epoch {epoch + 1} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        last_state = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        if config["eval_bleu4"]:
            last_state["val_BLEU-4"] = val_bleu4
        if use_wandb:
            wandb.log(last_state)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            cur_lr = scheduler.scheduler.get_last_lr()
            if use_wandb:
                wandb.log({"encoder_lr": cur_lr[0], "decoder_lr": cur_lr[1]})

        # Early stopping and checkpointing
        if last_path is not None and last_path != best_pth:
            # Only remove the last checkpoint if it is not the best one
            os.remove(last_path)
        last_path = os.path.join(ROOT, f"{checkpoint_dir}/LAST_{time_str()}_{str(round(avg_val_loss, 4)).replace(".", "-")}.pt")
        cur_state = save_checkpoint(model, last_path, optimizer, scheduler, avg_train_loss, avg_val_loss, cur_lr, epoch, epochs_no_improve, config)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if best_pth is not None:
                os.remove(best_pth)
            best_pth = last_path
            best_state = cur_state
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if config["patience"] is not None and epochs_no_improve >= config["patience"]:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
            if scheduler is not None:
                if (epochs_no_improve - 1) > 0 and (epochs_no_improve - 1) % config["scheduler"]["patience"] == 0:
                    logger.info(f"Reducing learning rate. Encoder LR: {cur_lr[0]}, Decoder LR: {cur_lr[1]}")

    logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")

    # check the best model is not the last model
    if best_state is not None and best_state["epoch"] == last_state["epoch"]:
        # if they are the same, then the best model is the last model
        logger.info("Best model is the last model")
        return None, None, last_path

    # rename the best model so it includes the prefix BEST
    best_pth_new = best_pth.replace("LAST", "BEST")
    os.rename(best_pth, best_pth_new)

    return best_pth_new, best_state, last_path


def resume(model: nn.Module, device: torch.device, optimizer: torch.optim, scheduler: SchedulerWrapper, checkpoint_path: str) -> tuple[float, int, int]:
    """
    Resume training from a checkpoint
    :param model: The model to resume training
    :param device: Device to run the training on
    :param optimizer: Optimizer for training
    :param scheduler: Scheduler for the optimizer
    :param checkpoint_path: Path to the checkpoint
    :return: The best validation loss, number of epochs without improvement, and the starting epoch
    """
    checkpoint = torch.load(os.path.join(ROOT, checkpoint_path))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    for state in optimizer.state.values():
        # Move optimizer states to current device
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if scheduler and checkpoint['scheduler_state']:
        scheduler.scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    epochs_no_improve = checkpoint['epochs_no_improve']
    logger.info(f"Resuming training from epoch {start_epoch} with val loss {best_val_loss:.4f}")
    return best_val_loss, epochs_no_improve, start_epoch


def train_load(model: nn.Module, train_loader: CaptionLoader, device: torch.device, epoch: int, criterion: nn.Module, optimizer: torch.optim,
               use_wandb: bool, run_config: dict) -> float:
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
    config = get_config(run_config, use_wandb)

    train_loss = 0.
    total_tokens = 0
    vocab = train_loader.vocab

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


def train_rl(model: nn.Module, train_loader: CaptionLoader, device: torch.device, epoch: int, criterion: nn.Module, optimizer: torch.optim,
             vocab: Vocabulary, use_wandb: bool, run_config: dict) -> float:
    """
    Reinforcement learning training with CIDEr optimization
    :param model:
    :param train_loader:
    :param device:
    :param epoch:
    :param criterion:
    :param optimizer:
    :param vocab:
    :param use_wandb:
    :param run_config:
    :return:
    """
    model.train()
    running_reward = 0.0
    running_loss = 0.0
    config = get_config(run_config, use_wandb)

    batch_progress = tqdm(train_loader, desc="RL Training")
    for images, _, images_id in batch_progress:
        images = images.to(device)
        optimizer.zero_grad()

        # Generate captions using beam search
        with torch.no_grad():
            # features: torch.Tensor, vocab: Vocabulary, max_length: int, beam_size: int
            generated, log_probs = gen_caption(model, images, vocab, config["max_caption_len"], device, None, config["beam_size"], True)
            # Get references and compute CIDEr scores
            references = metrics.get_references(train_loader.annotations, images_id)
            rewards = metrics.get_cider_score(generated, references)

        # Convert rewards to tensor
        rewards = torch.tensor(rewards, device=device)

        # Normalize rewards
        if config['rl_baseline']:
            rewards = rewards - rewards.mean()

        # Calculate loss
        loss = -torch.mean(log_probs * rewards)

        # Backpropagation
        loss.backward()
        if config['gradient_clip'] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        running_reward += rewards.mean().item()
        batch_progress.set_postfix({'loss': running_loss / (batch_progress.n + 1),
                                    'reward': running_reward / (batch_progress.n + 1)})

    return running_loss / len(train_loader)


def eval_load(model: nn.Module, val_loader: CaptionLoader, device: torch.device, epoch: int, criterion: nn.Module, use_wandb: bool,
              run_config: dict) -> tuple[float | int, float | None]:
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
    config = get_config(run_config, use_wandb)

    # for BLEU score
    val_ble4 = None
    all_hypotheses = []
    all_references = []
    smoothing = SmoothingFunction().method1
    calc_bleu4 = config["eval_bleu4"] and (epoch == 0 or (epoch + 1) % config["eval_bleu4"]["step"] == 0)

    val_loss = 0.0
    total_tokens = 0
    vocab = val_loader.vocab
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
                generated = gen_caption(model, images, vocab, config["max_caption_len"], device, config["temperature"], config["beam_size"], False)
                all_hypotheses.extend(generated)
                references = metrics.get_references(val_loader.annotations, images_id)
                all_references.extend(references)

            batch_progress.set_postfix({"loss": loss.item() / num_tokens if num_tokens > 0 else 0})

    if calc_bleu4:
        val_ble4 = metrics.get_bleu4_score(all_hypotheses, all_references, smoothing)
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

    # Create mask: True for valid tokens, False for banned tokens
    targets = targets.reshape(-1)
    mask = torch.ones_like(targets, dtype=torch.bool)
    banned_indices = [vocab.str_to_idx(token) for token in [PAD, SOS, UNK]]
    for banned_idx in banned_indices:
        mask &= (targets != banned_idx)

    # Apply mask and compute mean loss
    per_token_loss = per_token_loss[mask]
    num_tokens = mask.sum().item()

    if num_tokens > 0:
        loss = per_token_loss.sum()
    else:
        loss = torch.tensor(0.0, device=images.device)  # Avoid division by zero

    if loss > 1e8:
        logger.warning(f"Loss is too high: {loss.item()}")

    return loss, num_tokens


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
    caption = gen_caption(model, img, vocab, config["max_caption_len"], device, config["temperature"], config["beam_size"], False)[0]
    logger.info(f"Sample caption: {caption}")


def save_checkpoint(model: nn.Module, path: str, optimizer: torch.optim, scheduler: SchedulerWrapper, train_loss: float, val_loss: float, cur_lr: tuple,
                    epoch: int, epochs_no_improve: int, config: dict) -> dict:
    """
    Checkpoint the model
    :param model: Current model
    :param path: Path to save the checkpoint
    :param optimizer: Current optimizer
    :param scheduler: Scheduler for the optimizer
    :param train_loss: Current training loss
    :param val_loss: Current best validation loss
    :param cur_lr: Current learning rate in the encoder and decoder
    :param epoch: Current epoch
    :param epochs_no_improve: Number of epochs without improvement
    :param config: Run configuration
    :return: Dictionary with the state of the saved checkpoint
    """
    state = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.scheduler.state_dict() if scheduler else None,
        'best_val_loss': val_loss,
        'train_loss': train_loss,
        'epochs_no_improve': epochs_no_improve,
        'lr': cur_lr,
        'config': dict(config)
    }
    torch.save(state, path)
    return state
