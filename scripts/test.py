import os.path

import pandas as pd
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

from config.config import logger
from constants import ROOT
from scripts.caption import gen_caption
from scripts.dataset.dataloader import CaptionLoader
from scripts.metrics import get_references, get_cider_score, get_bleu_scores
from scripts.utils import time_str, get_config


def test(model: nn.Module, test_loader: CaptionLoader, device: torch.device, save_dir: str, tag: str, use_wandb: bool, run_config: dict) -> tuple:
    """
    Evaluate model on test set and log results
    :param model: Model to evaluate
    :param test_loader: Test data loader to use
    :param device: Device to use (cpu or cuda)
    :param save_dir: If not None, save results to this directory
    :param tag: Tag to use for saving results
    :param use_wandb: Whether to use Weights & Biases for logging
    :param run_config: Configuration for the run if not using wandb
    :return:
    """
    config = get_config(run_config, use_wandb)
    logger.info("Start testing model")

    results = []
    all_hypotheses = []
    all_references = []
    df = test_loader.annotations
    vocab = test_loader.vocab
    smoothing = SmoothingFunction().method1

    model.eval()
    with torch.no_grad():
        for images, _, image_ids in tqdm(test_loader):
            # Generate captions
            generated = gen_caption(model, images, vocab, config["max_caption_len"], device, config["temperature"], config["beam_size"], False)
            all_hypotheses.extend(generated)
            # Get references
            references = get_references(df, image_ids)
            all_references.extend(references)

            # Log results
            for img_id, ref, gen in zip(image_ids, references, generated):
                results.append({"image_id": img_id, "references": ref, "generated": gen})

    # BLEU scores
    bleu_1, bleu_2, bleu_4 = get_bleu_scores(all_hypotheses, all_references, smoothing)

    # CIDEr score
    cider_score, _ = get_cider_score(all_hypotheses, all_references)

    # Log time
    logger.info(f"Finished testing model.")
    logger.info(f"BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-4: {bleu_4:.4f}, CIDEr: {cider_score:.4f}")

    # Log metrics
    metrics = {
        f"test_BLEU-1": bleu_1,
        f"test_BLEU-2": bleu_2,
        f"test_BLEU-4": bleu_4,
        f"test_CIDEr": cider_score
    }
    results = pd.DataFrame(results)

    log_and_save(metrics, results, save_dir, tag, use_wandb)
    return results, metrics


def log_and_save(metrics: dict, results: pd.DataFrame, save_dir: str, tag: str, use_wandb: bool) -> None:
    """
    Log results and metrics to wandb and save them to disk
    :param metrics: Metrics to log and save
    :param results: Results to log and save
    :param save_dir: If not None, save results to this directory
    :param tag: Tag to use for saving results
    :param use_wandb: Whether to use wandb for logging
    :return:
    """
    results_path = None
    if save_dir is not None:
        time_ = time_str()
        # Save results
        results_path = os.path.join(ROOT, save_dir, f"results_{tag}_{time_}.csv")
        results.to_csv(results_path, index=False, header=True)
        # Save metrics
        metrics_pd = pd.DataFrame(metrics, index=[0])
        metrics_pd.to_csv(os.path.join(ROOT, save_dir, f"metrics_{tag}_{time_}.csv"), index=False,
                          header=["test_BLEU-1", "test_BLEU-2", "test_BLEU-4", "test_CIDEr"])
    # Log results and metrics
    if use_wandb:
        wandb.log(metrics)
        results_table = wandb.Table(dataframe=results)
        results_artifact = wandb.Artifact(f"test_results", type="evaluation", metadata={"metrics": metrics})
        results_artifact.add(results_table, f"results")
        if save_dir is not None:
            results_artifact.add_file(results_path)
        wandb.log({f"test_results": results_table})
        wandb.log_artifact(results_artifact)
