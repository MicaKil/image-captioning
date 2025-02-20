import os.path

import pandas as pd
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import logger
from constants import ROOT
from scripts.caption import gen_caption
from scripts.dataset.vocabulary import Vocabulary
from scripts.utils import time_str, get_dataset, get_vocab


def test(model: nn.Module, test_loader: DataLoader, device: torch.device, save_dir: str, tag: str) -> tuple:
	"""
	Evaluate model on test set and log results
	:param model: Model to evaluate
	:param test_loader: Test data loader to use
	:param device: Device to use (cpu or cuda)
	:param save_dir: If not None, save results to this directory
	:param tag: Tag to use for saving results
	:return:
	"""

	logger.info("Start testing model")

	results = []
	all_hypotheses = []
	all_references = []
	df = get_dataset(test_loader).df
	vocab = get_vocab(test_loader)
	smoothing = SmoothingFunction().method1

	model.eval()
	with torch.no_grad():
		for batch_idx, (images, _, image_ids) in enumerate(tqdm(test_loader)):
			# Generate captions
			generated = gen_captions(model, vocab, device, images)
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
	cider_score = get_cider_score(all_hypotheses, all_references)

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

	log_and_save(metrics, results, save_dir, tag)
	return results, metrics


def get_references(df: pd.DataFrame, image_ids: list) -> list:
	"""
	Get references for a list of image ids
	:param df: DataFrame containing image ids and its captions
	:param image_ids: List of image ids to get references for
	:return: List of references for each image id
	"""
	references = []
	for img_id in image_ids:
		refs = df[df["image_id"] == img_id]["caption"].values
		references.append([ref for ref in refs])
	return references


def gen_captions(model: nn.Module, vocab: Vocabulary, device: torch.device, images: list) -> list:
	"""
	Generate captions for a list of images
	:param model: Model to use for caption generation
	:param device: Device to use
	:param images: List of images to generate captions for
	:param vocab: Vocabulary of the dataset
	:return: List of generated captions
	"""
	config = wandb.config

	generated = []
	for img in images:
		img = img.unsqueeze(0).to(device)
		caption = gen_caption(model, img, vocab, config["max_caption_len"], device, config["temperature"], config["beam_size"])
		generated.append(caption)
	return generated


def get_cider_score(all_hypotheses: list, all_references: list) -> float:
	"""
	Calculate CIDEr score for a list of hypotheses and references
	:param all_hypotheses: Hypotheses (generated captions) to evaluate
	:param all_references: References (ground truth captions) to evaluate
	:return: CIDEr score
	"""
	hyp_dict = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
	ref_dict = {i: refs for i, refs in enumerate(all_references)}
	cider_scorer = Cider()
	cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)
	return cider_score


def get_bleu_scores(all_hypotheses: list, all_references: list, smoothing) -> tuple:
	"""
	Calculate BLEU scores for a list of hypotheses and references
	:param all_hypotheses: Hypotheses (generated captions) to evaluate
	:param all_references: References (ground truth captions) to evaluate
	:param smoothing: Smoothing function to use
	:return: BLEU-1, BLEU-2, BLEU-4 scores
	"""
	tokenized_hypotheses = [hyp.split() for hyp in all_hypotheses]
	tokenized_references = [[ref.split() for ref in refs] for refs in all_references]
	bleu_1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
	bleu_2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
	bleu_4 = corpus_bleu(tokenized_references, tokenized_hypotheses, smoothing_function=smoothing)
	return bleu_1, bleu_2, bleu_4


def get_bleu4_score(all_hypotheses: list, all_references: list, smoothing) -> float:
	"""
	Calculate BLEU4 score for a list of hypotheses and references
	:param all_hypotheses: Hypotheses (generated captions) to evaluate
	:param all_references: References (ground truth captions) to evaluate
	:param smoothing: Smoothing function to use
	:return: BLEU-4 score
	"""
	tokenized_hypotheses = [hyp.split() for hyp in all_hypotheses]
	tokenized_references = [[ref.split() for ref in refs] for refs in all_references]
	return corpus_bleu(tokenized_references, tokenized_hypotheses, smoothing_function=smoothing)


def log_and_save(metrics: dict, results: pd.DataFrame, save_dir: str, tag: str) -> None:
	"""
	Log results and metrics to wandb and save them to disk
	:param metrics: Metrics to log and save
	:param results: Results to log and save
	:param save_dir: If not None, save results to this directory
	:param tag: Tag to use for saving results
	:return:
	"""
	results_path = None
	if save_dir is not None:
		time_ = time_str()
		# Save results
		results_path = os.path.join(ROOT, f"{save_dir}/results_{tag}_{time_}.csv")
		results.to_csv(results_path, index=False, header=True)
		# Save metrics
		metrics_pd = pd.DataFrame(metrics, index=[0])
		metrics_pd.to_csv(os.path.join(ROOT, f"{save_dir}/metrics_{tag}_{time_}.csv"), index=False,
		                  header=["test_BLEU-1", "test_BLEU-2", "test_BLEU-4", "test_CIDEr"])
	# Log results and metrics
	wandb.log(metrics)
	results_table = wandb.Table(dataframe=results)
	results_artifact = wandb.Artifact(f"test_results", type="evaluation", metadata={"metrics": metrics})
	results_artifact.add(results_table, f"results")
	if save_dir is not None:
		results_artifact.add_file(results_path)
	wandb.log({f"test_results": results_table})
	wandb.log_artifact(results_artifact)
