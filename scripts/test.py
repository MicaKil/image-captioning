import logging
import os.path
import time

import pandas as pd
import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from constants import BASIC_RESULTS, ROOT
from scripts.caption import gen_caption
from scripts.dataset.flickr_dataset import FlickerDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.utils import time_str

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def test(model: nn.Module, test_loader: DataLoader, device: torch.device, max_caption_len: int, save_results: bool,
		 wandb_run: Run) -> tuple:
	"""
	Evaluate model on test set and log results
	:param model: Model to evaluate
	:param test_loader: Test data loader to use
	:param device: Device to use (cpu or cuda)
	:param max_caption_len: Maximum length of generated captions
	:param save_results: Whether to save results to disk
	:param wandb_run: Wandb run object
	:return:
	"""

	logger.info("Start testing model")
	start_time = time.time()

	results = []
	all_hypotheses = []
	all_references = []
	df = test_loader.dataset.df if isinstance(test_loader.dataset, FlickerDataset) else test_loader.dataset.dataset.df
	vocab = test_loader.dataset.vocab if isinstance(test_loader.dataset,
													FlickerDataset) else test_loader.dataset.dataset.vocab
	smoothing = SmoothingFunction().method1

	model.eval()
	with torch.no_grad():
		for batch_idx, (images, _, image_ids) in enumerate(tqdm(test_loader)):
			# Generate captions
			generated = gen_captions(model, vocab, device, images, max_caption_len)
			all_hypotheses.extend(generated)
			# Get references
			references = get_references(df, image_ids)
			all_references.extend(references)

			# Log results
			for img_id, ref, gen in zip(image_ids, references, generated):
				results.append({
					"image_id": img_id,
					"references": ref,
					"generated": gen
				})

		# BLEU scores
		bleu_1, bleu_2, bleu_4 = get_bleu_scores(all_hypotheses, all_references, smoothing)

		# CIDEr score
		cider_score = get_cider_score(all_hypotheses, all_references)

		# Log time
		test_time = time.time() - start_time
		wandb_run.log({"test_time": test_time})

		logger.info(f"Testing took {test_time:.2f} seconds")
		logger.info(f"BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-4: {bleu_4:.4f}, CIDEr: {cider_score:.4f}")

		# Log metrics
		metrics = {
			"test_BLEU-1": bleu_1,
			"test_BLEU-2": bleu_2,
			"test_BLEU-4": bleu_4,
			"test_CIDEr": cider_score
		}
		results = pd.DataFrame(results)

		log_and_save(metrics, results, save_results, wandb_run)
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


def gen_captions(model: nn.Module, vocab: Vocabulary, device: torch.device, images: list, max_caption_len: int) -> list:
	"""
	Generate captions for a list of images
	:param model: Model to use for caption generation
	:param device: Device to use
	:param images: List of images to generate captions for
	:param max_caption_len: Maximum length of generated captions
	:param vocab: Vocabulary of the dataset
	:return: List of generated captions
	"""
	generated = []
	for img in images:
		img = img.unsqueeze(0).to(device)
		caption = gen_caption(model, img, vocab, max_caption_len, device)
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
	bleu_2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.5, 0.5, 0, 0),
						 smoothing_function=smoothing)
	bleu_4 = corpus_bleu(tokenized_references, tokenized_hypotheses, smoothing_function=smoothing)
	return bleu_1, bleu_2, bleu_4


def log_and_save(metrics: dict, results: pd.DataFrame, save_results: bool, wandb_run):
	"""
	Log results and metrics to wandb and save them to disk
	:param metrics: Metrics to log and save
	:param results: Results to log and save
	:param save_results: Whether to save results to disk
	:param wandb_run: Wandb run object
	:return:
	"""
	results_path = None
	if save_results:
		time_ = time_str()
		# Save results
		results_path = os.path.join(ROOT, f"{BASIC_RESULTS}/results_{time_}.csv")
		results.to_csv(results_path, index=False, header=True)
		# Save metrics
		metrics_pd = pd.DataFrame(metrics, index=[0])
		metrics_pd.to_csv(os.path.join(ROOT, f"{BASIC_RESULTS}/metrics_{time_}.csv"), index=False,
						  header=["test_BLEU-1", "test_BLEU-2", "test_BLEU-4", "test_CIDEr"])
	# Log results and metrics
	wandb_run.log(metrics)
	results_table = wandb.Table(dataframe=results)
	results_artifact = wandb.Artifact("test_results", type="evaluation", metadata={"metrics": metrics})
	results_artifact.add(results_table, "results")
	if save_results:
		results_artifact.add_file(results_path)
	wandb_run.log({"test_results": results_table})
	wandb_run.log_artifact(results_artifact)
