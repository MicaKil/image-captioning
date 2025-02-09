import logging
import time

import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from caption import gen_caption
from dataset.flickr_dataset import FlickerDataset
from dataset.vocabulary import Vocabulary

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def test(model: nn.Module, test_loader: DataLoader, vocab: Vocabulary, device: torch.device, max_caption_len: int,
		 wandb_run: Run) -> tuple:
	"""
	Evaluate model on test set and log results in a wandb table
	:param wandb_run:
	"""
	logger.info("Start testing model")
	model.eval()
	results = []
	all_hypotheses = []
	all_references = []
	df = test_loader.dataset.df if isinstance(test_loader.dataset, FlickerDataset) else test_loader.dataset.dataset.df
	smoothing = SmoothingFunction().method1
	start_time = time.time()

	with torch.no_grad():
		for batch_idx, (images, _, image_ids) in enumerate(tqdm(test_loader)):
			# Generate captions
			generated = []
			for img in images:
				img = img.unsqueeze(0).to(device)
				caption = gen_caption(model, img, vocab, max_caption_len, device)
				generated.append(caption)
			all_hypotheses.extend(generated)
			# Get references
			references = []
			for img_id in image_ids:
				refs = df[df["image_id"] == img_id]["caption"].values
				references.append([ref for ref in refs])
			all_references.extend(references)

			# Log results
			for img_id, ref, gen in zip(image_ids, references, generated):
				results.append({
					"image_id": img_id,
					"references": ref,
					"generated": gen
				})

		# BLEU scores
		tokenized_hypotheses = [hyp.split() for hyp in all_hypotheses]
		tokenized_references = [[ref.split() for ref in refs] for refs in all_references]
		bleu_1 = corpus_bleu(tokenized_references,
							 tokenized_hypotheses,
							 weights=(1, 0, 0, 0),
							 smoothing_function=smoothing)
		bleu_2 = corpus_bleu(tokenized_references,
							 tokenized_hypotheses,
							 weights=(0.5, 0.5, 0, 0),
							 smoothing_function=smoothing)
		bleu_4 = corpus_bleu(tokenized_references,
							 tokenized_hypotheses,
							 smoothing_function=smoothing)

		# CIDEr score
		hyp_dict = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
		ref_dict = {i: refs for i, refs in enumerate(all_references)}
		cider_scorer = Cider()
		cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)

		# Log time
		test_time = time.time() - start_time
		logger.info(f"Testing took {test_time:.2f} seconds")
		wandb_run.log({"test_time": test_time})

		# Log metrics
		metrics = {
			"test_BLEU-1": bleu_1,
			"test_BLEU-2": bleu_2,
			"test_BLEU-4": bleu_4,
			"test_CIDEr": cider_score
		}
		logger.info(f"BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-4: {bleu_4:.4f}, CIDEr: {cider_score:.4f}")
		return pd.DataFrame(results), metrics
