import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from numpy._core.multiarray import _SCT
from pycocoevalcap.cider.cider import Cider


def get_references(df: pd.DataFrame, image_ids: list) -> list:
    """
    Get references for a list of image ids
    :param df: DataFrame containing image ids and its captions
    :param image_ids: List of image ids to get references for
    :return: List of references for each image id
    """
    references = []
    for img_id in image_ids:
        refs = df[df["file_name"] == img_id]["caption"].values
        references.append([ref for ref in refs])
    return references


def get_cider_score(all_hypotheses: list, all_references: list)  -> tuple[float, list]:
    """
    Calculate CIDEr score for a list of hypotheses and references
    :param all_hypotheses: Hypotheses (generated captions) to evaluate
    :param all_references: References (ground truth captions) to evaluate
    :return: CIDEr mean score, CIDEr scores for each hypothesis
    """
    hyp_dict = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
    ref_dict = {i: refs for i, refs in enumerate(all_references)}
    cider_scorer = Cider()
    cider_score, cider_scores = cider_scorer.compute_score(ref_dict, hyp_dict)
    return cider_score, cider_scores.tolist()


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
