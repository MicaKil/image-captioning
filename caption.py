from typing import Optional

import torch
from torch import nn

from dataset.vocabulary import Vocabulary


def gen_caption(model: nn.Module, images: torch.Tensor, vocab: Vocabulary, max_len: int = 30, device: torch.device = torch.device("cpu"),
                temp: Optional[float] = None, beam_size: int = 1, no_grad=True, return_atnn=False) -> tuple:
    """
    Generate a caption for a batch of images using greedy search, temperature-based sampling, or beam search.

    :param model: Trained model for image captioning
    :param images: Preprocessed batch of image tensors (B, C, H, W)
    :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
    :param max_len: Maximum caption length
    :param device: Device to use
    :param temp: Temperature for sampling (None for greedy search)
    :param beam_size: Beam size for beam search (1 for greedy search/temperature sampling)
    :param no_grad:
    :param return_atnn:

    :return: List of generated captions
    """

    model = model.to(device)
    return model.generate(images=images, vocab=vocab, max_len=max_len, device=device, temp=temp, beam_size=beam_size,
                          no_grad=no_grad, return_atnn=return_atnn)
