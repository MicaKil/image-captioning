import os.path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from torch import nn, Tensor
from torchvision.transforms import v2

from constants import ROOT
from dataset.vocabulary import Vocabulary


def gen_caption(model: nn.Module, images: torch.Tensor, vocab: Vocabulary, max_length: int = 30, device: torch.device = torch.device("cpu"),
                temperature: Optional[float] = None, beam_size: int = 1, no_grad=True) -> tuple[list[str], Tensor]:
    """
    Generate a caption for a batch of images using greedy search, temperature-based sampling, or beam search.

    :param model: Trained model for image captioning
    :param images: Preprocessed batch of image tensors (B, C, H, W)
    :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
    :param max_length: Maximum caption length
    :param device: Device to use
    :param temperature: Temperature for sampling (None for greedy search)
    :param beam_size: Beam size for beam search (1 for greedy search/temperature sampling)
    :param no_grad:

    :return: List of generated captions
    """

    model = model.to(device)
    return model.generate(images=images, vocab=vocab, max_length=max_length, device=device, temperature=temperature, beam_size=beam_size,
                          no_grad=no_grad)


def plot_attention(img_tensor: torch.Tensor, caption: str, tokens: list[str], attns: list, mean: list[float], std: list[float], columns,
                   save_name: str = None, save_dir: str = None):
    """
    Plot attention maps over the image for each step in the caption generation process.

    :param save_dir:
    :param img_tensor: Original image tensor (after normalization)
    :param caption: Generated caption
    :param tokens: List of tokens in the caption
    :param attns: List of attention maps (steps x layers x 49)
    :param mean: Mean values for normalization
    :param std: Standard deviation values for normalization
    :param columns: The number of columns to display the attention maps
    :param save_name: Path to save the plot (optional)
    """
    assert len(attns) == len(tokens), "attentions length must match caption length"
    # Inverse normalize the image
    inverse_normalize = v2.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    image = inverse_normalize(img_tensor).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    num_steps = len(attns)

    # height = (num_steps // columns + 1) * 3
    # width = columns * 2
    plt.figure(figsize=(10, 6), dpi=100)

    for step in range(num_steps):
        ax = plt.subplot(num_steps // columns + 1, columns, step + 1)
        attn = attns[step].reshape(8, 8)
        attn = zoom(attn, (256 / 8, 256 / 8))  # 7x7 -> 256x256

        ax.imshow(image)
        ax.imshow(attn, cmap='Greys_r', alpha=0.65)
        ax.set_title(f"{tokens[step]}", fontsize=12, pad=6)
        ax.axis('off')
    plt.suptitle(caption, fontsize=16)
    plt.tight_layout()
    if save_name:
        i = 0
        while save_name in os.listdir(os.path.join(ROOT, save_dir)):
            name = os.path.splitext(save_name)[0]
            save_name = f"{name}_{i:03d}.png"
            i += 1
        plt.savefig(os.path.join(ROOT, save_dir, save_name), bbox_inches='tight', dpi=150)
    plt.show()


def preprocess_image(img_path: str, transform: v2.Compose) -> torch.Tensor:
    """
    Preprocess an image for the model.

    :param img_path: Path to the image file
    :param transform: Transform to apply to the image
    :return: Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension
