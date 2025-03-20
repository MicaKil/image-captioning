from typing import Optional

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torchvision.transforms import v2

from scripts.dataset.vocabulary import Vocabulary


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


def plot_attention(image_tensor: torch.Tensor, caption: list[str], attentions: list, mean: list[float], std: list[float], save_path: str = None):
    """
    Plot attention maps over the image for each step in the caption generation process.

    :param image_tensor: Original image tensor (after normalization)
    :param caption: Generated caption (list of words)
    :param attentions: List of attention maps (steps x layers x 49)
    :param mean: Mean values for normalization
    :param std: Standard deviation values for normalization
    :param save_path: Path to save the plot (optional)
    """
    # print(f"caption: {len(caption)}")
    # print(f"ann: {len(attentions)}")
    assert len(attentions) == len(caption), "attentions length must match caption length"
    # Inverse normalize the image
    inverse_normalize = v2.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    image = inverse_normalize(image_tensor).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    num_layers = len(attentions[0])
    num_steps = len(attentions)

    plt.figure(figsize=(20, 20))
    for step in range(num_steps):
        for layer in range(num_layers):
            ax = plt.subplot(num_steps, num_layers, step * num_layers + layer + 1)
            # Reshape attention to 7x7 and upscale to image size
            attn = attentions[step][layer].reshape(7, 7)
            attn = np.kron(attn, np.ones((32, 32)))  # 7x7 -> 224x224

            ax.imshow(image)
            ax.imshow(attn, cmap='jet', alpha=0.5)
            ax.set_title(f"Step {step + 1}: {caption[step]}\nLayer {layer + 1}")
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
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
