from datetime import datetime

import torch
from matplotlib import pyplot as plt


def show_img(img: torch.tensor, mean: list[float] = None, std: list[float] = None, batch_dim=False) -> None:
    """
    Display an image.
    :param img: Image tensor
    :param mean: Mean values for normalization
    :param std: Standard deviation values for normalization
    :param batch_dim: Whether the image tensor has a batch dimension
    :return: None
    """
    if batch_dim:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    if std is not None:
        img = img * torch.tensor(std)
    if mean is not None:
        img = img + torch.tensor(mean)
    img = img.clamp(0, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def time_str() -> str:
    """
    Return the current time as a string in the format "YYYY-MM-DD_HH-MM".
    :return: Current time as a string
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def date_str() -> str:
    """
    Return the current date as a string in the format "YYYY-MM-DD".
    :return: Current date as a string
    """
    return datetime.now().strftime("%Y-%m-%d")
