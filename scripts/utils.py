import pickle
from datetime import datetime
from typing import Union

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from scripts.dataset.vocabulary import Vocabulary


def dump(obj: any, path: str):
	"""
	Dump an object to a file.
	:param obj: The object to dump
	:param path: Path to the file where the object will be dumped
	:return:
	"""
	with open(path, "wb") as f:
		pickle.dump(obj, f)


def load(path: str) -> any:
	"""
	Load an object from a file.
	:param path: Path to the file where the object is stored
	:return: The object loaded from the file
	"""
	with open(path, "rb") as f:
		return pickle.load(f)


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


def get_vocab(obj: Union[DataLoader, Dataset]) -> Vocabulary:
	"""
	Get the vocabulary from a DataLoader or Dataset object.
	:param obj: DataLoader or Dataset object
	:return: Vocabulary object
	"""
	obj = get_dataset(obj)
	return obj.vocab


def get_dataset(obj: Union[DataLoader, Dataset]) -> Dataset:
	"""
	Get the vocabulary from a DataLoader or Dataset object.
	:param obj: DataLoader or Dataset object
	:return: Vocabulary object
	"""
	while not isinstance(obj, Dataset):
		obj = obj.dataset
	return obj
