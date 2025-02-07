import pickle
from datetime import datetime

import torch
from matplotlib import pyplot as plt


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


def show_img(img: torch.tensor, mean: list[float] = None, std: list[float] = None) -> None:
	"""
	Display an image.
	:param img: Image tensor
	:param mean: Mean values for normalization
	:param std: Standard deviation values for normalization
	:return: None
	"""
	img = img.squeeze(0).permute(1, 2, 0)
	if std is not None:
		img = img * torch.tensor(std)
	if mean is not None:
		img = img + torch.tensor(mean)
	img = img.clamp(0, 1)
	plt.imshow(img)
	plt.show()


def time_str() -> str:
	"""
	Return the current time as a string.
	:return: Current time as a string
	"""
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
