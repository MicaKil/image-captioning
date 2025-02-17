from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from config import logger
from constants import PAD
from scripts.dataset.flickr_dataset import FlickrDataset
from scripts.utils import get_vocab


class FlickrDataLoader(DataLoader):
	"""
	Custom DataLoader for the Flickr8k dataset.
	"""

	def __init__(self, dataset: Union[FlickrDataset | Subset], batch_size=32, num_workers=4, shuffle=True, pin_memory=True):
		"""
		Initialize the DataLoader for the Flickr8k dataset.

		:param dataset: Dataset object to load
		:param batch_size: Number of samples per batch.
		:param num_workers: Number of subprocesses to use for data loading.
		:param shuffle: Whether to shuffle the data.
		:param pin_memory: Whether to pin memory.
		"""
		logger.info(f"Initializing DataLoader.")
		vocab = get_vocab(dataset)
		super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
		                 collate_fn=Collate(vocab.to_idx(PAD)))
		self.vocab = vocab
		logger.info(f"FlickerDataLoader initialized.")


class Collate:
	"""
	Collate function to pad sequences and move tensors to the specified device.
	"""

	def __init__(self, pad_idx: int):
		"""
		:param pad_idx: Index of the padding token
		"""
		self.pad_idx = pad_idx

	def __call__(self, batch):
		"""
		Collate function to pad sequences and move tensors to the specified device.
		:param batch: List of samples to collate
		:return: Tuple (images, captions) where images is a tensor and captions is a padded tensor
		"""
		images, captions, image_ids = zip(*batch)
		images = torch.stack(images)
		captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
		return images, captions, image_ids
