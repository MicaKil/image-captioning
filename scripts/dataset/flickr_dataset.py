import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

from config import logger
from constants import ROOT, EOS, SOS, FLICKR8K_CSV_FILE
from scripts.dataset.vocabulary import Vocabulary


class FlickrDataset(Dataset):
	"""
	Custom Dataset for loading Flickr8k images and captions.
	"""

	def __init__(self, img_dir: str, df_captions: pd.DataFrame, vocab: Vocabulary, transform=None, target_transform=None):
		"""
		:param img_dir: Path to the directory containing the images
		:param df_captions: DataFrame containing the image IDs and captions
		:param vocab: Vocabulary object to use if provided
		:param transform: Transform to apply to the images
		:param target_transform: Transform to apply to the target captions
		"""
		logger.info("Initializing FlickerDataset.")
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.df = df_captions
		self.img_ids, self.captions = self.df["image_id"], self.df["caption"]
		self.vocab = vocab

	def __len__(self):
		"""
		Return the number of samples in the dataset.
		:return: Number of samples
		"""
		return len(self.captions)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
		"""
		Get a sample (image, caption, image ID) from the dataset.
		:param idx: Index of the sample to retrieve
		:return: Tuple containing the image tensor, caption tensor, and image ID
		"""
		img = decode_image(str(os.path.join(self.img_dir, self.img_ids[idx])), mode=ImageReadMode.RGB)
		if self.transform:
			img = self.transform(img)

		caption = [self.vocab.to_idx(SOS)] + self.vocab.to_idx_list(self.captions[idx]) + [self.vocab.to_idx(EOS)]
		caption = torch.tensor(caption, dtype=torch.long)

		if self.target_transform:
			caption = self.target_transform(caption)

		return img, caption, self.img_ids[idx]


def load_captions(path: str, save_captions=False) -> pd.DataFrame:
	"""
	Load the captions from the annotation file.
	:param path: Path to the annotation file
	:param save_captions: If True, save to a CVS file or overwrite the existing CSV file
	:return: DataFrame containing the image filenames and corresponding captions
	"""

	if os.path.splitext(path)[1] == ".csv":
		logger.info("Loading captions from CSV file.")
		df = pd.read_csv(path)
	else:
		logger.info("Loading captions from annotation file.")
		df = pd.DataFrame(extract_captions(path))  # Convert to DataFrame
	if save_captions:
		logger.info("Saving captions to CSV file.")
		df.to_csv(str(os.path.join(ROOT, FLICKR8K_CSV_FILE)), header=True, index=False)
	return df


def extract_captions(path: str) -> list[dict[str, str]]:
	"""
	Extract the captions from the annotation file.
	Sample line:
		"1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way."
	:param path: Path to the annotation file
	:return: List of dictionaries containing the image ID and caption
	"""
	captions = []
	with open(path, "r") as f:
		for line in f:
			image_id, caption = line.strip().split("\t")
			image_id = image_id.split("#")[0]
			captions.append({"image_id": image_id, "caption": caption})
	return captions


def split_dataframe(df: pd.DataFrame, split_lengths: list[int]) -> list[pd.DataFrame]:
	"""
	Split a DataFrame containing image IDs and captions into multiple DataFrames based on the specified lengths.
	:param df: DataFrame containing the Flickr8k image IDs and captions
	:param split_lengths: List of integers specifying the lengths of the splits. Must sum to the number of unique images.
	:return: List of DataFrames containing the splits
	"""
	# Extract all unique image IDs from the dataframe
	unique_images = df['image_id'].unique()
	n_total = len(unique_images)

	# Verify that the sum of split lengths equals the number of unique images
	if sum(split_lengths) != n_total:
		raise ValueError(f"Sum of split lengths ({sum(split_lengths)}) must equal the number of unique images ({n_total}).")

	# Shuffle the unique image IDs to ensure randomness
	shuffled_images = np.random.permutation(unique_images)

	# Calculate the indices where the splits occur
	split_indices = np.cumsum(split_lengths[:-1])

	# Split the shuffled image IDs into groups according to split_lengths
	image_splits = np.split(shuffled_images, split_indices)

	# Create dataframe splits based on the image IDs in each split
	df_splits = []
	for images in image_splits:
		mask = df['image_id'].isin(images)
		df_split = df[mask].reset_index(drop=True)
		df_splits.append(df_split)

	return df_splits
