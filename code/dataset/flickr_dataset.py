import logging
import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

import utils
from constants import ROOT, EOS, SOS, VOCAB_FILE, FLICKR8K_CSV_FILE
from dataset.vocabulary import Vocabulary

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class FlickerDataset(Dataset):
	"""
	Custom Dataset for loading Flickr8k images and captions.
	"""

	def __init__(self,
				 ann_file: str,
				 img_dir: str,
				 save_captions=False,
				 vocab_threshold=2,
				 vocab: Vocabulary = None,
				 save_vocab=False,
				 transform=None,
				 target_transform=None):
		"""
		:param ann_file: Path to the annotation file with the image IDs and captions
		:param img_dir: Path to the directory containing the images
		:param save_captions: If True, save the captions to a CSV file
		:param vocab_threshold: Minimum frequency of a word to be included in the vocabulary
		:param vocab: Vocabulary object to use if provided
		:param save_vocab: If True, save the vocabulary to a file
		:param transform: Transform to apply to the images
		:param target_transform: Transform to apply to the target captions
		"""
		logger.info("Initializing FlickerDataset.")
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

		self.df = load_captions(ann_file, save_captions)
		self.img_ids, self.captions = self.df["image_id"], self.df["caption"]

		if vocab is not None:
			logger.info("Using existing vocabulary.")
			self.vocab = Vocabulary
		else:
			self.vocab = Vocabulary(vocab_threshold, self.captions)

		if save_vocab:
			utils.dump(self.vocab, str(os.path.join(ROOT, VOCAB_FILE)))

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
