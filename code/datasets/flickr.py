import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image, ImageReadMode

import utils
from constants import ROOT
from datasets.vocabulary import Vocabulary

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
				 target_transform=None
				 ):
		"""
		:param ann_file: Path to the annotation file with the image IDs and captions
		:param img_dir: Path to the directory containing the images
		:param save_captions: If True, save the captions to a CSV file
		:param vocab_threshold: Minimum frequency of a word to be included in the vocabulary
		:param vocab: Path to the vocabulary file
		:param save_vocab: If True, save the vocabulary to a file
		:param transform: Transform to apply to the images
		:param target_transform: Transform to apply to the target captions
		"""
		logger.info("Initializing FlickerDataset.")
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

		df = load_captions(ann_file, save_captions)
		self.img_ids, self.captions = df["image_id"], df["caption"]

		if vocab is not None and not save_vocab:
			logger.info("Using existing vocabulary.")
			self.vocab = Vocabulary
		else:
			self.vocab = Vocabulary(vocab_threshold, self.captions)
			utils.dump(self.vocab, os.path.join(ROOT, "code/datasets/vocab.pkl"))

	def __len__(self):
		"""
		Return the number of samples in the dataset.
		:return: Number of samples
		"""
		return len(self.captions)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Get a sample (image and caption) from the dataset.
		:param idx: Index of the sample to retrieve
		:return: Tuple containing the image and caption tensors
		"""
		img = decode_image(str(os.path.join(self.img_dir, self.img_ids[idx])), mode=ImageReadMode.RGB)
		if self.transform:
			img = self.transform(img)

		caption = [self.vocab.to_idx("<SOS>")] + self.vocab.to_idx_list(self.captions[idx]) + [
			self.vocab.to_idx("<EOS>")]
		caption = torch.tensor(caption, dtype=torch.long)

		if self.target_transform:
			caption = self.target_transform(caption)

		return img, caption


def load_captions(path: str, overwrite=False) -> pd.DataFrame:
	"""
	Load the captions from the annotation file.
	:param path: Path to the annotation file
	:param overwrite: If True, overwrite the existing CSV file
	:return: DataFrame containing the image filenames and corresponding captions
	"""

	if os.path.splitext(path)[1] == ".csv" and not overwrite:
		logger.info("Loading captions from CSV file.")
		return pd.read_csv(path)

	logger.info("Loading captions from annotation file.")
	df = pd.DataFrame(extract_captions(path))  # Convert to DataFrame
	df.to_csv(os.path.join(ROOT, "../datasets/flickr8k/captions.csv"), index=False)
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
		images, captions = zip(*batch)
		images = torch.stack(images)
		captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
		return images, captions


def data_loader(ann_file: str,
				img_dir: str,
				save_captions=False,
				vocab_threshold=2,
				vocab: Vocabulary = None,
				save_vocab=False,
				transform=None,
				batch_size=32,
				num_workers=4,
				shuffle=True,
				pin_memory=True
				) -> DataLoader:
	"""
	Create a DataLoader for the Flickr8k dataset.

	:param ann_file: Path to the annotation file.
	:param img_dir: Path to the directory containing the images.
	:param save_captions: Whether to save the captions to a CSV file.
	:param vocab_threshold: Minimum frequency of a word to be included in the vocabulary.
	:param vocab: Vocabulary object.
	:param save_vocab: Whether to save the vocabulary to a file.
	:param transform: Transform to apply to the images.
	:param batch_size: Number of samples per batch.
	:param num_workers: Number of subprocesses to use for data loading.
	:param shuffle: Whether to shuffle the data.
	:param pin_memory: Whether to pin memory.

	:return: DataLoader for the Flickr8k dataset.
	"""
	dataset = FlickerDataset(ann_file, img_dir, save_captions, vocab_threshold, vocab, save_vocab,
							 transform)
	logger.info(f"FlickerDataset loaded.")
	pad_idx = dataset.vocab.to_idx("<PAD>")
	logger.info(f"Initializing DataLoader.")
	return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
					  collate_fn=Collate(pad_idx))


if __name__ == "__main__":
	root_dir_ = "../../datasets/flickr8k/images"
	ann_file_ = "../../datasets/flickr8k/captions.csv"

	transform_ = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
	])

	dataloader = data_loader(ann_file_, root_dir_, transform=transform_, vocab=utils.load("datasets/vocab.pkl"),
							 pin_memory=True, num_workers=8)

	for i, (images_, captions_) in enumerate(dataloader):
		print(f"Images shape: {images_.size()}")
		print(f"Captions shape: {captions_.size()}")

		# Display the first image and caption
		print(f"Image from batch:\n {images_[0]}")

		img_ = images_[0].permute(1, 2, 0).numpy()  # Permute dimensions to (H, W, C)
		plt.imshow(img_)
		plt.show()

		print(f"Caption: \n{captions_[0]}")
		print(f"Text of the caption: {dataloader.vocab.to_text(captions_[0])}")
		break
