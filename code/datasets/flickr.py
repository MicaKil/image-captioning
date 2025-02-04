import logging
import os.path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from nltk import word_tokenize, TreebankWordDetokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image, ImageReadMode

import utils

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class FlickerDataset(Dataset):
	"""
	Custom Dataset for loading Flickr8k images and captions.
	"""

	def __init__(self, ann_file: str, img_dir: str, save_captions=False, vocab_threshold=2, vocab_file: str = None,
				 save_vocab=False, transform=None, target_transform=None):
		"""
		:param ann_file: Path to the annotation file with the image IDs and captions
		:param img_dir: Path to the directory containing the images
		:param save_captions: If True, save the captions to a CSV file
		:param vocab_threshold: Minimum frequency of a word to be included in the vocabulary
		:param vocab_file: Path to the vocabulary file
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

		if vocab_file is not None and not save_vocab:
			logger.info("Loading vocabulary from file.")
			self.vocab = utils.load(vocab_file)
		else:
			self.vocab = Vocabulary(vocab_threshold, self.captions)
			utils.dump(self.vocab, "vocab.pkl")

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
		caption = torch.tensor(caption, dtype=torch.float32)
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
	df.to_csv("../datasets/flickr8k/captions.csv", index=False)
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


class Vocabulary:
	"""
	Vocabulary class that builds a vocabulary from a list of texts.
	"""

	def __init__(self, freq_threshold: int, text_list: list[str] = None):
		"""
		:param freq_threshold: Minimum frequency of a word to be included in the vocabulary
		:param text_list: List of texts to build the vocabulary from
		"""
		self.freq_threshold = freq_threshold
		self._to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
		self._to_str = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
		self.word_counts = Counter()

		if text_list is not None:
			self.build_vocab(text_list)

	@staticmethod
	def tokenize_eng(text: str) -> list[str]:
		"""
		Tokenize an English text.
		:param text: English text
		:return: List of word tokens
		"""
		return word_tokenize(text.lower())

	def build_vocab(self, text_list: list[str]):
		"""
		Build the vocabulary from a list of texts.
		:param text_list: List of texts
		:return: None
		"""
		logger.info("Building vocabulary.")
		for text in text_list:
			self.word_counts.update(self.tokenize_eng(text))

		idx = 4
		for word, count in self.word_counts.items():
			if count >= self.freq_threshold:
				self._to_idx[word] = idx
				self._to_str[idx] = word
				idx += 1

	def to_idx_list(self, text: str) -> list[int]:
		"""
		Convert a text to a list of word indices.
		:param text: Input text
		:return: A list of word indices
		"""
		return [self.to_idx(word) for word in self.tokenize_eng(text)]

	def to_idx(self, word: str) -> int:
		"""
		Convert a word to its index.
		:param word: Word to convert
		:return: Index of the word or the index of "<UNK>" if the word is not in the vocabulary
		"""
		return self._to_idx.get(word, self._to_idx["<UNK>"])

	def to_text(self, idxs: list[int]) -> str:
		"""
		Convert a list of indices to text.
		:param idxs: List of indices to convert
		:return: Text corresponding to the indices
		"""
		return TreebankWordDetokenizer().detokenize([self.to_str(int(idx)) for idx in idxs])

	def to_str(self, idx: int) -> str:
		"""
		Convert an index to its word.
		:param idx: Index to convert
		:return: Word corresponding to the index or "<UNK>" if the index is not in the vocabulary
		"""
		return self._to_str.get(idx, "<UNK>")

	def in_vocab(self, word: str) -> bool:
		"""
		Check if a word is in the vocabulary.
		:param word: Word to check
		:return: True if the word is in the vocabulary, False otherwise
		"""
		return word in self._to_idx

	def __str__(self):
		"""
		Return a string representation of the vocabulary.
		:return: String representation of the vocabulary
		"""
		return str({word: self.word_counts[word] for _, word in self._to_str.items()})


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


def data_loader(ann_file: str, img_dir: str, save_captions=False, vocab_threshold=2, vocab_file: str = None,
				save_vocab=False, transform=None, batch_size=32, num_workers=4, shuffle=True,
				pin_memory=True) -> DataLoader:
	"""
	Create a DataLoader for the Flickr8k dataset.

	:param ann_file: Path to the annotation file.
	:param img_dir: Path to the directory containing the images.
	:param save_captions: Whether to save the captions to a CSV file.
	:param vocab_threshold: Minimum frequency of a word to be included in the vocabulary.
	:param vocab_file: Path to the vocabulary file.
	:param save_vocab: Whether to save the vocabulary to a file.
	:param transform: Transform to apply to the images.
	:param batch_size: Number of samples per batch.
	:param num_workers: Number of subprocesses to use for data loading.
	:param shuffle: Whether to shuffle the data.
	:param pin_memory: Whether to pin memory.

	:return: DataLoader for the Flickr8k dataset.
	"""
	dataset = FlickerDataset(ann_file, img_dir, save_captions, vocab_threshold, vocab_file, save_vocab,
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

	device_ = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	logger.info(f"Device: {device_}")

	dataloader = data_loader(ann_file_, root_dir_, transform=transform_, vocab_file="vocab.pkl", pin_memory=True, num_workers=8)

	for i, (images_, captions_) in enumerate(dataloader):
		print(f"Images shape: {images_.size()}")
		print(f"Captions shape: {captions_.size()}")

		# Display the first image and caption
		print(f"Image from batch:\n {images_[0]}")

		img_ = images_[0].permute(1, 2, 0).numpy()  # Permute dimensions to (H, W, C)
		plt.imshow(img_)
		plt.show()

		print(f"Caption: \n{captions_[0]}")
		print(f"Text of the caption: {dataloader.dataset.vocab.to_text(captions_[0])}")
		break
