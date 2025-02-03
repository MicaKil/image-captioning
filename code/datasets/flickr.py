import os.path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode


class FlickerDataset(Dataset):
	def __init__(self, ann_file: str, img_dir: str, vocab_threshold=2, transform=None, target_transform=None):
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

		df = load_captions(ann_file)
		self.imgs_ids, self.captions = df["image_id"], df["caption"]

		self.vocab = Vocabulary(vocab_threshold, self.captions)

	def __len__(self):
		return len(self.captions)

	def __getitem__(self, idx: int):
		img = decode_image(str(os.path.join(self.img_dir, self.imgs_ids[idx])), mode=ImageReadMode.RGB)
		if self.transform:
			img = self.transform(img)

		caption = [self.vocab.to_idx("<SOS>")] + self.vocab.to_idxs(self.captions[idx]) + [self.vocab.to_idx("<EOS>")]
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
		print("Loading captions from CSV file.")
		return pd.read_csv(path)

	print("Loading captions from text file.")
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
		for text in text_list:
			self.word_counts.update(self.tokenize_eng(text))

		idx = 4
		for word, count in self.word_counts.items():
			if count >= self.freq_threshold:
				self._to_idx[word] = idx
				self._to_str[idx] = word
				idx += 1

	def to_idxs(self, text: str) -> list[int]:
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
		return str({word: self.word_counts[word] for _, word in self._to_str.items()})


class Collate:
	def __init__(self, pad_idx: int):
		self.pad_idx = pad_idx

	def __call__(self, batch):
		images, captions = zip(*batch)
		images = torch.stack(images)
		captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
		return images, captions


def data_loader(ann_file: str, img_dir: str, vocab_threshold=2, transform=None, batch_size=16, num_workers=4,
				shuffle=True, pin_memory=True):
	dataset = FlickerDataset(ann_file, img_dir, vocab_threshold, transform)
	pad_idx = dataset.vocab.to_idx("<PAD>")
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
									   pin_memory=pin_memory, collate_fn=Collate(pad_idx))


if __name__ == "__main__":
	# root_dir = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\images"
	root_dir_ = "../../datasets/flickr8k/images"
	# ann_file = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\Flickr8k.token.txt"
	ann_file_ = "../../datasets/flickr8k/captions.csv"

	# Define image transformations
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	transform_ = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
		# v2.Normalize(mean=mean, std=std)  # Normalize image
	])

	print("Loading data...")
	dataloader = data_loader(ann_file_, root_dir_, transform=transform_)

	for i, (images_, captions_) in enumerate(dataloader):
		print(f"Images shape: {images_.size()}")
		print(f"Captions shape: {captions_.size()}")

		# Display the first image and caption
		print(images_[0].size())
		print(images_[0])

		img_ = images_[0].permute(1, 2, 0).numpy()
		# img_ = (img_ * std) + mean # Unnormalize the image
		plt.imshow(img_)
		plt.show()

		print("Caption:")
		print(captions_[0])
		print([dataloader.dataset.vocab.to_str(int(i)) for i in captions_[0]])
		break


