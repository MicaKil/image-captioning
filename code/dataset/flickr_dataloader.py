import logging
import os

import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from constants import PAD, ROOT, FLICKR8K_ANN_FILE, FLICKR8K_IMG_DIR
from dataset.flickr_dataset import FlickerDataset
from dataset.vocabulary import Vocabulary

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class FlickerDataLoader(DataLoader):
	"""
	Custom DataLoader for the Flickr8k dataset.
	"""
	def __init__(self,
				 ann_file: str,
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
				 ):
		"""
		Initialize the DataLoader for the Flickr8k dataset.

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
		"""
		dataset = FlickerDataset(ann_file, img_dir, save_captions, vocab_threshold, vocab, save_vocab, transform)
		logger.info(f"FlickerDataset loaded.")
		pad_idx = dataset.vocab.to_idx(PAD)
		logger.info(f"Initializing DataLoader.")
		super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
						 pin_memory=pin_memory, collate_fn=Collate(pad_idx))
		self.vocab = dataset.vocab
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
		images, captions = zip(*batch)
		images = torch.stack(images)
		captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
		return images, captions


if __name__ == "__main__":
	root_dir_ = os.path.join(ROOT, FLICKR8K_IMG_DIR)
	ann_file_ = os.path.join(ROOT, FLICKR8K_ANN_FILE)

	transform_ = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
	])

	dataloader = FlickerDataLoader(str(ann_file_),
								   str(root_dir_),
								   save_captions=True,
								   save_vocab=True,
								   transform=transform_,
								   num_workers=8,
								   pin_memory=True
								   )

	images_, captions_ = next(iter(dataloader))
	print(f"Images shape: {images_.size()}")  # (batch_size, C, H, W)
	print(f"Captions shape: {captions_.size()}")  # (batch_size, max_seq_len)

	# Display the first image and caption
	print(f"Image from batch:\n {images_[0]}")

	img_ = images_[0].permute(1, 2, 0).numpy()  # Permute dimensions to (H, W, C)
	plt.imshow(img_)
	plt.show()

	print(f"Caption: \n{captions_[0]}")
	print(f"Text of the caption: {dataloader.vocab.to_text(captions_[0])}")
