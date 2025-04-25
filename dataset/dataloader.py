import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from constants import PAD
from dataset.dataset import CaptionDataset


class CaptionLoader(DataLoader):
    """
    Custom DataLoader for the Image Captioning task.
    Extends the PyTorch DataLoader to include a custom collate function and additional dataset attributes.
    """

    def __init__(self, dataset: CaptionDataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True):
        """
        Initialize the DataLoader for the dataset.

        :param dataset: CaptionDataset object to load.
        :param batch_size: Number of samples per batch (default: 64).
        :param num_workers: Number of subprocesses to use for data loading (default: 4).
        :param shuffle: Whether to shuffle the data (default: True).
        :param pin_memory: Whether to pin memory for faster data transfer to GPU (default: True).
        """
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                         collate_fn=Collate(dataset.vocab.str_to_idx(PAD)))
        self.vocab = dataset.vocab  # Vocabulary object associated with the dataset.
        self.annotations = dataset.annotations  # Annotations associated with the dataset.


class Collate:
    """
    Collate function to pad sequences and move tensors to the specified device.
    Used to process batches of data before they are returned by the DataLoader.
    """

    def __init__(self, pad_idx: int):
        """
        Initialize the collate function.

        :param pad_idx: Index of the padding token in the vocabulary.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Collate function to process a batch of samples.

        :param batch: List of samples, where each sample is a tuple (image, caption, image_id).
        :return: Tuple (images, captions, image_ids):
                 - images: Tensor of stacked image tensors.
                 - captions: Padded tensor of caption sequences.
                 - image_ids: List of image IDs corresponding to the samples.
        """
        images, captions, image_ids = zip(*batch)  # Unpack the batch into images, captions, and IDs.
        images = torch.stack(images)  # Stack image tensors into a single tensor.
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)  # Pad captions to the same length.
        return images, captions, image_ids
