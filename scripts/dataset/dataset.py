import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

from configs.config import logger
from constants import EOS, SOS
from scripts.dataset.vocabulary import Vocabulary


class CaptionDataset(Dataset):
    """
    Custom Dataset for loading Flickr images and captions.
    """

    def __init__(self, img_dir: str, df_captions: pd.DataFrame, vocab: Vocabulary, transform=None, target_transform=None):
        """
        :param img_dir: Path to the directory containing the images
        :param df_captions: DataFrame containing the image IDs and captions. Headers must be "image_id" and "caption".
        :param vocab: Vocabulary object to use if provided
        :param transform: Transform to apply to the images
        :param target_transform: Transform to apply to the target captions
        """
        logger.info("Initializing Dataset.")
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
