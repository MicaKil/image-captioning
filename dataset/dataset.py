import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

from constants import EOS, SOS
from dataset.vocabulary import Vocabulary


class CaptionDataset(Dataset):
    """
    Custom Dataset for loading Flickr and COCO images and captions.
    """

    def __init__(self, img_dir: str, df_captions: pd.DataFrame, vocab: Vocabulary, transform=None, target_transform=None):
        """
        Initialize the CaptionDataset. Any dataset can be used as long as it has a DataFrame with image file names and captions.

        :param img_dir: Path to the directory containing the images.
        :param df_captions: DataFrame containing the image file names and captions.
                            The DataFrame must have columns "file_name" and "caption".
        :param vocab: Vocabulary object used for encoding captions.
        :param transform: Optional transform to apply to the images.
        :param target_transform: Optional transform to apply to the target captions.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = df_captions
        self.file_names, self.captions = self.annotations["file_name"], self.annotations["caption"]
        self.vocab = vocab

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: Number of samples in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve a sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: A tuple containing:
                 - img: The image tensor (RGB format).
                 - caption: The encoded caption tensor with SOS and EOS tokens.
                 - file_name: The file name of the image.
        """
        # Load and decode the image from the file path
        img = decode_image(str(os.path.join(self.img_dir, self.file_names[idx])), mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)  # Apply the image transform if provided

        # Encode the caption with SOS and EOS tokens
        caption = [self.vocab.str_to_idx(SOS)] + self.vocab.encode_as_ids(self.captions[idx]) + [self.vocab.str_to_idx(EOS)]
        caption = torch.tensor(caption, dtype=torch.long)  # Convert the caption to a tensor

        if self.target_transform:
            caption = self.target_transform(caption)  # Apply the target transform if provided

        return img, caption, self.file_names[idx]
