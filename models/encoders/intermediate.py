import torch
from torch import nn as nn
from torchvision import models as models
from torchvision.models import ResNet50_Weights

from models.encoders.base import EncoderBase


class Encoder(EncoderBase):
    """
    Encoder class that uses a pretrained ResNet-50 model to extract features from images.
    """

    def __init__(self, embed_dim: int, dropout: float, fine_tune: str) -> None:
        """
        Constructor for the EncoderResnet class

        :param embed_dim: Size of the embedding vector
        :param dropout: Dropout probability
        :param fine_tune: Whether to fine-tune the last two layers of the ResNet-50 model
        """
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1]  # Remove the last FC layer (Classification layer)
        )

        self.set_requires_grad(fine_tune)

        # Add a linear layer to transform the features to the embedding size
        self.linear = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder

        :param image: Input image tensor of shape (batch_size, 3, 224, 224)
        :return: 1D feature vector of shape (batch_size, feature_dim)
        """
        features = self.resnet(image)  # Shape: (batch_size, 2048, 1, 1) where feature_dim=2048
        features = features.view(features.size(0), -1)  # Shape: (batch_size, 2048)
        features = self.linear(features)  # Shape: (batch_size, embed_size)
        return features
