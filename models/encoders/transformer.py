import torch
from torch import nn as nn
from torchvision import models as models
from torchvision.models import ResNet50_Weights

import models.encoders.base as base


class Encoder(base.Encoder):
    """
    Encoder class that uses a pretrained ResNet-50 model to extract features from images.
    """

    def __init__(self, embed_dim: int, dropout: float, fine_tune: str):
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
            *list(self.resnet.children())[:-2]  # Remove the last 2 layers (AvgPool and FC)
        )  # Output shape: (batch, in_features, 7, 7)

        self.set_requires_grad(fine_tune)

        self.projection = nn.Sequential(
            nn.Conv2d(in_features, embed_dim, kernel_size=1),  # Reduce to embed_dim (Shape: (batch, embed_dim, 1, 1))
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder
        :param image: Input image tensor of shape (batch_size, 3, 224, 224)
        :return: Feature tensor of shape (batch_size, embed_dim, 1, 1)
        """
        features = self.resnet(image)  # Shape: (batch_size, feature_dim, 7, 7)
        features = self.projection(features)  # Shape: (batch_size, embed_size, 1, 1)
        return features
