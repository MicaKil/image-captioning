import torch
from torch import nn as nn
from torchvision import models as models
from torchvision.models import ResNet50_Weights

import models.encoders.base as base


class Encoder(base.Encoder):
    """
    Encoder class that uses a pretrained ResNet-50 model to extract features from images.
    """

    def __init__(self, embed_dim: int, fine_tune: str):
        """
        Initialize the Encoder class.

        :param embed_dim: Size of the embedding vector.
        :param fine_tune: Fine-tuning strategy for the ResNet-50 model.
                          Options are "full" (train all layers), "partial" (train last two layers), or "none" (freeze all layers).
        """
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # Load pretrained ResNet-50 model.
        in_features = self.resnet.fc.in_features  # Get the number of input features for the FC layer.
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1]  # Remove the last fully connected (classification) layer.
        )

        self.set_requires_grad(fine_tune)  # Set the requires_grad attribute based on the fine-tuning strategy.

        # Add a linear layer to transform the features to the embedding size.
        self.linear = nn.Linear(in_features, embed_dim)
        self.linear.bias.data.zero_()  # Initialize the bias to zeros.
        self.linear.bias.data[self.banned_indices] = -1e9  # Set banned tokens' bias to -1e9 initially.

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        :param image: Input image tensor of shape (batch_size, 3, 224, 224).
        :return: 1D feature vector of shape (batch_size, embed_dim).
        """
        features = self.resnet(image)  # Extract features using ResNet-50. Shape: (batch_size, 2048, 1, 1).
        features = features.view(features.size(0), -1)  # Flatten the features. Shape: (batch_size, 2048).
        features = self.linear(features)  # Transform features to the embedding size. Shape: (batch_size, embed_dim).
        return features
