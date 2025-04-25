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
        Initialize the Encoder class.

        :param embed_dim: Size of the embedding vector.
        :param dropout: Dropout probability to apply in the projection layer.
        :param fine_tune: Fine-tuning strategy for the ResNet-50 model.
                          Options are "full" (train all layers), "partial" (train last two layers), or "none" (freeze all layers).
        """
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # Load pretrained ResNet-50 model.
        in_features = self.resnet.fc.in_features  # Get the number of input features for the FC layer.
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2]  # Remove the last two layers (AvgPool and FC).
        )  # Output shape: (batch_size, in_features, 7, 7).

        self.set_requires_grad(fine_tune)  # Set the requires_grad attribute based on the fine-tuning strategy.

        # Add a projection layer to reduce the feature dimensions to the embedding size.
        self.projection = nn.Sequential(
            nn.Conv2d(in_features, embed_dim, kernel_size=1),  # 1x1 convolution to adjust channel dimensions.
            nn.ReLU(),  # Apply ReLU activation.
            nn.Dropout2d(dropout)  # Apply spatial dropout.
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the encoder.

        :param image: Input image tensor of shape (batch_size, 3, 224, 224).
        :return: Feature tensor of shape (batch_size, embed_dim, 1, 1).
        """
        features = self.resnet(image)  # Extract features using ResNet-50. Shape: (batch_size, in_features, 7, 7).
        features = self.projection(features)  # Apply projection layer. Shape: (batch_size, embed_dim, 1, 1).
        return features
