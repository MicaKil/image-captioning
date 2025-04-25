import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_V2_S_Weights

import models.encoders.base as base


class Encoder(base.Encoder):
    """
    Encoder class that uses a pretrained Swin Transformer to extract features from images.
    """

    def __init__(self, embed_dim: int, dropout: float, fine_tune: str):
        """
        Initialize the Swin Transformer encoder.

        :param embed_dim: Size of the embedding vector.
        :param dropout: Dropout probability to apply in the projection layer.
        :param fine_tune: Fine-tuning strategy for the Swin Transformer.
                          Options are "full" (train all layers), "partial" (train last two blocks), or "none" (freeze all layers).
        """
        super().__init__()

        # Load pretrained Swin-S model
        self.swin = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)

        # Remove classification head and get feature dimension
        self.features = self.swin.features
        in_channels = self.swin.head.in_features

        # Projection layer to match embed_dim
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),  # 1x1 convolution to adjust channel dimensions.
            nn.ReLU(),  # Apply ReLU activation.
            nn.Dropout2d(dropout)  # Apply spatial dropout.
        )

        self.set_requires_grad(fine_tune)  # Set gradient requirements based on fine-tuning strategy.

    def set_requires_grad(self, fine_tune: str) -> None:
        """
        Set the requires_grad attribute for the Swin Transformer layers based on the fine_tune argument.

        :param fine_tune: Fine-tuning strategy for the Swin Transformer.
                          Options are "full" (train all layers), "partial" (train last two blocks), or "none" (freeze all layers).
        """
        if fine_tune == "full":
            return  # All layers remain trainable.

        # Freeze all layers initially
        for param in self.features.parameters():
            param.requires_grad = False

        if fine_tune == "partial":
            # Unfreeze the last two Swin blocks
            for block in self.features[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the encoder.

        :param image: Input image tensor of shape (batch_size, 3, 256, 256).
        :return: Feature tensor of shape (batch_size, embed_dim, H, W).
        """
        features = self.features(image)  # Extract features using the Swin Transformer. Shape: (batch_size, in_channels, 7, 7).
        features = features.permute(0, 3, 1, 2)  # Convert to channels-first format. Shape: (batch_size, in_channels, H, W).
        features = self.projection(features)  # Apply projection layer. Shape: (batch_size, embed_dim, 7, 7).
        return features
