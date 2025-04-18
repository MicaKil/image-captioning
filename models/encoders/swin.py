import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_V2_S_Weights

from models.encoders.base import EncoderBase


class SwinEncoder(EncoderBase):
    """
    Encoder class that uses a pretrained Swin Transformer to extract features from images.
    """

    def __init__(self, embed_dim: int, dropout: float, fine_tune: str):
        """
        Constructor for SwinTransformer encoder

        :param embed_dim: Size of the embedding vector
        :param dropout: Dropout probability
        :param fine_tune: Fine-tuning strategy ('full', 'partial', or 'none')
        """
        super().__init__()

        # Load pretrained Swin-S model
        self.swin = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)

        self.attention_weights = []
        self._register_attention_hooks()

        # Remove classification head and get feature dimension
        self.features = self.swin.features
        in_channels = self.swin.head.in_features

        # Projection layer to match embed_dim
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        self.set_requires_grad(fine_tune)

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from Swin blocks"""

        def hook_fn(output):
            # output[1] contains attention weights in Swin blocks
            if len(output) > 1:
                self.attention_weights.append(output[1].detach().cpu())

        # Attach hooks to all Swin Transformer blocks
        for block in self.swin.features:
            if hasattr(block, "attn"):
                block.coco.register_forward_hook(hook_fn)

    def set_requires_grad(self, fine_tune: str) -> None:
        """
        Set gradient requirements for Swin layers
        """
        if fine_tune == "full":
            return  # All layers trainable

        # Freeze all layers initially
        for param in self.features.parameters():
            param.requires_grad = False

        if fine_tune == "partial":
            # Unfreeze last two Swin blocks
            for block in self.features[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder

        :param image: Input tensor (batch_size, 3, 256, 256)
        :return: Feature tensor (batch_size, embed_dim, H, W)
        """
        self.attention_weights = []  # Reset on each forward pass
        features = self.features(image)  # (batch_size, in_channels, 7, 7)
        features = features.permute(0, 3, 1, 2)  # Convert to channels-first: (batch_size, in_channels, H, W)
        features = self.projection(features)  # (batch_size, embed_dim, 7, 7)
        return features
