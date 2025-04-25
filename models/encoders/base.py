import torch.nn as nn


class Encoder(nn.Module):
    """
    Base class for the Encoder. This class is inherited by different encoder implementations.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Encoder base class.

        :param args: Positional arguments for the encoder.
        :param kwargs: Keyword arguments for the encoder.
        """
        super().__init__()

    def set_requires_grad(self, fine_tune: str) -> None:
        """
        Set the requires_grad attribute of the parameters based on the fine_tune argument.

        :param fine_tune: String indicating the fine-tuning strategy.
                          Can be "full" (train all layers), "partial" (train last two layers), or "none" (freeze all layers).
        :return: None
        """
        if fine_tune == "full":
            # If fine-tuning is set to "full", all layers remain trainable.
            return

        # Freeze all layers of the ResNet-50 model
        for param in self.resnet.parameters():
            param.requires_grad = False

        if fine_tune == "partial":
            # Unfreeze the last two layers of the ResNet-50 model
            for layer in list(self.resnet.children())[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
