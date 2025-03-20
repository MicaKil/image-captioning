import torch.nn as nn


class EncoderBase(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def set_requires_grad(self, fine_tune: str) -> None:
        """
        Set the requires_grad attribute of the parameters based on the fine_tune argument.
        :param fine_tune: String indicating the fine-tuning strategy. Can be "full", "partial", or "none".
        :return:
        """
        if fine_tune == "full":
            return

        # Freeze all layers except the last two layers of the ResNet-50 model
        for param in self.resnet.parameters():
            param.requires_grad = False

        if fine_tune == "partial":
            # Unfreeze the last two layers of the ResNet-50 model
            for layer in list(self.resnet.children())[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
