import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
	"""
	Encoder class that uses a pretrained ResNet-50 model to extract features from images, i.e., encode images into a
	fixed-size feature vector suitable for captioning.
	"""

	def __init__(self):
		super(Encoder, self).__init__()
		self.model = models.resnet50(pretrained=True)
		self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the last FC layer (Classification layer)
		self.model.eval()  # Set the model to evaluation mode (don't update weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the encoder
		:param x: Input image tensor of shape (batch_size, 3, 224, 224)
		:return: 1D feature vector of shape (batch_size, 2048)
		"""
		with torch.no_grad():  # No need to compute gradients
			features = self.model(x)
		return features.view(features.size(0), -1)  # Flatten the feature vector
