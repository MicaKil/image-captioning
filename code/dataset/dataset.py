import torch
import torchvision.transforms.v2 as v2

# Paths
# root_dir = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\images"  # Directory where images are stored
# ann_file = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\Flickr8k.token.txt"  # Annotation file

root_dir = "../../dataset/flickr8k/images"
ann_file = "../../dataset/flickr8k/Flickr8k.token.txt"

# Define image transformations
transform = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),  # Resize for CNN models
	v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
	v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
