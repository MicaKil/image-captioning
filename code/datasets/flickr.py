import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset


class FlickerDataset(Dataset):
	def __init__(self, root_dir, ann_file, threshold=2, transform=None):
		self.root_dir = root_dir
		self.df = load_captions(ann_file)
		self.transform = transform

		self.imgs = self.df["image_id"]
		self.captions = self.df["caption"]


def load_captions(path: str, overwrite=False) -> pd.DataFrame:
	"""
	Load the captions from the annotation file.
	Sample line from the annotation file: "1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way."

	:param path: Path to the annotation file
	:param overwrite: If True, overwrite the existing CSV file
	:return: DataFrame containing the image filenames and corresponding captions
	"""

	if path.endswith(".csv") and not overwrite:
		print("Loading captions from CSV file.")
		return pd.read_csv(path)

	print("Loading captions from text file.")
	annotations = extract_captions(path)

	# Convert to DataFrame
	df = pd.DataFrame(annotations)
	df.to_csv("../datasets/flickr8k/captions.csv", index=False)
	return df


def extract_captions(path: str) -> list:
	annotations = []
	with open(path, "r") as f:
		for line in f:
			image_id, caption = line.strip().split("\t")
			image_id = image_id.split("#")[0]
			annotations.append({"image_id": image_id, "caption": caption})
	return annotations


class Vocabulary:
	def __init__(self, threshold):
		self.threshold = threshold
		self.str2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
		self.idx2str = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

	def __len__(self):
		return len(self.str2idx)

	def tokenize(self, text):
		return []


if __name__ == "__main__":
	# root_dir = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\images"
	root_dir_ = "../../../datasets/flickr8k/images"
	# ann_file = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\Flickr8k.token.txt"
	ann_file_ = "../../../datasets/flickr8k/captions.csv"

	# Define image transformations
	transform_ = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
	])

	# Load the captions
	df_ = load_captions(ann_file_, overwrite=False)
	print(df_.head())
