import pandas as pd
import torch
import torchvision.transforms.v2 as v2


def load_captions(path: str, overwrite=False) -> pd.DataFrame:
	"""
	Load the captions from the annotation file
	:param path: Path to the annotation file
	:param overwrite: If True, overwrite the existing CSV file
	:return: DataFrame containing the image filenames and corresponding captions
	"""

	if path.endswith(".csv") and not overwrite:
		return pd.read_csv(path)

	annotations = []
	with open(path, "r") as f:
		for line in f:
			try:
				image_id, caption = line.strip().split("\t")
				image_id = image_id.split("#")[0]
				annotations.append({"image_id": image_id, "caption": caption})
			except Exception as e:
				print(f"Error processing line: {line}")
				print(e)

	# Convert to DataFrame
	df = pd.DataFrame(annotations)
	df.to_csv("../datasets/flickr8k/captions.csv", index=False)
	return df


if __name__ == "__main__":

	# root_dir = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\images"  # Directory where images are stored
	root_dir = "../datasets/flickr8k/images"
	# ann_file = "C:\\Users\\micae\\OneDrive\\image-captioning\\dataset\\flickr8k\\Flickr8k.token.txt"  # Annotation file
	ann_file = "../datasets/flickr8k/Flickr8k.token.txt"

	# Define image transformations
	transform = v2.Compose([
		v2.ToImage(),
		v2.Resize((224, 224)),  # Resize for CNN models
		v2.ToDtype(torch.float32, scale=True),  # Convert image to tensor
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
	])

	# Load the captions
	df_ = load_captions(ann_file, overwrite=True)
	print(df_.head())
