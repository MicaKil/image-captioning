import logging

import torch
from torch.utils.data import random_split
from torchvision.transforms import v2

from constants import *
from dataset.flickr_dataloader import FlickerDataLoader
from dataset.flickr_dataset import FlickerDataset

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# full dataset
ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])

VOCAB_THRESHOLD = 2

# split dataset

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

# dataloaders

BATCH_SIZE = 32
NUM_WORKERS = 2
SHUFFLE = True
PIN_MEMORY = True

if __name__ == "__main__":
	full_dataset = FlickerDataset(ann_file, img_dir, vocab_threshold=VOCAB_THRESHOLD, transform=transform)

	total_size = len(full_dataset)
	train_size = int(TRAIN_SIZE * total_size)
	val_size = int(VAL_SIZE * total_size)
	test_size = total_size - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

	logger.info(f"Split dataset sizes {{Train: {train_size}, Validation: {val_size}, Test: {test_size}}}")
	print(len(train_dataset))
	train_dataloader = FlickerDataLoader(train_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	test_dataloader = FlickerDataLoader(test_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	val_dataloader = FlickerDataLoader(val_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)

	imgs, captions, imgs_id = next(iter(train_dataloader))

	df = full_dataset.df

	print(f"Id: {imgs_id[0]}")
	print(f"Caption: {train_dataloader.vocab.to_text(captions[0])}")

	print(f"Captions: {df[df['image_id'] == imgs_id[0]]['caption'].values}")

	# print(f"Image:")
	#
	# # get image name
	# # print image
	# utils.show_img(imgs[0], mean=MEAN, std=STD)
