import logging

import torch
from torchvision.transforms import v2

from constants import *
from dataset.flickr_dataloader import FlickerDataLoader
from models.basic import ImageCaptioning
from train import train

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])

VOCAB_THRESHOLD = 2

# split dataset

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

# dataloaders

BATCH_SIZE = 32
NUM_WORKERS = 2
SHUFFLE = True
PIN_MEMORY = True

# model param
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.1
FREEZE_ENCODER = True

# training params
ENCODER_LR = 1e-4
DECODER_LR = 1e-4
MAX_EPOCHS = 10
PATIENCE = None
CALC_BLEU = False
MAX_CAPTION_LEN = 30

CLIP_GRAD = False
GRAD_MAX_NORM = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
	# ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
	# img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))

	# full_dataset = FlickerDataset(ann_file, img_dir, vocab_threshold=VOCAB_THRESHOLD, transform=TRANSFORM)
	# torch.save(full_dataset, os.path.join(ROOT, f"datasets/flickr8k/full_dataset_{date_str()}.pt"))
	full_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/full_dataset_2025-02-07.pt"), weights_only=False)
	vocab = full_dataset.vocab
	pad_idx = vocab.to_idx(PAD)

	# total_size = len(full_dataset)
	# train_size = int(TRAIN_SIZE * total_size)
	# val_size = int(VAL_SIZE * total_size)
	# test_size = total_size - train_size - val_size
	#
	# train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
	#
	# torch.save(train_dataset, os.path.join(ROOT, f"datasets/flickr8k/train_dataset_s-{int(TRAIN_SIZE*100)}_{date_str()}.pt"))
	# torch.save(val_dataset, os.path.join(ROOT, f"datasets/flickr8k/val_dataset_s-{int(VAL_SIZE*100)}_{date_str()}.pt"))
	# torch.save(test_dataset, os.path.join(ROOT, f"datasets/flickr8k/test_dataset_s-{int(TEST_SIZE*100)}_{date_str()}.pt"))

	train_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s-80_2025-02-07.pt"),
							   weights_only=False)
	val_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s-10_2025-02-07.pt"),
							 weights_only=False)
	test_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/test_dataset_s-10_2025-02-07.pt"),
							  weights_only=False)

	print(test_dataset.indices)

	train_dataloader = FlickerDataLoader(train_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	val_dataloader = FlickerDataLoader(val_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	test_dataloader = FlickerDataLoader(test_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)

	model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, len(vocab), DROPOUT, NUM_LAYERS, pad_idx, FREEZE_ENCODER)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
	optimizer = torch.optim.Adam([
		{"params": model.encoder.parameters(), "lr": ENCODER_LR},
		{"params": model.decoder.parameters(), "lr": DECODER_LR},
	])

	train(model, train_dataloader, val_dataloader, DEVICE, vocab, MAX_EPOCHS, criterion, optimizer, CHECKPOINT_DIR,
		  CLIP_GRAD, GRAD_MAX_NORM, PATIENCE, CALC_BLEU, MAX_CAPTION_LEN)
