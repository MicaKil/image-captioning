import os

import torch
from torchvision.transforms import v2

# Environment variables with defaults
DATASET = os.getenv("DATASET_NAME", "flickr8k")
DATASET_VERSION = os.getenv("DATASET_VERSION", "2025-02-16")
DATASET_SPLIT = {
	"train": int(os.getenv("DATASET_SPLIT_TRAIN", 80)),
	"val": int(os.getenv("DATASET_SPLIT_VAL", 10)),
	"test": int(os.getenv("DATASET_SPLIT_TEST", 10))
}
TRAIN_PATH = "datasets/flickr8k/train_80_2025-02-16.pt"
VAL_PATH = "datasets/flickr8k/val_10_2025-02-16.pt"
TEST_PATH = "datasets/flickr8k/test_10_2025-02-16.pt"

# Model parameters
EMBED_SIZE = int(os.getenv("EMBED_SIZE", 256))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 512))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", 1))
DROPOUT = float(os.getenv("DROPOUT", 0.3))
FREEZE_ENCODER = os.getenv("FREEZE_ENCODER", "True") == "True"

# Training configuration
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 10))
PATIENCE = int(os.getenv("PATIENCE", 10))
MAX_CAPTION_LEN = int(os.getenv("MAX_CAPTION_LEN", 30))
ENCODER_LR = float(os.getenv("ENCODER_LR", 1e-4))
DECODER_LR = float(os.getenv("DECODER_LR", 1e-4))
GRAD_MAX_NORM = float(os.getenv("GRAD_MAX_NORM", 5.0))

# Scheduler config
SCHEDULER_FACTOR = float(os.getenv("SCHEDULER_FACTOR", 0.5))
SCHEDULER_PATIENCE = os.getenv("SCHEDULER_PATIENCE", None)
SCHEDULER_PATIENCE = int(SCHEDULER_PATIENCE) if SCHEDULER_PATIENCE else None

# System config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
SHUFFLE = os.getenv("SHUFFLE", "True") == "True"
PIN_MEMORY = os.getenv("PIN_MEMORY", "True") == "True"
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORM = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])
VOCAB_THRESHOLD = 3
PROJECT = "image-captioning-v1"
RUN_TAGS = ["basic", "flickr8k"]

# Generated RUN_CONFIG
RUN_CONFIG = {
	"encoder": "resnet50",
	"decoder": "LSTM",
	"batch_size": BATCH_SIZE,
	"embed_size": EMBED_SIZE,
	"hidden_size": HIDDEN_SIZE,
	"num_layers": NUM_LAYERS,
	"dropout": DROPOUT,
	"freeze_encoder": FREEZE_ENCODER,
	"encoder_lr": ENCODER_LR,
	"decoder_lr": DECODER_LR,
	"criterion": "CrossEntropyLoss",
	"optimizer": "Adam",
	"max_epochs": MAX_EPOCHS,
	"patience": PATIENCE,
	"gradient_clip": GRAD_MAX_NORM,
	"dataset": {
		"name": DATASET,
		"version": DATASET_VERSION,
		"split": DATASET_SPLIT
	},
	"vocab": {
		"freq_threshold": VOCAB_THRESHOLD
	},
	"max_caption_len": MAX_CAPTION_LEN,
	"scheduler": {
		"type": "ReduceLROnPlateau",
		"factor": SCHEDULER_FACTOR,
		"patience": SCHEDULER_PATIENCE,
	} if SCHEDULER_PATIENCE else None
}
