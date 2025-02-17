# transforms
import torch
from torchvision.transforms import v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])

# vocab
VOCAB_THRESHOLD = 3

# dataset
DATASET = "flickr8k"
DATASET_VERSION = "2025-02-16"
DATASET_SPLIT = {"train": 80, "val": 10, "test": 10}
TRAIN_PATH = "datasets/flickr8k/train_80_2025-02-16.pt"
VAL_PATH = "datasets/flickr8k/val_10_2025-02-16.pt"
TEST_PATH = "datasets/flickr8k/test_10_2025-02-16.pt"

# dataloaders
BATCH_SIZE = 64
NUM_WORKERS = 8
SHUFFLE = True
PIN_MEMORY = True

# model param
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5
FREEZE_ENCODER = True

# training
MAX_EPOCHS = 10
PATIENCE = None
MAX_CAPTION_LEN = 40
ENCODER_LR = 1e-4
DECODER_LR = 5e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = None
GRAD_MAX_NORM = 5.0

# run
PROJECT = "image-captioning-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	"max_caption_len": MAX_CAPTION_LEN
}

if SCHEDULER_PATIENCE is None:
	RUN_CONFIG["scheduler"] = None
else:
	RUN_CONFIG["scheduler"] = {
		"type": "ReduceLROnPlateau",
		"factor": SCHEDULER_FACTOR,
		"patience": SCHEDULER_PATIENCE,
	}

RUN_TAGS = ["basic", "flickr8k", "test-new-dataset"]
