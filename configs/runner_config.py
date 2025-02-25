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

# dataloaders
NUM_WORKERS = 4
SHUFFLE = True
PIN_MEMORY = True

# run
PROJECT = "image-captioning-v1"
RUN_TAGS = ["intermediate", "flickr8k"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_scheduler = False

RUN_CONFIG = {
    "model": RUN_TAGS[0],
    "encoder": "resnet50",
    "decoder": "LSTM",
    "batch_size": 64,
    "embed_size": 256,
    "hidden_size": 256,
    "num_layers": 3,
    "encoder_dropout": 0.2,
    "dropout": 0.5,  # decoder dropout
    "freeze_encoder": False,
    "encoder_lr": 0.00001,
    "decoder_lr": 0.00001,
    "criterion": "CrossEntropyLoss",
    "optimizer": "Adam",
    "max_epochs": 100,
    "patience": 30,
    "gradient_clip": None,
    "dataset": {
        "name": "flickr8k",
        "version": "2025-02-16",
        "split": {
            "train": 80,
            "val": 10,
            "test": 10
        }
    },
    "vocab": {
        "freq_threshold": 3
    },
    "max_caption_len": 30,
    "temperature": None,
    "beam_size": 5,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 10,
    } if use_scheduler else None,
    "validation": {
        "bleu4": True,
        "bleu4_step": 10
    }
}
