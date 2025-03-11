import torch
from torchvision.transforms import v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

# dataloaders
NUM_WORKERS = 4
SHUFFLE = True
PIN_MEMORY = True

# run
PROJECT = "image-captioning-v1"
TAGS = ["transformer", "flickr8k"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_scheduler = False

DEFAULT_CONFIG = {
    "model": TAGS[0],
    "encoder": "resnet50",
    "decoder": "LSTM",
    "batch_size": 64,
    "embed_size": None,
    "hidden_size": 512,
    "num_layers": 1,
    "num_heads": 2 if TAGS[0] == "transformer" else None,
    "encoder_dropout": 0.5,
    "dropout": 0.5,  # decoder dropout
    "fine_tune_encoder": "partial",
    "encoder_lr": 0.0001,
    "decoder_lr": 0.001,
    "criterion": "CrossEntropyLoss",
    "optimizer": "AdamW",
    "max_epochs": 100,
    "patience": 10,
    "gradient_clip": 2.0,
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
    "max_caption_len": 50,
    "temperature": 0,
    "beam_size": 0,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 5,
    } if use_scheduler else None,
    "validation": {
        "bleu4": False,
        "bleu4_step": None
    }
}

SWEEP_CONFIG = {
    "project": PROJECT,
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        # architecture
        "hidden_size": {
            "values": [256, 512, 1024]
        },
        "num_layers": {
            "values": [1, 2, 3]
        },
        "num_heads": {
            "values": [2, 4, 8]
        },
        # regularisation
        "dropout": {
            "values": [0.1, 0.3, 0.5]
        },
        "encoder_dropout": {
            "values": [0.1, 0.3, 0.5]
        },
        # optimisation
        "encoder_lr": {
            "values": [0.0001, 0.0005, 0.00001, 0.00005]
        },
        "decoder_lr": {
            "values": [0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005]
        },
    }
}
