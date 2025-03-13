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
RUN_TAGS = ["transformer", "flickr8k"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_scheduler = True

RUN_CONFIG = {
    "model": RUN_TAGS[0],
    "encoder": "resnet50",
    "decoder": "Attention",
    "batch_size": 128,
    "embed_size": None,
    "hidden_size": 512,
    "num_layers": 3,
    "num_heads": 4 if RUN_TAGS[0] == "transformer" else None,
    "encoder_dropout": 0.1,
    "dropout": 0.5,  # decoder dropout
    "fine_tune_encoder": "partial",
    "encoder_lr": 0.00001,
    "decoder_lr": 0.0001,
    "criterion": "CrossEntropyLoss",
    "optimizer": "AdamW",
    "max_epochs": 100,
    "patience": 10,
    "gradient_clip": 2.0,
    "dataset": {
        "name": "coco",
        "version": "2025-02-26",
        "split": {
            "train": 75,
            "val": 15,
            "test": 15
        }
    },
    "vocab": {
        "freq_threshold": None,
        "tokenizer": "sp-bpe",
        "vocab_size": 8500
    },
    "max_caption_len": 60,
    "temperature": 0,
    "beam_size": 3,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 5,
    } if use_scheduler else None,
    "validation": {
        "bleu4": True,
        "bleu4_step": 5
    }
}
