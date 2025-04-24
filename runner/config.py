import torch
from torchvision.transforms import v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform_resize = (256, 256)
TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize(transform_resize),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

# dataloaders
NUM_WORKERS = 12
SHUFFLE = True
PIN_MEMORY = True

# run
PROJECT = "image-captioning-v1"
TAGS = ["transformer", "flickr8k", "best-attn-config"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_scheduler = False
eval_bleu4 = False

CONFIG = {
    "model": TAGS[0],
    "encoder": "resnet50",
    "decoder": "Attention" if TAGS[0] == "transformer" else "LSTM",
    "batch_size": 256,
    "transform_resize": transform_resize,
    "embed_size": 512,
    "hidden_size": 512,
    "num_layers": 2,
    "num_heads": 2 if TAGS[0] == "transformer" else None,
    "encoder_dropout": 0.1,
    "dropout": 0.5,  # decoder dropout
    "fine_tune_encoder": "partial",
    "encoder_lr": 0.00001,
    "decoder_lr": 0.0001,
    "criterion": "CrossEntropyLoss",
    "optimizer": "AdamW" if TAGS[0] == "transformer" else "Adam",
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
    } if TAGS[1] == "coco" else {
        "name": "flickr8k",
        "version": "2025-02-16",
        "split": {
            "train": 80,
            "val": 10,
            "test": 10
        }
    },
    "vocab": {
        "freq_threshold": 3,
        "tokenizer": "word",  # "sp-bpe",
        "vocab_size": 6500 if TAGS[1] == "coco" else 3500
    },
    "max_caption_len": 40,
    "temperature": 0,
    "beam_size": 5,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 5,
    } if use_scheduler else None,
    "eval_bleu4": {
        "step": 5
    } if eval_bleu4 else None,
    "allow_rl_switch": False
}
