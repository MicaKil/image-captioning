import os

# PATHS ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
# dataset
FLICKR8K_DIR = "datasets/flickr8k"
FLICKR8K_ANN_FILE = "datasets/flickr8k/Flickr8k.token.txt"
FLICKR8K_CSV_FILE = "datasets/flickr8k/captions.csv"
FLICKR8K_IMG_DIR = "datasets/flickr8k/images"
# dataset splits
TRAIN_CSV = "datasets/flickr8k/train_80_2025-02-16.csv"
VAL_CSV = "datasets/flickr8k/val_10_2025-02-16.csv"
TEST_CSV = "datasets/flickr8k/test_10_2025-02-16.csv"
# checkpoints
CHECKPOINT_DIR = "checkpoints/basic"
BASIC_RESULTS = "results/basic"

# VOCABULARY -----------------------------------------------------------------------------------------------------------
# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# TEST_PIC
PATH_ALVARITO = "datasets/mine/2025-02-21_22-33.png"
