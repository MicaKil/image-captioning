import os

# PATHS ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
# dataset
FLICKR8K_DIR = "data/flickr8k"
FLICKR8K_ANN_FILE = "data/flickr8k/Flickr8k.token.txt"
FLICKR8K_CSV_FILE = "data/flickr8k/captions.csv"
FLICKR8K_IMG_DIR = "data/flickr8k/images"
# dataset splits
TRAIN_CSV = "data/flickr8k/train_80_2025-02-16.csv"
VAL_CSV = "data/flickr8k/val_10_2025-02-16.csv"
TEST_CSV = "data/flickr8k/test_10_2025-02-16.csv"
# checkpoints
CHECKPOINT_DIR = "checkpoints/"
RESULTS_DIR = "results/"

# VOCABULARY -----------------------------------------------------------------------------------------------------------
# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# TEST_PIC
PATH_ALVARITO = "data/mine/2025-02-21_22-33.png"
