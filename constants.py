import os

# PATHS ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

# datasets
FLICKR8K_DIR = "data/flickr8k"
FLICKR8K_ANN_FILE = "data/flickr8k/Flickr8k.token.txt"
FLICKR8K_CSV_FILE = "data/flickr8k/captions.csv"
FLICKR8K_IMG_DIR = "data/flickr8k/images"

FLICKR_TRAIN_CSV = "data/flickr8k/train_80_2025-02-16.csv"
FLICKR_VAL_CSV = "data/flickr8k/val_10_2025-02-16.csv"
FLICKR_TEST_CSV = "data/flickr8k/test_10_2025-02-16.csv"

FLICKR_CORPUS = "data/flickr8k/corpus.txt"

COCO_DIR = "data/coco"
COCO_TRAIN_ANN = "data/coco/annotations/captions_train2014.json"
COCO_VAL_ANN = "data/coco/annotations/captions_val2014.json"
COCO_IMGS_DIR = "data/coco/images"  # combined train and val images

COCO_TRAIN_CSV = "data/coco/coco_train.csv"
COCO_VAL_CSV = "data/coco/coco_val_15.csv"
COCO_TEST_CSV = "data/coco/coco_test_15.csv"

COCO_TRAIN_PKL = "data/coco/coco_train.pkl"
COCO_VAL_PKL = "data/coco/coco_val_15.pkl"
COCO_TEST_PKL = "data/coco/coco_test_15.pkl"

COCO_CORPUS =  "data/coco/corpus.txt"

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
