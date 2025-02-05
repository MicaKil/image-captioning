import os

# PATHS ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
# dataset
FLICKR8K_ANN_FILE = "datasets/flickr8k/Flickr8k.token.txt"
FLICKR8K_CSV_FILE = "datasets/flickr8k/captions.csv"
FLICKR8K_IMG_DIR = "datasets/flickr8k/images"
# code
VOCAB_FILE = "code/dataset/vocab.pkl"

# VOCABULARY -----------------------------------------------------------------------------------------------------------
# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
