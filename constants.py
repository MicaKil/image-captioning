import os

# wandb
PROJECT = "image-captioning-v1"

# PATHS ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
# dataset
FLICKR8K_ANN_FILE = "datasets/flickr8k/Flickr8k.token.txt"
FLICKR8K_CSV_FILE = "datasets/flickr8k/captions.csv"
FLICKR8K_IMG_DIR = "datasets/flickr8k/images"
# code
VOCAB_FILE = "scripts/dataset/vocab.pkl"
# checkpoints
CHECKPOINT_DIR = "checkpoints/basic"
BASIC_RESULTS = "results/basic"

# VOCABULARY -----------------------------------------------------------------------------------------------------------
# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# TEST DATA ------------------------------------------------------------------------------------------------------------
TEST_IMG = "datasets/flickr8k/images/300222673_573fd4044b.jpg"
TEST_IMG_CAPTIONS = ["A man plays a song on the guitar for his cat.",
					 "A man plays a yellow guitar while a cat watches him.",
					 "A man plays his yellow guitar while staring at his cat.",
					 "A man, sitting by his computer, playing guitar to his cat.",
					 "The man is playing guitar and sitting with a cat."]
