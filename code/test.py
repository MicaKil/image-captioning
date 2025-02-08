# 44856031_0d82c2c7d1.jpg
import os.path

import torch

from caption import preprocess_image, gen_caption
from constants import ROOT, PAD, FLICKR8K_CSV_FILE, FLICKR8K_IMG_DIR, TEST_IMG
from dataset.flickr_dataset import FlickerDataset
from models.basic import ImageCaptioning
from runner import EMBED_SIZE, HIDDEN_SIZE, DROPOUT, NUM_LAYERS, FREEZE_ENCODER, VOCAB_THRESHOLD, TRANSFORM, MEAN, STD, \
	MAX_CAPTION_LEN, DEVICE
from utils import show_img

if __name__ == "__main__":
	ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
	img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))

	full_dataset = FlickerDataset(ann_file, img_dir, vocab_threshold=VOCAB_THRESHOLD, transform=TRANSFORM)
	vocab = full_dataset.vocab
	pad_idx = vocab.to_idx(PAD)

	model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, len(vocab), DROPOUT, NUM_LAYERS, pad_idx, FREEZE_ENCODER)
	model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/basic/best_val_2025-02-07_19-07-00.pt"),
									 weights_only=True))
	model.eval()

	img_path = os.path.join(ROOT, "datasets/flickr8k/images/44856031_0d82c2c7d1.jpg")
	img = preprocess_image(str(img_path), TRANSFORM)

	print("Generating image caption.")
	print(gen_caption(model, img, vocab, max_length=MAX_CAPTION_LEN, device=DEVICE, temperature=None))
	show_img(img, MEAN, STD)
