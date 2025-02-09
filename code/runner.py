import logging
import os.path

import torch
from torch.utils.data import random_split
from torchvision.transforms import v2

import wandb
from constants import *
from dataset.flickr_dataloader import FlickerDataLoader
from dataset.flickr_dataset import FlickerDataset
from models.basic import ImageCaptioning
from test import test
from train import train
from utils import date_str

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# RUN CONFIG -----------------------------------------------------------------------------------------------------------

# transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
	v2.ToImage(),
	v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=MEAN, std=STD),
])

VOCAB_THRESHOLD = 2

# split dataset
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

# dataloaders
BATCH_SIZE = 32
NUM_WORKERS = 8
SHUFFLE = True
PIN_MEMORY = True

# model param
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.1
FREEZE_ENCODER = True

# training params
ENCODER_LR = 1e-4
DECODER_LR = 1e-4
MAX_EPOCHS = 1
PATIENCE = None
CALC_BLEU = False
MAX_CAPTION_LEN = 30

GRAD_MAX_NORM = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# run params
# MODEL_PATH = "checkpoints/basic/best_val_2025-02-07_19-07-00.pt"
MODEL_PATH = None

# wandb
PROJECT = "image-captioning"

# ----------------------------------------------------------------------------------------------------------------------

def run(crete_dataset=False, train_model=True, test_model=True, model_path: str = None, save_results=True):
	if crete_dataset:
		ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
		img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))

		full_dataset = FlickerDataset(ann_file, img_dir, vocab_threshold=VOCAB_THRESHOLD, transform=TRANSFORM)
		torch.save(full_dataset, os.path.join(ROOT, f"datasets/flickr8k/full_dataset_{date_str()}.pt"))
	else:
		full_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/full_dataset_2025-02-07.pt"),
								  weights_only=False)
		vocab = full_dataset.vocab
		pad_idx = vocab.to_idx(PAD)

	if crete_dataset:
		total_size = len(full_dataset)
		train_size = int(TRAIN_SIZE * total_size)
		val_size = int(VAL_SIZE * total_size)
		test_size = total_size - train_size - val_size

		train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

		torch.save(train_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/train_dataset_s-{int(TRAIN_SIZE * 100)}_{date_str()}.pt"))
		torch.save(val_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/val_dataset_s-{int(VAL_SIZE * 100)}_{date_str()}.pt"))
		torch.save(test_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/test_dataset_s-{int(TEST_SIZE * 100)}_{date_str()}.pt"))
	else:
		train_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s-80_2025-02-07.pt"),
								   weights_only=False)
		val_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s-10_2025-02-07.pt"),
								 weights_only=False)
		test_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/test_dataset_s-10_2025-02-07.pt"),
								  weights_only=False)

	if train_model:
		train_dataloader = FlickerDataLoader(train_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		val_dataloader = FlickerDataLoader(val_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
	if test_model:
		test_dataloader = FlickerDataLoader(test_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)

	model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, len(vocab), DROPOUT, NUM_LAYERS, pad_idx, FREEZE_ENCODER)
	if model_path is not None:
		model.load_state_dict(torch.load(os.path.join(ROOT, model_path), weights_only=True))

	wandb_run = wandb.init(
		project=PROJECT,
		tags=["basic", "flickr8k", "config-test"],
		config={
			"encoder": "resnet50",
			"decoder": "LSTM",
			"batch_size": BATCH_SIZE,
			"embed_size": EMBED_SIZE,
			"hidden_size": HIDDEN_SIZE,
			"num_layers": NUM_LAYERS,
			"dropout": DROPOUT,
			"freeze_encoder": FREEZE_ENCODER,
			"encoder_learning_rate": ENCODER_LR,
			"decoder_learning_rate": DECODER_LR,
			"criterion": "CrossEntropyLoss",
			"optimizer": "Adam",
			"max_epochs": MAX_EPOCHS,
			"patience": PATIENCE,
			"gradient_clip": GRAD_MAX_NORM,
			"dataset": "flickr8k",
			"dataset_version": "2025-02-07",
			"vocab_threshold": VOCAB_THRESHOLD,
			"vocab_size": len(vocab),
		}
	)

	if train_model:
		criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
		optimizer = torch.optim.Adam([
			{"params": model.encoder.parameters(), "lr": ENCODER_LR},
			{"params": model.decoder.parameters(), "lr": DECODER_LR},
		])
		train(model, train_dataloader, val_dataloader, DEVICE, vocab, MAX_EPOCHS, criterion, optimizer, CHECKPOINT_DIR,
			  wandb_run, GRAD_MAX_NORM, PATIENCE, CALC_BLEU, MAX_CAPTION_LEN)

	if test_model:
		results, metrics = test(model, test_dataloader, vocab, DEVICE, MAX_CAPTION_LEN, wandb_run)
		if save_results:
			results_path = os.path.join(ROOT, f"{BASIC_RESULTS}/results_{date_str()}.csv")
			results.to_csv(results_path, index=False)
			with open(os.path.join(ROOT, f"{BASIC_RESULTS}/metrics_{date_str()}.txt"), "w") as f:
				f.write(str(metrics))
		# log results to wandb
		wandb_run.log(metrics)
		results_table = wandb.Table(dataframe=results)
		results_artifact = wandb.Artifact("test_results", type="evaluation", metadata={"metrics": metrics})
		results_artifact.add(results_table, "results")
		if save_results:
			results_artifact.add_file(results_path)
		wandb_run.log({"test_results": results_table})
		wandb_run.log_artifact(results_artifact)
	wandb_run.finish()


if __name__ == "__main__":
	run(crete_dataset=False, train_model=True, test_model=True, model_path=MODEL_PATH, save_results=True)
