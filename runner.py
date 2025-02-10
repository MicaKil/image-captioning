import logging
import os.path

import torch
from torch.utils.data import random_split
from torchvision.transforms import v2
from wandb.sdk.wandb_run import Run

import wandb
from code.dataset.flickr_dataloader import FlickerDataLoader
from code.dataset.flickr_dataset import FlickerDataset
from code.dataset.vocabulary import Vocabulary
from code.models.basic import ImageCaptioning
from code.test import test
from code.train import train
from code.utils import date_str
from constants import ROOT, FLICKR8K_CSV_FILE, FLICKR8K_IMG_DIR, CHECKPOINT_DIR, PROJECT, PAD

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

# vocab
VOCAB_THRESHOLD = 3

# dataset
DATASET = "flickr8k"
DATASET_VERSION = "2025-02-09"

TRAIN_SIZE = 80
VAL_SIZE = 10
TEST_SIZE = 100 - TRAIN_SIZE - VAL_SIZE

# dataloaders
BATCH_SIZE = 32
NUM_WORKERS = 8
SHUFFLE = True
PIN_MEMORY = True

# model param
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.3
FREEZE_ENCODER = True

# training params
ENCODER_LR = 1e-4
DECODER_LR = 1e-4
MAX_EPOCHS = 1
PATIENCE = None
CALC_BLEU = False
MAX_CAPTION_LEN = 30

GRAD_MAX_NORM = 5.0

# run params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/basic/best_val_2025-02-09_23-07.pt"


# ----------------------------------------------------------------------------------------------------------------------

def run(create_dataset=False, train_model=True, test_model=True, model_path: str = None, save_results=True):
	date = date_str()

	# create or load dataset
	if create_dataset:
		ann_file = str(os.path.join(ROOT, FLICKR8K_CSV_FILE))
		img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))

		full_dataset = FlickerDataset(ann_file, img_dir, vocab_threshold=VOCAB_THRESHOLD, transform=TRANSFORM)
		torch.save(full_dataset, os.path.join(ROOT, f"datasets/flickr8k/full_dataset_{date}.pt"))

		total_size = len(full_dataset)
		train_size = int((TRAIN_SIZE / 100) * total_size)
		val_size = int((VAL_SIZE / 100) * total_size)
		test_size = total_size - train_size - val_size

		train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

		# save new datasets
		torch.save(train_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/train_dataset_s-{TRAIN_SIZE}_{date}.pt"))
		torch.save(val_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/val_dataset_s-{VAL_SIZE}_{date}.pt"))
		torch.save(test_dataset,
				   os.path.join(ROOT, f"datasets/flickr8k/test_dataset_s-{TEST_SIZE}_{date}.pt"))
	else:
		full_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/full_dataset_2025-02-09.pt"),
								  weights_only=False)
		train_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/train_dataset_s-80_2025-02-09.pt"),
								   weights_only=False)
		val_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/val_dataset_s-10_2025-02-09.pt"),
								 weights_only=False)
		test_dataset = torch.load(os.path.join(ROOT, "datasets/flickr8k/test_dataset_s-10_2025-02-09.pt"),
								  weights_only=False)

	# create or load model
	vocab = full_dataset.vocab
	pad_idx = vocab.to_idx(PAD)
	wandb_run = init_wandb_run(vocab)
	config = wandb_run.config

	model = ImageCaptioning(config["embed_size"], config["hidden_size"], config["vocab_size"],
							config["dropout"], config["num_layers"], pad_idx,
							config["freeze_encoder"])
	if model_path is not None:
		logger.info(f"Loading model from {model_path}")
		model.load_state_dict(torch.load(os.path.join(ROOT, model_path), weights_only=True))

	if train_model:
		train_dataloader = FlickerDataLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		val_dataloader = FlickerDataLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
		optimizer = torch.optim.Adam([
			{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
			{"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
		])
		train(model, train_dataloader, val_dataloader, DEVICE, criterion, optimizer, CHECKPOINT_DIR, wandb_run,
			  MAX_CAPTION_LEN)

	if test_model:
		# create test dataloader
		test_dataloader = FlickerDataLoader(test_dataset, BATCH_SIZE, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
		# init wandb run if not already done
		test(model, test_dataloader, DEVICE, MAX_CAPTION_LEN, save_results, wandb_run)

	if create_dataset:
		log_dataset(wandb.Artifact(f"{DATASET}_full_dataset", type="dataset",
								   metadata={"version": DATASET_VERSION}),
					os.path.join(ROOT, f"datasets/flickr8k/full_dataset_{date}.pt"), wandb_run)
		log_dataset(wandb.Artifact(f"{DATASET}_train_dataset", type="dataset",
								   metadata={"version": DATASET_VERSION}),
					os.path.join(ROOT, f"datasets/flickr8k/train_dataset_s-{TRAIN_SIZE}_{date}.pt"), wandb_run)
		log_dataset(wandb.Artifact(f"{DATASET}_val_dataset", type="dataset", metadata={"version": DATASET_VERSION}),
					os.path.join(ROOT, f"datasets/flickr8k/val_dataset_s-{VAL_SIZE}_{date}.pt"), wandb_run)
		log_dataset(wandb.Artifact(f"{DATASET}_test_dataset", type="dataset",
								   metadata={"version": DATASET_VERSION}),
					os.path.join(ROOT, f"datasets/flickr8k/test_dataset_s-{TEST_SIZE}_{date}.pt"), wandb_run)
	wandb_run.finish()


def init_wandb_run(vocab: Vocabulary) -> Run:
	"""
	Initialize wandb run
	:param vocab: Vocabulary of the dataset
	:return: Wandb run
	"""
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
			"encoder_lr": ENCODER_LR,
			"decoder_lr": DECODER_LR,
			"criterion": "CrossEntropyLoss",
			"optimizer": "Adam",
			"max_epochs": MAX_EPOCHS,
			"patience": PATIENCE,
			"gradient_clip": GRAD_MAX_NORM,
			"dataset": DATASET,
			"dataset_version": DATASET_VERSION,
			"vocab_threshold": VOCAB_THRESHOLD,
			"vocab_size": len(vocab),
		}
	)
	wandb_run.define_metric("train_loss", summary="min")
	wandb_run.define_metric("val_loss", summary="min")
	wandb_run.define_metric("val_BLEU-4", summary="max")
	wandb_run.define_metric("test_BLEU-1", summary="max")
	wandb_run.define_metric("test_BLEU-2", summary="max")
	wandb_run.define_metric("test_BLEU-4", summary="max")
	wandb_run.define_metric("test_CIDEr", summary="max")
	return wandb_run


def log_dataset(wandb_artifact: wandb.Artifact, dataset_path: str, wandb_run: Run):
	"""
	Log dataset to wandb
	:param wandb_artifact: Wandb artifact
	:param dataset_path: Path to the dataset
	:param wandb_run:
	"""
	wandb_artifact.add_file(dataset_path)
	wandb_run.log_artifact(wandb_artifact)


if __name__ == "__main__":
	run(create_dataset=False, train_model=False, test_model=True, model_path=MODEL_PATH, save_results=True)
