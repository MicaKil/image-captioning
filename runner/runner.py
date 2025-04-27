"""
Module for running training and testing pipelines on image captioning models.
"""

import os.path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from wandb.sdk import Config
from wandb.sdk.wandb_run import Run

import models.encoders.basic as b_encoder
import models.encoders.intermediate as i_encoder
import models.encoders.transformer as t_encoder
from config.config import logger
from constants import ROOT, CHECKPOINT_DIR, PAD, RESULTS_DIR
from dataset.dataloader import CaptionLoader
from dataset.dataset import CaptionDataset
from dataset.vocabulary import Vocabulary
from models import intermediate, transformer, basic
from models.encoders import swin
from runner.config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY
from scheduler import SchedulerWrapper
from test import test
from train import train
from utils import date_str


class Runner:
    """
    Runner class responsible for setting up datasets, models, training, and testing pipelines.
    """

    def __init__(self, use_wandb: bool, create_ds: bool, save_ds: bool, train_model: bool, test_model: bool, checkpoint_pth: Optional[str],
                 img_dir: str, ds_splits: tuple[str, str, str], ds_dir: str, project: str, run_tags: list[str], run_config: dict):
        """
        Initialize the Runner with configuration parameters.

        :param use_wandb: Whether to use wandb for experiment tracking.
        :param create_ds: Flag for creating a new dataset.
        :param save_ds: Flag for saving the dataset to disk.
        :param train_model: Flag for training the model.
        :param test_model: Flag for testing the model.
        :param checkpoint_pth: Optional checkpoint path to load a saved model.
        :param img_dir: Directory path for dataset images.
        :param ds_splits: Tuple with paths for train, validation, and test dataframes.
        :param ds_dir: Directory for dataset files.
        :param project: Wandb project name.
        :param run_tags: List of tags for the run.
        :param run_config: Configuration dictionary for the run parameters.
        """
        self.use_wandb = use_wandb
        self.create_ds = create_ds
        self.save_ds = save_ds
        self.train_model = train_model
        self.test_model = test_model
        self.checkpoint = checkpoint_pth
        self.img_dir = img_dir
        self.ds_splits = ds_splits
        self.ds_dir = ds_dir
        self.project = project
        self.run_tags = run_tags
        self.run_config = run_config

    def run(self):
        """
        Run the training and testing pipeline.

        Sets up the wandb run, datasets, model, optimizer, scheduler, and triggers training and testing processes.
        """

        if self.train_model == self.test_model == False:
            raise ValueError("At least one of train_model or test_model must be True.")
        if self.test_model and not self.train_model and self.checkpoint is None:
            raise ValueError("If only testing a model, a saved model (checkpoint) must be provided.")

        date = date_str()
        if self.use_wandb:
            self.init_wandb_run()
            config = wandb.config
        else:
            config = self.run_config

        train_dataset, val_dataset, test_dataset, vocab = self.get_datasets(config, date)
        pad_idx = vocab.str_to_idx(PAD)
        model = self.get_model(config, vocab, pad_idx)
        print("Model summary:")
        print(model)
        save_dir = RESULTS_DIR + config["model"]

        batch_size = config["batch_size"]
        if self.train_model:
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if self.use_wandb:
                wandb.run.summary["trainable_parameters"] = parameter_count
            logger.info(f"Number of trainable parameters: {parameter_count}")

            # Initialize data loaders for training and validation sets.
            train_dataloader = CaptionLoader(train_dataset, batch_size, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            val_dataloader = CaptionLoader(val_dataset, batch_size, NUM_WORKERS, SHUFFLE, PIN_MEMORY)

            criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
            optim = get_optimizer(config, model)
            scheduler = get_scheduler(config, optim, config["encoder_lr"])

            best_path, best_state, _ = train(model, train_dataloader, val_dataloader, DEVICE, criterion, optim, scheduler,
                                             CHECKPOINT_DIR + config["model"], self.use_wandb, config, self.checkpoint)

            if self.test_model:
                # Test last model checkpoint.
                test_dataloader = CaptionLoader(test_dataset, batch_size, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
                test(model, test_dataloader, DEVICE, save_dir, "LAST", self.use_wandb, config)
                if self.use_wandb:
                    wandb.finish()

                # Test best model checkpoint.
                if best_path is not None:
                    if self.use_wandb:
                        self.init_wandb_run()
                        config = wandb.config
                    else:
                        config = self.run_config
                    best = self.get_model(config, vocab, pad_idx)
                    best_checkpoint = torch.load(best_path)
                    best.load_state_dict(best_checkpoint["model_state"])
                    test(best, test_dataloader, DEVICE, save_dir, "BEST", self.use_wandb, config)
                    if self.use_wandb:
                        wandb.log({"epoch": best_state["epoch"],
                                   "train_loss": best_state["train_loss"],
                                   "val_loss": best_state["best_val_loss"],
                                   "encoder_lr": best_state["lr"][0],
                                   "decoder_lr": best_state["lr"][1]})
                        wandb.log_model(path=best_path)
            wandb.finish()
            return

        if self.test_model and self.checkpoint is not None:
            logger.info(f"Testing model from checkpoint.")
            checkpoint = torch.load(os.path.join(ROOT, self.checkpoint), weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            test_dataloader = CaptionLoader(test_dataset, batch_size, NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            test(model, test_dataloader, DEVICE, save_dir, "checkpoint", self.use_wandb, config)

        if self.use_wandb:
            wandb.finish()

    def init_wandb_run(self) -> Run:
        """
        Initialize a new wandb run with defined metrics.

        :return: Wandb run object.
        """
        wandb_run = wandb.init(project=self.project, tags=self.run_tags, config=self.run_config)
        wandb_run.define_metric("train_loss", summary="min")
        wandb_run.define_metric("val_loss", summary="min")
        wandb_run.define_metric("val_BLEU-4", summary="max")
        wandb_run.define_metric("test_BLEU-1", summary="max")
        wandb_run.define_metric("test_BLEU-2", summary="max")
        wandb_run.define_metric("test_BLEU-4", summary="max")
        wandb_run.define_metric("test_CIDEr", summary="max")
        wandb_run.define_metric("epoch", summary="max")
        return wandb_run

    def get_datasets(self, config: dict | Config, date: Optional[str]) -> tuple[CaptionDataset, CaptionDataset, CaptionDataset, Vocabulary]:
        """
        Create or load the datasets based on the configuration.

        :param config: Run configuration dictionary or wandb Config.
        :param date: Date string for dataset versioning.
        :return: Tuple containing training, validation, testing datasets and the vocabulary.
        """
        img_dir = os.path.join(ROOT, self.img_dir)
        if self.create_ds:
            raise NotImplementedError("Creating new datasets is being refactored.")
        else:
            train_df, val_df, test_df = self.get_dataframes()

        tokenizer = config["vocab"]["tokenizer"]
        freq_threshold = config["vocab"]["freq_threshold"]
        match tokenizer:
            case "word":
                vocab_file = f"vocab_freq-{freq_threshold}.pt"
                if vocab_file in os.listdir(os.path.join(ROOT, self.ds_dir)):  # check if vocab file exists
                    vocab = Vocabulary(tokenizer, freq_threshold, sp_model_path=None)
                    vocab.load_dict(torch.load(os.path.join(ROOT, self.ds_dir, vocab_file)))
                else:
                    vocab = Vocabulary(tokenizer, freq_threshold, train_df["caption"], None)
            case "sp-bpe":
                sp_model = os.path.join(ROOT, self.ds_dir, f"{config["dataset"]["name"]}_{config["vocab"]["vocab_size"]}.model")
                vocab = Vocabulary(tokenizer, None, text=None, sp_model_path=sp_model)
            case _:
                raise ValueError("Invalid tokenizer type.")

        train_dataset = CaptionDataset(img_dir, train_df, vocab, transform=TRANSFORM)
        val_dataset = CaptionDataset(img_dir, val_df, vocab, transform=TRANSFORM)
        test_dataset = CaptionDataset(img_dir, test_df, vocab, transform=TRANSFORM)

        if self.save_ds and tokenizer == "word":
            # Save datasets and vocabulary to disk.
            self.save_datasets(None, train_dataset, val_dataset, test_dataset, date, config)
            vocab_dict = {
                "str_to_idx": vocab.stoi_dict,
                "idx_to_str": vocab.itos_dict,
                "word_counts": vocab.word_counts,
                "freq_threshold": vocab.freq_threshold
            }
            torch.save(vocab_dict, os.path.join(ROOT, f"{self.ds_dir}/vocab_freq-{vocab.freq_threshold}.pt"))
            # log datasets to wandb
            if self.use_wandb:
                self.log_datasets(date, False)
        return train_dataset, val_dataset, test_dataset, vocab

    def get_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the dataframes from the dataset splits based on the file extension

        :return: Tuple with the training, validation, and test dataframes
        """
        match os.path.splitext(self.ds_splits[0])[1]:
            case ".csv":
                train_df = pd.read_csv(os.path.join(ROOT, self.ds_splits[0]))
                val_df = pd.read_csv(os.path.join(ROOT, self.ds_splits[1]))
                test_df = pd.read_csv(os.path.join(ROOT, self.ds_splits[2]))
            case ".pkl":
                train_df = pd.read_pickle(os.path.join(ROOT, self.ds_splits[0]))
                val_df = pd.read_pickle(os.path.join(ROOT, self.ds_splits[1]))
                test_df = pd.read_pickle(os.path.join(ROOT, self.ds_splits[2]))
            case ".json":
                train_df = pd.read_json(os.path.join(ROOT, self.ds_splits[0]))
                val_df = pd.read_json(os.path.join(ROOT, self.ds_splits[1]))
                test_df = pd.read_json(os.path.join(ROOT, self.ds_splits[2]))
            case _:
                raise ValueError("Dataframe file format not recognized")
        return train_df, val_df, test_df

    def get_model(self, config: dict | Config, vocab: Vocabulary, pad_idx: int) -> nn.Module:
        """
        Build and return the image captioning model based on the configuration.

        :param config: Run configuration dictionary or wandb Config.
        :param vocab: Vocabulary object for tokenized captions.
        :param pad_idx: Padding token index.
        :return: Configured image captioning model.
        """
        embed_dim = config["embed_size"]
        fine_tune = config["fine_tune_encoder"]
        hidden_size = config["hidden_size"]
        decoder_dropout = config["dropout"]
        encoder_dropout = config["encoder_dropout"]
        num_layers = config["num_layers"]
        match config["encoder"]:
            case "resnet50":
                match config["model"]:
                    case "basic":
                        encoder = b_encoder.Encoder(embed_dim, fine_tune)
                        decoder = basic.Decoder(embed_dim, hidden_size, len(vocab), decoder_dropout, num_layers, pad_idx)
                        return basic.ImageCaptioner(encoder, decoder)
                    case "intermediate":
                        encoder = i_encoder.Encoder(embed_dim, encoder_dropout, fine_tune)
                        decoder = intermediate.Decoder(embed_dim, hidden_size, vocab, decoder_dropout, num_layers, pad_idx)
                        return intermediate.ImageCaptioner(encoder, decoder)
                    case "transformer":
                        encoder = t_encoder.Encoder(hidden_size, encoder_dropout, fine_tune)
                        config["actual_max_seq_len"] = self.max_seq_len(vocab)
                        return transformer.ImageCaptioner(encoder, vocab, hidden_size, num_layers, config["num_heads"],
                                                          config["actual_max_seq_len"], decoder_dropout)
                    case _:
                        raise ValueError(f"Model {config['model']} not recognized")
            case "swin":
                encoder = swin.Encoder(hidden_size, encoder_dropout, fine_tune)
                match config["model"]:
                    case "basic":
                        decoder = basic.Decoder(embed_dim, hidden_size, len(vocab), decoder_dropout, num_layers, pad_idx)
                        return basic.ImageCaptioner(encoder, decoder)
                    case "intermediate":
                        decoder = intermediate.Decoder(embed_dim, hidden_size, vocab, decoder_dropout, num_layers, pad_idx)
                        return intermediate.ImageCaptioner(encoder, decoder)
                    case "transformer":
                        config["actual_max_seq_len"] = self.max_seq_len(vocab)
                        return transformer.ImageCaptioner(encoder, vocab, hidden_size, num_layers, config["num_heads"],
                                                          config["actual_max_seq_len"], decoder_dropout)
                    case _:
                        raise ValueError(f"Model {config['model']} not recognized")
            case _:
                raise ValueError(f"Encoder {config['encoder']} not recognized")

    def save_annotations(self, config: dict, date: str, test_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Save dataframes to disk
        :param config: Run configuration
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
        :param test_df: Test dataframe to be saved
        :param train_df: Training dataframe to be saved
        :param val_df: Validation dataframe to be saved
        """
        dataset_splits = config['dataset']['split']
        train_df.to_csv(os.path.join(ROOT, self.ds_dir, f"train_{date}_{dataset_splits['train']}.csv"), header=["image_id", "caption"], index=False)
        val_df.to_csv(os.path.join(ROOT, self.ds_dir, f"val_{date}_{dataset_splits['val']}.csv"), header=["image_id", "caption"], index=False)
        test_df.to_csv(os.path.join(ROOT, self.ds_dir, f"test_{date}_{dataset_splits['test']}.csv"), header=["image_id", "caption"], index=False)

    def save_datasets(self, full_dataset: Optional[CaptionDataset], train_dataset: CaptionDataset, val_dataset: CaptionDataset,
                      test_dataset: CaptionDataset, date: str, config: dict):
        """
        Save datasets to disk
        :param full_dataset: Complete dataset
        :param test_dataset: Test dataset
        :param train_dataset: Training dataset
        :param val_dataset: Validation dataset
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
        :param config: Run configuration
        """
        if full_dataset is not None:
            torch.save(full_dataset, os.path.join(ROOT, f"{self.ds_dir}/full_dataset_{date}.pt"))
        dataset_splits = config["dataset"]["split"]
        torch.save(train_dataset, os.path.join(ROOT, f"{self.ds_dir}/train_{date}_{dataset_splits["train"]}.pt"))
        torch.save(val_dataset, os.path.join(ROOT, f"{self.ds_dir}/val_{date}_{dataset_splits["val"]}.pt"))
        torch.save(test_dataset, os.path.join(ROOT, f"{self.ds_dir}/test_{date}_{dataset_splits["test"]}.pt"))

    def log_datasets(self, date: str, has_full_ds: bool):
        """
        Log datasets to wandb.

        :param has_full_ds:
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
        """
        config = wandb.config
        dataset_name = config["dataset"]["name"]
        dataset_version = config["dataset"]["version"]
        if has_full_ds:
            self.log_dataset(
                wandb.Artifact(f"{dataset_name}_full_dataset", type="dataset", metadata={"version": dataset_version}),
                os.path.join(ROOT, f"{self.ds_dir}/full_dataset_{date}.pt")
            )
        dataset_splits = config["dataset"]["split"]
        self.log_dataset(
            wandb.Artifact(f"{dataset_name}_train_dataset", type="dataset", metadata={"version": dataset_version}),
            os.path.join(ROOT, f"{self.ds_dir}/train_{date}_{dataset_splits["train"]}.pt")
        )
        self.log_dataset(
            wandb.Artifact(f"{dataset_name}_val_dataset", type="dataset", metadata={"version": dataset_version}),
            os.path.join(ROOT, f"{self.ds_dir}/val_{date}_{dataset_splits["val"]}.pt")
        )
        self.log_dataset(
            wandb.Artifact(f"{dataset_name}_test_dataset", type="dataset", metadata={"version": dataset_version}),
            os.path.join(ROOT, f"{self.ds_dir}/test_{date}_{dataset_splits["test"]}.pt")
        )

    @staticmethod
    def log_dataset(artifact: wandb.Artifact, dataset_path: str):
        """
        Log dataset to wandb.

        :param artifact: Wandb artifact.
        :param dataset_path: Path to the dataset.
        """
        artifact.add_file(dataset_path)
        wandb.log_artifact(artifact)

    def max_seq_len(self, vocab: Vocabulary):
        """
        Calculate the maximum sequence length in the dataset.

        :param vocab:
        :return: Maximum sequence length
        """
        train_df, val_df, test_df = self.get_dataframes()
        max_len = max(train_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                      val_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                      test_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max())
        return max_len + 2  # add 2 for start and end tokens


def get_scheduler(config: dict | Config, optim: torch.optim, encoder_lr: float) -> Optional[SchedulerWrapper]:
    """
    Get the scheduler based on the configuration.

    :param config: The run configuration
    :param optim: The optimizer to be used in training
    :param encoder_lr: The initial learning rate for the encoder
    :return: The scheduler to be used in training
    """
    scheduler = config["scheduler"]
    if scheduler is not None:
        return SchedulerWrapper(ReduceLROnPlateau(optim, mode="min", factor=scheduler["factor"], patience=scheduler["patience"], min_lr=1e-6),
                                encoder_lr)
    return None


def get_optimizer(config: dict | Config, model: nn.Module) -> torch.optim:
    """
    Get the optimizer based on the configuration.

    :param config: The run configuration
    :param model: The model to be trained
    :return:
    """
    encoder_lr = config["encoder_lr"]
    decoder_lr = config["decoder_lr"]
    match config["model"]:
        case "basic" | "intermediate":
            params = [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.decoder.parameters(), "lr": decoder_lr}
            ]
        case "transformer":
            params = [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.seq_embedding.parameters(), "lr": decoder_lr},
                {"params": model.decoder_layers.parameters(), "lr": decoder_lr},
                {"params": model.output_layer.parameters(), "lr": decoder_lr}
            ]
        case _:
            raise ValueError(f"Model {config['model']} not recognized")

    match config["optimizer"]:
        case "Adam":
            return Adam(params)
        case "AdamW":
            return AdamW(params)
        case _:
            raise ValueError(f"Optimizer {config['optimizer']} not recognized")
