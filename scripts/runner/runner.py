import os.path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from wandb.sdk.wandb_run import Run

from config.config import logger
from constants import ROOT, CHECKPOINT_DIR, PAD, RESULTS_DIR
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.dataset import CaptionDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.models import basic, intermediate, transformer
from scripts.runner.config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY
from scripts.scheduler import SchedulerWrapper
from scripts.utils import date_str


class Runner:
    def __init__(self, use_wandb: bool, create_ds: bool, save_ds: bool, train_model: bool, test_model: bool, checkpoint_pth: Optional[str],
                 img_dir: str, ds_splits: tuple[str, str, str], ds_dir: str, project: str, run_tags: list[str], run_config: dict):
        """
        :param use_wandb: Whether to use wandb
        :param create_ds: Whether to create a new dataset or load an existing one. Saves the dataset to disk if a new one is created
        :param save_ds: Whether to save the datasets to disk
        :param train_model: Whether to train the model
        :param test_model: Whether to test the model
        :param checkpoint_pth: Tuple containing the model path and the model tag. If not None, a new model is created.
        :param img_dir: Path to the image directory of the dataset
        :param ds_splits: List with the paths to the train, val, and test dataframes
        :param ds_dir: Path to the directory where the datasets are
        :return:
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
        Run the training and testing pipeline
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

        train_dataset, val_dataset, test_dataset, vocab = self.get_ds(config, date)
        pad_idx = vocab.to_idx(PAD)
        model = self.get_model(config, vocab, pad_idx)
        save_dir = RESULTS_DIR + config["model"]

        if self.train_model:
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if self.use_wandb:
                wandb.run.summary["trainable_parameters"] = parameter_count
            logger.info(f"Number of trainable parameters: {parameter_count}")

            # dataloaders
            train_dataloader = CaptionLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            val_dataloader = CaptionLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)

            criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
            optimizer = get_optimizer(config, model)
            scheduler = get_scheduler(config, optimizer, config["encoder_lr"])

            best_path, best_state, _ = model.train_model(train_dataloader, val_dataloader, DEVICE, criterion, optimizer, scheduler,
                                                         CHECKPOINT_DIR + config["model"], self.use_wandb, config, self.checkpoint)

            if self.test_model:
                # test last model
                test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
                model.test_model(test_dataloader, DEVICE, save_dir, "last-model", self.use_wandb, config)
                if self.use_wandb:
                    wandb.finish()

                # test best model
                if best_path is not None:
                    if self.use_wandb:
                        self.init_wandb_run()
                        config = wandb.config
                    else:
                        config = self.run_config
                    best = self.get_model(config, vocab, pad_idx)
                    best_checkpoint = torch.load(os.path.join(ROOT, best_path))
                    best.load_state_dict(best_checkpoint["model_state"])
                    best.test_model(test_dataloader, DEVICE, save_dir, "best-model", self.use_wandb, config)
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
            test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            model.test_model(test_dataloader, DEVICE, save_dir, "checkpoint", self.use_wandb, config)

        if self.use_wandb:
            wandb.finish()

    def init_wandb_run(self) -> Run:
        """
        Initialize wandb run
        :return: Wandb run
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

    def get_ds(self, config: dict, date: str) -> tuple[CaptionDataset, CaptionDataset, CaptionDataset, Vocabulary]:
        """
        Creates or loads the datasets.
        :param config: The run configuration.
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names.
        :return: Tuple with the training, validation, and test datasets and the vocabulary.
        """
        img_dir = os.path.join(ROOT, self.img_dir)
        if self.create_ds:
            raise NotImplementedError("Creating new datasets is being refactored.")
            # df_captions = load_flickr_captions( (os.path.join(ROOT, FLICKR8K_ANN_FILE)), True)
            # total_size = len(df_captions["image_id"].unique())
            # train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
            # val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
            # test_size = total_size - train_size - val_size
            # train_df, val_df, test_df = split_dataframe(df_captions, [train_size, val_size, test_size])
            # if save_ds:
            #     save_df(config, date, test_df, train_df, val_df)  # save new dataframes to disk
        else:
            train_df, val_df, test_df = self.get_dataframes()

        vocab_file = f"vocab_freq-{config["vocab"]["freq_threshold"]}.pt"
        if vocab_file in os.listdir(os.path.join(ROOT, self.ds_dir)): # check if vocab file exists
            vocab = Vocabulary(config["vocab"]["freq_threshold"])
            vocab.load_dict(torch.load(os.path.join(ROOT, self.ds_dir, vocab_file)))
        else:
            vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])

        train_dataset = CaptionDataset(img_dir, train_df, vocab, transform=TRANSFORM)
        val_dataset = CaptionDataset(img_dir, val_df, vocab, transform=TRANSFORM)
        test_dataset = CaptionDataset(img_dir, test_df, vocab, transform=TRANSFORM)

        if self.save_ds:
            # save datasets to disk
            self.save_datasets(None, train_dataset, val_dataset, test_dataset, date, config)
            # save vocab to disk
            vocab_dict = {
                "str_to_idx": vocab.str_to_idx,
                "idx_to_str": vocab.idx_to_str,
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

    def get_model(self, config: dict, vocab: Vocabulary, pad_idx: int) -> nn.Module:
        """
        Get the model based on the configuration
        :param config: The run configuration
        :param vocab: Vocabulary of the dataset
        :param pad_idx: Index of the padding token
        :return:
        """
        match config["model"]:
            case "basic":
                encoder = basic.Encoder(config["embed_size"], not config["freeze_encoder"])
                decoder = basic.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
                return basic.BasicImageCaptioner(encoder, decoder)
            case "intermediate":
                encoder = intermediate.Encoder(config["embed_size"], config["encoder_dropout"], not config["freeze_encoder"])
                decoder = intermediate.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"],
                                               pad_idx)
                return intermediate.IntermediateImageCaptioner(encoder, decoder)
            case "transformer":
                return transformer.ImageCaptioningTransformer(vocab, config["hidden_size"], config["num_layers"], config["num_heads"],
                                                              self.calc_max_sequence_length(vocab), config["encoder_dropout"],
                                                              config["dropout"], not config["freeze_encoder"])
            case _:
                raise ValueError(f"Model {config['model']} not recognized")

    def save_df(self, config: dict, date: str, test_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Save dataframes to disk
        :param config: Run configuration
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
        :param test_df: Test dataframe to be saved
        :param train_df: Training dataframe to be saved
        :param val_df: Validation dataframe to be saved
        :return:
        """
        train_df.to_csv(os.path.join(ROOT, self.ds_dir, f"train_{date}_{config['dataset']['split']['train']}.csv"), header=["image_id", "caption"],
                        index=False)
        val_df.to_csv(os.path.join(ROOT, self.ds_dir, f"val_{date}_{config['dataset']['split']['val']}.csv"), header=["image_id", "caption"],
                      index=False)
        test_df.to_csv(os.path.join(ROOT, self.ds_dir, f"test_{date}_{config['dataset']['split']['test']}.csv"), header=["image_id", "caption"],
                       index=False)

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
        :return:
        """
        if full_dataset is not None:
            torch.save(full_dataset, os.path.join(ROOT, f"{self.ds_dir}/full_dataset_{date}.pt"))
        torch.save(train_dataset, os.path.join(ROOT, f"{self.ds_dir}/train_{date}_{config["dataset"]["split"]["train"]}.pt"))
        torch.save(val_dataset, os.path.join(ROOT, f"{self.ds_dir}/val_{date}_{config["dataset"]["split"]["val"]}.pt"))
        torch.save(test_dataset, os.path.join(ROOT, f"{self.ds_dir}/test_{date}_{config["dataset"]["split"]["test"]}.pt"))

    def log_datasets(self, date: str, has_full_ds: bool):
        """
        Log datasets to wandb
        :param has_full_ds:
        :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
        :return:
        """
        config = wandb.config
        if has_full_ds:
            self.log_dataset(
                wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
                os.path.join(ROOT, f"{self.ds_dir}/full_dataset_{date}.pt")
            )
        self.log_dataset(
            wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
            os.path.join(ROOT, f"{self.ds_dir}/train_{date}_{config["dataset"]["split"]["train"]}.pt")
        )
        self.log_dataset(
            wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
            os.path.join(ROOT, f"{self.ds_dir}/val_{date}_{config["dataset"]["split"]["val"]}.pt")
        )
        self.log_dataset(
            wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
            os.path.join(ROOT, f"{self.ds_dir}/test_{date}_{config["dataset"]["split"]["test"]}.pt")
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

    def calc_max_sequence_length(self, vocab):
        """
        Calculate the maximum sequence length in the dataset
        :param vocab:
        :return: Maximum sequence length
        """
        train_df, val_df, test_df = self.get_dataframes()
        max_len = max(train_df["caption"].apply(lambda x: len(vocab.tokenize_eng(x))).max(),
                      val_df["caption"].apply(lambda x: len(vocab.tokenize_eng(x))).max(),
                      test_df["caption"].apply(lambda x: len(vocab.tokenize_eng(x))).max())
        return max_len + 2  # add 2 for start and end tokens


def get_scheduler(config: dict, optimizer: torch.optim, encoder_lr: float) -> Optional[SchedulerWrapper]:
    """
    Get the scheduler based on the configuration
    :param config: The run configuration
    :param optimizer: The optimizer to be used in training
    :param encoder_lr: The initial learning rate for the encoder
    :return:
    """
    if config["scheduler"] is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"])
        return SchedulerWrapper(scheduler, encoder_lr)
    return None


def get_optimizer(config: dict, model: nn.Module) -> torch.optim:
    """
    Get the optimizer based on the configuration.
    :param config: The run configuration
    :param model: The model to be trained
    :return:
    """
    match config["model"]:
        case "basic", "intermediate":
            params = [
                {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
                {"params": model.decoder.parameters(), "lr": config["decoder_lr"]}
            ]
        case "transformer":
            params = [
                {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
                {"params": model.seq_embedding.parameters(), "lr": config["decoder_lr"]},
                {"params": model.decoder_layers.parameters(), "lr": config["decoder_lr"]},
                {"params": model.output_layer.parameters(), "lr": config["decoder_lr"]}
            ]
        case _:
            raise ValueError(f"Model {config['model']} not recognized")

    match config["optimizer"]:
        case "Adam":
            return Adam(params)
        case "AdamW":
            return Adam(params)
