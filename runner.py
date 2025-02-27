import os.path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from wandb.sdk.wandb_run import Run

from configs.config import logger
from configs.runner_config import TRANSFORM, DEVICE, NUM_WORKERS, SHUFFLE, PIN_MEMORY, RUN_CONFIG, PROJECT, RUN_TAGS
from constants import ROOT, FLICKR8K_IMG_DIR, CHECKPOINT_DIR, PAD, FLICKR8K_DIR, FLICKR8K_ANN_FILE, RESULTS_DIR, FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, \
    FLICKR_TEST_CSV
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.dataset import CaptionDataset
from scripts.dataset.helper import load_flickr_captions, split_dataframe
from scripts.dataset.vocabulary import Vocabulary
from scripts.models import basic, intermediate, transformer
from scripts.utils import date_str


def run(use_wandb: bool, create_ds: bool, save_ds: bool, train_model: bool, test_model: bool, saved_model: Optional[tuple[str, str]]):
    """
    Run the training and testing pipeline
    :param use_wandb: Whether to use wandb
    :param create_ds: Whether to create a new dataset or load an existing one. Saves the dataset to disk if a new one is created
    :param save_ds: Whether to save the datasets to disk
    :param train_model: Whether to train the model
    :param test_model: Whether to test the model
    :param saved_model: Tuple containing the model path and the model tag. If not None, a new model is created.
    :return:
    """
    if train_model == test_model == False:
        raise ValueError("At least one of train_model or test_model must be True")
    if test_model and saved_model is None:
        raise ValueError("If testing a model, a saved model must be provided")

    date = date_str()
    if use_wandb:
        init_wandb_run(project=PROJECT, tags=RUN_TAGS, config=RUN_CONFIG)
        config = wandb.config
    else:
        config = RUN_CONFIG

    test_dataset, train_dataset, val_dataset, vocab = get_ds(config, create_ds, date, save_ds, use_wandb)

    pad_idx = vocab.to_idx(PAD)
    model = get_model(config, vocab, pad_idx)

    save_dir = RESULTS_DIR + config["model"]
    if saved_model is not None:
        handle_saved_model(config, model, save_dir, saved_model, test_dataset, test_model, use_wandb)
        return

    if train_model:
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if use_wandb:
            wandb.run.summary["trainable_parameters"] = parameter_count
        logger.info(f"Number of trainable parameters: {parameter_count}")

        # dataloaders
        train_dataloader = CaptionLoader(train_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
        val_dataloader = CaptionLoader(val_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)

        best_val_model, best_val_info, _ = model.train_model(train_dataloader, val_dataloader, DEVICE, criterion, optimizer, scheduler,
                                                             CHECKPOINT_DIR + config["model"], use_wandb, config)

        if test_model:
            # test last model
            test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            model.test_model(test_dataloader, DEVICE, save_dir, "last-model", use_wandb, config)
            if use_wandb:
                wandb.finish()
            # test best model
            if best_val_model is not None:
                if use_wandb:
                    init_wandb_run(project=PROJECT, tags=RUN_TAGS, config=RUN_CONFIG)
                    config = wandb.config
                else:
                    config = RUN_CONFIG
                best = get_model(config, vocab, pad_idx)
                best.load_state_dict(torch.load(best_val_model, weights_only=True))
                best.test_model(test_dataloader, DEVICE, save_dir, "best-model", use_wandb, config)
                if use_wandb:
                    wandb.log(best_val_info)
                    wandb.log_model(path=best_val_model)

    if use_wandb:
        wandb.finish()


def init_wandb_run(project: str, tags: list, config: dict) -> Run:
    """
    Initialize wandb run
    :param project: Project name
    :param tags: List of tags
    :param config: Run configuration dictionary
    :return: Wandb run
    """
    wandb_run = wandb.init(project=project, tags=tags, config=config)
    wandb_run.define_metric("train_loss", summary="min")
    wandb_run.define_metric("val_loss", summary="min")
    wandb_run.define_metric("val_BLEU-4", summary="max")
    wandb_run.define_metric("test_BLEU-1", summary="max")
    wandb_run.define_metric("test_BLEU-2", summary="max")
    wandb_run.define_metric("test_BLEU-4", summary="max")
    wandb_run.define_metric("test_CIDEr", summary="max")
    wandb_run.define_metric("epoch", summary="max")
    return wandb_run


def get_ds(config, create_ds, date, save_ds, use_wandb):
    # create or load dataset
    img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
    if create_ds:
        df_captions = load_flickr_captions(str(os.path.join(ROOT, FLICKR8K_ANN_FILE)), True)
        total_size = len(df_captions["image_id"].unique())
        train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
        val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
        test_size = total_size - train_size - val_size
        train_df, val_df, test_df = split_dataframe(df_captions, [train_size, val_size, test_size])
    else:
        train_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_TRAIN_CSV)))
        val_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_VAL_CSV)))
        test_df = pd.read_csv(str(os.path.join(ROOT, FLICKR_TEST_CSV)))

    vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])
    train_dataset = CaptionDataset(img_dir, train_df, vocab, transform=TRANSFORM)
    val_dataset = CaptionDataset(img_dir, val_df, vocab, transform=TRANSFORM)
    test_dataset = CaptionDataset(img_dir, test_df, vocab, transform=TRANSFORM)

    if save_ds:
        save_df(config, date, test_df, train_df, val_df)
        save_datasets(None, train_dataset, val_dataset, test_dataset, date, config)  # save new datasets
        if use_wandb:
            log_datasets(date, False)
    return test_dataset, train_dataset, val_dataset, vocab


def get_model(config, vocab, pad_idx):
    match config["model"]:
        case "basic":
            encoder = basic.Encoder(config["embed_size"], not config["freeze_encoder"])
            decoder = basic.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
            return basic.BasicImageCaptioner(encoder, decoder)
        case "intermediate":
            encoder = intermediate.Encoder(config["embed_size"], config["encoder_dropout"], not config["freeze_encoder"])
            decoder = intermediate.Decoder(config["embed_size"], config["hidden_size"], len(vocab), config["dropout"], config["num_layers"], pad_idx)
            return intermediate.IntermediateImageCaptioner(encoder, decoder)
        case "transformer":
            return transformer.ImageCaptioningTransformer(vocab, config["hidden_size"], config["num_layers"], config["num_heads"],
                                                          config["max_caption_len"], config["encoder_dropout"], config["dropout"],
                                                          not config["freeze_encoder"])
        case _:
            raise ValueError(f"Model {config['model']} not recognized")


def handle_saved_model(config, model, save_dir, saved_model, test_dataset, test_model, use_wandb):
    # Load model from saved model
    logger.info(f"Loading model from {saved_model[0]}")
    model.load_state_dict(torch.load(str(os.path.join(ROOT, saved_model[0])), weights_only=True))
    if test_model:
        test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
        model.test_model(test_dataloader, DEVICE, save_dir, saved_model[1], use_wandb, config)
    if use_wandb:
        wandb.finish()


def get_optimizer(config, model):
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


def get_scheduler(config, optimizer):
    scheduler = None
    if config["scheduler"] is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"])
    return scheduler


def save_df(config: dict, date: str, test_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Save dataframes to disk
    :param config: Run configuration
    :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
    :param test_df: Test dataframe to be saved
    :param train_df: Training dataframe to be saved
    :param val_df: Validation dataframe to be saved
    :return:
    """
    train_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"train_{date}_{config['dataset']['split']['train']}.csv")), header=["image_id", "caption"],
                    index=False)
    val_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"val_{date}_{config['dataset']['split']['val']}.csv")), header=["image_id", "caption"],
                  index=False)
    test_df.to_csv(str(os.path.join(ROOT, FLICKR8K_DIR, f"test_{date}_{config['dataset']['split']['test']}.csv")), header=["image_id", "caption"],
                   index=False)


def save_datasets(full_dataset: Optional[CaptionDataset], train_dataset: CaptionDataset, val_dataset: CaptionDataset, test_dataset: CaptionDataset,
                  date: str, config: dict):
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
        torch.save(full_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt"))
    torch.save(train_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/train_{date}_{config["dataset"]["split"]["train"]}.pt"))
    torch.save(val_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/val_{date}_{config["dataset"]["split"]["val"]}.pt"))
    torch.save(test_dataset, os.path.join(ROOT, f"{FLICKR8K_DIR}/test_{date}_{config["dataset"]["split"]["test"]}.pt"))


def log_datasets(date: str, has_full_ds: bool):
    """
    Log datasets to wandb
    :param has_full_ds:
    :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
    :return:
    """
    config = wandb.config
    if has_full_ds:
        log_dataset(
            wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
            os.path.join(ROOT, f"{FLICKR8K_DIR}/full_dataset_{date}.pt")
        )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{FLICKR8K_DIR}/train_{date}_{config["dataset"]["split"]["train"]}.pt")
    )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{FLICKR8K_DIR}/val_{date}_{config["dataset"]["split"]["val"]}.pt")
    )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{FLICKR8K_DIR}/test_{date}_{config["dataset"]["split"]["test"]}.pt")
    )


def log_dataset(artifact: wandb.Artifact, dataset_path: str):
    """
    Log dataset to wandb
    :param artifact: Wandb artifact
    :param dataset_path: Path to the dataset
    """
    artifact.add_file(dataset_path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    wandb.teardown()
    saved_model_ = ("checkpoints/transformer/best_val_2025-02-27_03-13_3-6759.pt", "test")
    run(use_wandb=False, create_ds=False, save_ds=False, train_model=False, test_model=True, saved_model=saved_model_)
