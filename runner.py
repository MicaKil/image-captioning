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
from constants import ROOT, CHECKPOINT_DIR, PAD, RESULTS_DIR
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.dataset import CaptionDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.models import basic, intermediate, transformer
from scripts.utils import date_str


def run(use_wandb: bool, create_ds: bool, save_ds: bool, train_model: bool, test_model: bool, checkpoint: Optional[str], img_dir,
        ds_splits, ds_dir):
    """
    Run the training and testing pipeline
    :param ds_dir:
    :param img_dir:
    :param ds_splits:
    :param use_wandb: Whether to use wandb
    :param create_ds: Whether to create a new dataset or load an existing one. Saves the dataset to disk if a new one is created
    :param save_ds: Whether to save the datasets to disk
    :param train_model: Whether to train the model
    :param test_model: Whether to test the model
    :param checkpoint: Tuple containing the model path and the model tag. If not None, a new model is created.
    :return:
    """
    if train_model == test_model == False:
        raise ValueError("At least one of train_model or test_model must be True.")
    if test_model and not train_model and checkpoint is None:
        raise ValueError("If only testing a model, a saved model (checkpoint) must be provided.")

    date = date_str()
    if use_wandb:
        init_wandb_run(project=PROJECT, tags=RUN_TAGS, config=RUN_CONFIG)
        config = wandb.config
    else:
        config = RUN_CONFIG

    train_dataset, val_dataset, test_dataset, vocab = get_ds(config, create_ds, date, save_ds, use_wandb, img_dir, ds_splits, ds_dir)
    pad_idx = vocab.to_idx(PAD)
    model = get_model(config, vocab, pad_idx, ds_splits)
    save_dir = RESULTS_DIR + config["model"]

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

        best_model, best_state, _ = model.train_model(train_dataloader, val_dataloader, DEVICE, criterion, optimizer, scheduler,
                                                      CHECKPOINT_DIR + config["model"], use_wandb, config, checkpoint)

        if test_model:
            # test last model
            test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
            model.test_model(test_dataloader, DEVICE, save_dir, "last-model", use_wandb, config)
            if use_wandb:
                wandb.finish()

            # test best model
            if best_model is not None:
                if use_wandb:
                    init_wandb_run(project=PROJECT, tags=RUN_TAGS, config=RUN_CONFIG)
                    config = wandb.config
                else:
                    config = RUN_CONFIG
                best = get_model(config, vocab, pad_idx, ds_splits)
                best.load_state_dict(torch.load(best_model, weights_only=True))
                best.test_model(test_dataloader, DEVICE, save_dir, "best-model", use_wandb, config)
                if use_wandb:
                    wandb.log(best_state)
                    wandb.log_model(path=best_model)
        wandb.finish()
        return

    if test_model and checkpoint is not None:
        logger.info(f"Testing model from checkpoint.")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        test_dataloader = CaptionLoader(test_dataset, config["batch_size"], NUM_WORKERS, SHUFFLE, PIN_MEMORY)
        model.test_model(test_dataloader, DEVICE, save_dir, "checkpoint", use_wandb, config)

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


def get_ds(config: dict, create_ds: bool, date: str, save_ds: bool, use_wandb: bool, img_dir: str, ds_splits: tuple[str, str, str], ds_dir: str):
    # create or load dataset
    img_dir = str(os.path.join(ROOT, img_dir))
    if create_ds:
        raise NotImplementedError("Creating new datasets is being refactored.")
        # df_captions = load_flickr_captions(str(os.path.join(ROOT, FLICKR8K_ANN_FILE)), True)
        # total_size = len(df_captions["image_id"].unique())
        # train_size = int((config["dataset"]["split"]["train"] / 100) * total_size)
        # val_size = int((config["dataset"]["split"]["val"] / 100) * total_size)
        # test_size = total_size - train_size - val_size
        # train_df, val_df, test_df = split_dataframe(df_captions, [train_size, val_size, test_size])
        # if save_ds:
        #     save_df(config, date, test_df, train_df, val_df)  # save new dataframes to disk
    else:
        train_df, val_df, test_df = get_dataframes(ds_splits)

    vocab = Vocabulary(config["vocab"]["freq_threshold"], train_df["caption"])
    train_dataset = CaptionDataset(img_dir, train_df, vocab, transform=TRANSFORM)
    val_dataset = CaptionDataset(img_dir, val_df, vocab, transform=TRANSFORM)
    test_dataset = CaptionDataset(img_dir, test_df, vocab, transform=TRANSFORM)

    if save_ds:
        save_datasets(None, train_dataset, val_dataset, test_dataset, date, config, ds_dir)  # save datasets to disk
        if use_wandb:
            log_datasets(date, False, ds_dir)
    return train_dataset, val_dataset, test_dataset, vocab


def get_dataframes(ds_splits):
    match os.path.splitext(ds_splits[0])[1]:
        case ".csv":
            train_df = pd.read_csv(str(os.path.join(ROOT, ds_splits[0])))
            val_df = pd.read_csv(str(os.path.join(ROOT, ds_splits[1])))
            test_df = pd.read_csv(str(os.path.join(ROOT, ds_splits[2])))
        case ".pkl":
            train_df = pd.read_pickle(str(os.path.join(ROOT, ds_splits[0])))
            val_df = pd.read_pickle(str(os.path.join(ROOT, ds_splits[1])))
            test_df = pd.read_pickle(str(os.path.join(ROOT, ds_splits[2])))
        case ".json":
            train_df = pd.read_json(str(os.path.join(ROOT, ds_splits[0])))
            val_df = pd.read_json(str(os.path.join(ROOT, ds_splits[1])))
            test_df = pd.read_json(str(os.path.join(ROOT, ds_splits[2])))
        case _:
            raise ValueError("Dataframe file format not recognized")
    return train_df, val_df, test_df


def get_model(config, vocab, pad_idx, ds_splits):
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
                                                          calc_max_sequence_length(ds_splits) + 2, config["encoder_dropout"], config["dropout"],
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


def save_df(config: dict, date: str, test_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, ds_dir: str):
    """
    Save dataframes to disk
    :param ds_dir:
    :param config: Run configuration
    :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
    :param test_df: Test dataframe to be saved
    :param train_df: Training dataframe to be saved
    :param val_df: Validation dataframe to be saved
    :return:
    """
    train_df.to_csv(str(os.path.join(ROOT, ds_dir, f"train_{date}_{config['dataset']['split']['train']}.csv")), header=["image_id", "caption"],
                    index=False)
    val_df.to_csv(str(os.path.join(ROOT, ds_dir, f"val_{date}_{config['dataset']['split']['val']}.csv")), header=["image_id", "caption"],
                  index=False)
    test_df.to_csv(str(os.path.join(ROOT, ds_dir, f"test_{date}_{config['dataset']['split']['test']}.csv")), header=["image_id", "caption"],
                   index=False)


def save_datasets(full_dataset: Optional[CaptionDataset], train_dataset: CaptionDataset, val_dataset: CaptionDataset, test_dataset: CaptionDataset,
                  date: str, config: dict, ds_dir: str):
    """
    Save datasets to disk
    :param ds_dir:
    :param full_dataset: Complete dataset
    :param test_dataset: Test dataset
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
    :param config: Run configuration
    :return:
    """
    if full_dataset is not None:
        torch.save(full_dataset, os.path.join(ROOT, f"{ds_dir}/full_dataset_{date}.pt"))
    torch.save(train_dataset, os.path.join(ROOT, f"{ds_dir}/train_{date}_{config["dataset"]["split"]["train"]}.pt"))
    torch.save(val_dataset, os.path.join(ROOT, f"{ds_dir}/val_{date}_{config["dataset"]["split"]["val"]}.pt"))
    torch.save(test_dataset, os.path.join(ROOT, f"{ds_dir}/test_{date}_{config["dataset"]["split"]["test"]}.pt"))


def log_datasets(date: str, has_full_ds: bool, ds_dir: str):
    """
    Log datasets to wandb
    :param ds_dir:
    :param has_full_ds:
    :param date: Date string in the format "YYYY-MM-DD" to be appended to the dataset file names
    :return:
    """
    config = wandb.config
    if has_full_ds:
        log_dataset(
            wandb.Artifact(f"{config["dataset"]["name"]}_full_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
            os.path.join(ROOT, f"{ds_dir}/full_dataset_{date}.pt")
        )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_train_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{ds_dir}/train_{date}_{config["dataset"]["split"]["train"]}.pt")
    )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_val_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{ds_dir}/val_{date}_{config["dataset"]["split"]["val"]}.pt")
    )
    log_dataset(
        wandb.Artifact(f"{config["dataset"]["name"]}_test_dataset", type="dataset", metadata={"version": config["dataset"]["version"]}),
        os.path.join(ROOT, f"{ds_dir}/test_{date}_{config["dataset"]["split"]["test"]}.pt")
    )


def log_dataset(artifact: wandb.Artifact, dataset_path: str):
    """
    Log dataset to wandb
    :param artifact: Wandb artifact
    :param dataset_path: Path to the dataset
    """
    artifact.add_file(dataset_path)
    wandb.log_artifact(artifact)


def calc_max_sequence_length(ds_splits: tuple[str, str, str]):
    """
    Calculate the maximum sequence length in the dataset
    :param ds_splits: Tuple containing the paths to the train, val, and test dataframes
    :return: Maximum sequence length
    """
    train_df, val_df, test_df = get_dataframes(ds_splits)
    max_len = max(train_df["caption"].apply(lambda x: len(x.split(" "))).max(),
                  val_df["caption"].apply(lambda x: len(x.split(" "))).max(),
                  test_df["caption"].apply(lambda x: len(x.split(" "))).max())
    return max_len


if __name__ == "__main__":
    from constants import COCO_IMGS_DIR, COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL, FLICKR8K_DIR, FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV, \
        FLICKR8K_IMG_DIR, COCO_DIR

    wandb.teardown()

    match RUN_CONFIG["dataset"]["name"]:
        case "flickr8k":
            ds_dir_ = FLICKR8K_DIR
            img_dir_ = FLICKR8K_IMG_DIR
            ds_splits_ = (FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV)
        case "coco":
            ds_dir_ = COCO_DIR
            img_dir_ = COCO_IMGS_DIR
            ds_splits_ = (COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL)
        case _:
            raise ValueError("Dataset not recognized")

    # print(calc_max_sequence_length(ds_splits_))
    saved_model_ = "checkpoints/transformer/best_val_2025-02-27_03-13_3-6759.pt"

    run(use_wandb=True,
        create_ds=False,
        save_ds=True,
        train_model=True,
        test_model=True,
        checkpoint=None,
        img_dir=img_dir_,
        ds_splits=ds_splits_,
        ds_dir=ds_dir_
        )
