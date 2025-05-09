import torch
import wandb
from torch.nn import CrossEntropyLoss

from config.config import logger
from constants import PAD, CHECKPOINT_DIR, RESULTS_DIR
from dataset.dataloader import CaptionLoader
from runner.runner import Runner, get_optimizer, get_scheduler


class Sweeper(Runner):
    """
    Sweeper class that extends the Runner class to perform model training and testing
    using wandb for experiment tracking.
    """

    def __init__(self, img_dir: str, ds_splits: tuple[str, str, str], ds_dir: str, project: str, run_tags: list[str], run_config: dict):
        """
        Initialize the Sweeper with configuration parameters.

        :param img_dir: Directory path for dataset images.
        :param ds_splits: Tuple with paths for train, validation, and test dataframes.
        :param ds_dir: Directory for dataset files.
        :param project: Wandb project name.
        :param run_tags: List of tags for the run.
        :param run_config: Configuration dictionary for the run parameters.
        """
        super().__init__(True, False, False, True, True, None, img_dir, ds_splits, ds_dir, project, run_tags, run_config)

    def __call__(self, *args, **kwargs):
        """
        Callable method to execute the run method.
        """
        self.run()

    def run(self):
        """
        Execute the training and testing pipeline.

        Sets up the wandb run, datasets, model, optimizer, scheduler, and triggers
        the training and testing processes.
        """
        num_workers = 4
        shuffle = True
        pin_memory = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize wandb run
        self.init_wandb_run()
        config = wandb.config

        # Load datasets
        train_dataset, val_dataset, test_dataset, vocab = self.get_datasets(config, None)
        pad_idx = vocab.str_to_idx(PAD)

        # Create data loaders
        train_dataloader = CaptionLoader(train_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
        val_dataloader = CaptionLoader(val_dataset, config["batch_size"], num_workers, shuffle, pin_memory)
        test_dataloader = CaptionLoader(test_dataset, config["batch_size"], num_workers, shuffle, pin_memory)

        # Initialize model, loss function, optimizer, and scheduler
        model = self.get_model(config, vocab, pad_idx)
        criterion = CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer, config["encoder_lr"])

        # Log the number of trainable parameters
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary["trainable_parameters"] = parameter_count
        logger.info(f"Number of trainable parameters: {parameter_count}")

        model.train_model(train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, CHECKPOINT_DIR + config["model"], True, config,
                          None)
        model.test_model(test_dataloader, device, RESULTS_DIR + config["model"], "LAST", True, config)
