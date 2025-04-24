from datetime import datetime

import wandb
from wandb.sdk import Config


def time_str() -> str:
    """
    Return the current time as a string in the format "YYYY-MM-DD_HH-MM".
    :return: Current time as a string
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def date_str() -> str:
    """
    Return the current date as a string in the format "YYYY-MM-DD".
    :return: Current date as a string
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_config(run_config: dict, use_wandb: bool) -> Config | dict:
    """
    Get the configuration for the run
    :param run_config: Default configuration
    :param use_wandb: Whether to use Weights & Biases
    :return: The configuration for the run
    """
    if use_wandb:
        config = wandb.config
    else:
        config = run_config
    return config
