import wandb

from constants import FLICKR_VAL_CSV, FLICKR_TRAIN_CSV, FLICKR_TEST_CSV, FLICKR8K_IMG_DIR, FLICKR8K_DIR
from sweeper import DEFAULT_CONFIG, SWEEP_CONFIG, PROJECT, TAGS
from sweeper.sweeper import Sweeper

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT)
    print(f"Sweep id: {sweep_id}")
    wandb.agent(
        sweep_id=sweep_id,
        function=Sweeper(FLICKR8K_IMG_DIR, (FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV), FLICKR8K_DIR, PROJECT, TAGS, DEFAULT_CONFIG),
        count=30
    )
