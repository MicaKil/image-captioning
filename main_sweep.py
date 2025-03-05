import wandb

from constants import FLICKR_VAL_CSV, FLICKR_TRAIN_CSV, FLICKR_TEST_CSV, FLICKR8K_IMG_DIR, FLICKR8K_DIR
from scripts.sweeper.config import DEFAULT_CONFIG, SWEEP_CONFIG, PROJECT, TAGS
from scripts.sweeper.sweeper import Sweeper

sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT)
print(f"Sweep id: {sweep_id}")
sweeper = Sweeper(FLICKR8K_IMG_DIR, (FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV), FLICKR8K_DIR, PROJECT, TAGS, DEFAULT_CONFIG)
wandb.agent(sweep_id=sweep_id, function=sweeper.run(), count=20)
