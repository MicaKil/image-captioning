import wandb

from constants import COCO_IMGS_DIR, COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL, FLICKR8K_DIR, FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV, \
    FLICKR8K_IMG_DIR, COCO_DIR
from scripts.runner.config import PROJECT, TAGS, CONFIG
from scripts.runner.runner import Runner

wandb.teardown()

match CONFIG["dataset"]["name"]:
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

checkpoint_ = "checkpoints/transformer/LAST_2025-03-13_15-01_2-3249.pt"

if __name__ == "__main__":
    run = Runner(use_wandb=False,
                 create_ds=False,
                 save_ds=False,
                 train_model=True,
                 test_model=True,
                 checkpoint_pth=checkpoint_,
                 img_dir=img_dir_,
                 ds_splits=ds_splits_,
                 ds_dir=ds_dir_,
                 project=PROJECT,
                 run_tags=TAGS,
                 run_config=CONFIG)

    run.run()
