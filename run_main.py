import wandb

from constants import COCO_IMGS_DIR, COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL, FLICKR8K_DIR, FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV, \
    FLICKR8K_IMG_DIR, COCO_DIR
from scripts.runner.config import PROJECT, RUN_TAGS, RUN_CONFIG
from scripts.runner.runner import Runner

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

saved_model_ = "checkpoints/transformer/LAST_2025-03-02_14-22_2-2577.pt"

if __name__ == "__main__":
    run = Runner(use_wandb=True,
                 create_ds=False,
                 save_ds=False,
                 train_model=True,
                 test_model=True,
                 checkpoint_pth=None,
                 img_dir=img_dir_,
                 ds_splits=ds_splits_,
                 ds_dir=ds_dir_,
                 project=PROJECT,
                 run_tags=RUN_TAGS,
                 run_config=RUN_CONFIG)

    run.run()
