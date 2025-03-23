from os.path import join

import pandas as pd

from constants import COCO_TEST_PKL, COCO_VAL_PKL, COCO_TRAIN_PKL, ROOT

ds_splits = (COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL)

train_df = pd.read_pickle(join(ROOT, ds_splits[0]))
print(train_df.shape)
val_df = pd.read_pickle(join(ROOT, ds_splits[1]))
print(val_df.shape)
test_df = pd.read_pickle(join(ROOT, ds_splits[2]))
print(test_df.shape)
combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(len(combined))

unique_images = combined['image_id'].unique()
n_total = len(unique_images)
print(n_total)
