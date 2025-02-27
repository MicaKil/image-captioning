import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.config import logger
from constants import ROOT, FLICKR8K_CSV_FILE


def load_flickr_captions(ann_path: str, save_captions=False) -> pd.DataFrame:
    """
    Load the captions from the annotation file.
    :param ann_path: Path to the annotation file
    :param save_captions: If True, save to a CVS file or overwrite the existing CSV file
    :return: DataFrame containing the image filenames and corresponding captions
    """

    if os.path.splitext(ann_path)[1] == ".csv":
        logger.info("Loading captions from CSV file.")
        df = pd.read_csv(ann_path)
    else:
        logger.info("Loading captions from annotation file.")
        df = pd.DataFrame(extract_flickr_captions(ann_path))  # Convert to DataFrame
    if save_captions:
        logger.info("Saving captions to CSV file.")
        df.to_csv(str(os.path.join(ROOT, FLICKR8K_CSV_FILE)), header=True, index=False)
    return df


def extract_flickr_captions(ann_path: str) -> list[dict[str, str]]:
    """
    Extract the captions from the annotation file.
    Sample line:
        "1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entryway."
    :param ann_path: Path to the annotation file
    :return: List of dictionaries containing the image ID and caption
    """
    captions = []
    with open(ann_path, "r") as f:
        for line in f:
            image_id, caption = line.strip().split("\t")
            image_id = image_id.split("#")[0]
            captions.append({"image_id": image_id, "caption": caption})
    return captions


def process_coco(ann: str, csv_path: str, json_path: str, pkl_path: str, root: str) -> pd.DataFrame:
    """
    Processes COCO dataset annotations and exports them to CSV, JSON, and Pickle formats.

    :param ann: Annotation file name (JSON).
    :param csv_path: Output CSV file path.
    :param json_path: Output JSON file path.
    :param pkl_path: Output Pickle file path.
    :param root: Root directory where files are stored.

    :returns: pd.DataFrame: Processed dataset.
    """
    ann_file = os.path.join(root, ann)

    # Load JSON data
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded COCO data with keys: {list(data.keys())}")

    # Create a dictionary mapping image_id to captions
    annotations_dict = {}
    for annotation in tqdm(data["annotations"], desc="Creating Dictionary"):
        image_id = annotation["image_id"]
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append({"caption_id": annotation["id"], "caption": annotation["caption"]})

    logger.info(f"Created dictionary with {len(annotations_dict)} image IDs.")

    # Process images and merge annotations
    processed = []
    for image in tqdm(data["images"], desc="Processing Images"):
        image_id = image["id"]
        if image_id in annotations_dict:
            for ann in annotations_dict[image_id]:
                processed.append({
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "caption_id": ann["caption_id"],
                    "caption": ann["caption"],
                })

    # Convert to DataFrame and sort
    df = pd.DataFrame(processed).sort_values(by="image_id").reset_index(drop=True)

    logger.info(f"Processed {len(df)} image-caption pairs.")

    # Save DataFrame in multiple formats
    df.to_csv(os.path.join(root, csv_path), header=True, index=False, encoding="utf-8")
    df.to_json(os.path.join(root, json_path), orient="records", force_ascii=False)
    df.to_pickle(os.path.join(root, pkl_path))

    logger.info("Saved processed data to CSV, JSON, and Pickle files.")

    return df


def split_dataframe(df: pd.DataFrame, split_lengths: list[int]) -> list[pd.DataFrame]:
    """
    Split a DataFrame containing image IDs and captions into multiple DataFrames based on the specified lengths.
    :param df: DataFrame containing the image IDs and captions. Headers must be "image_id" and "caption".
    :param split_lengths: List of integers specifying the lengths of the splits. Must sum to the number of unique images.
    :return: List of DataFrames containing the splits
    """
    # Extract all unique image IDs from the dataframe
    unique_images = df['image_id'].unique()
    n_total = len(unique_images)

    # Verify that the sum of split lengths equals the number of unique images
    if sum(split_lengths) != n_total:
        raise ValueError(f"Sum of split lengths ({sum(split_lengths)}) must equal the number of unique images ({n_total}).")

    # Shuffle the unique image IDs to ensure randomness
    shuffled_images = np.random.permutation(unique_images)

    # Calculate the indices where the splits occur
    split_indices = np.cumsum(split_lengths[:-1])

    # Split the shuffled image IDs into groups according to split_lengths
    image_splits = np.split(shuffled_images, split_indices)

    # Create dataframe splits based on the image IDs in each split
    df_splits = []
    for images in image_splits:
        mask = df['image_id'].isin(images)
        df_split = df[mask].reset_index(drop=True)
        df_splits.append(df_split)

    return df_splits


if __name__ == "__main__":
    process_coco("data/coco/annotations/captions_val2014.json", "data/coco/coco_train.csv", "data/coco/coco_train.json", "data/coco/coco_train.pkl", ROOT)
