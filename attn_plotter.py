import os.path

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from torchvision.transforms import v2

import models.encoders.transformer as t_encoder
from caption import gen_caption
from constants import FLICKR8K_DIR, COCO_DIR, ROOT, FLICKR_TRAIN_CSV, FLICKR_TEST_CSV, FLICKR_VAL_CSV, COCO_TRAIN_PKL, COCO_TEST_PKL, \
    COCO_VAL_PKL
from dataset.vocabulary import Vocabulary
from models import transformer
from models.encoders import swin

COLOR_INFO = "bright_blue"
COLOR_SUCCESS = "bright_green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_HIGHLIGHT = "bright_white"


def plot_attention(img_tensor: torch.Tensor, caption: str, tokens: list[str], attns: list, mean: list[float], std: list[float], columns: int,
                   save_name: str = None, save_dir: str = None):
    """
    Plot attention maps over the image for each step in the caption generation process.

    :param img_tensor: Original image tensor (after normalization)
    :param caption: Generated caption
    :param tokens: List of tokens in the caption
    :param attns: List of attention maps
    :param mean: Mean values for normalization
    :param std: Standard deviation values for normalization
    :param columns: The number of columns to display the attention maps
    :param save_name: Path to save the plot (optional)
    :param save_dir:
    """
    assert len(attns) == len(tokens), "attentions length must match caption length"
    # Inverse normalize the image
    inverse_normalize = v2.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    image = inverse_normalize(img_tensor).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    num_steps = len(attns)

    # height = (num_steps // columns + 1) * 3
    # width = columns * 2
    plt.figure(figsize=(10, 6), dpi=100)

    for step in range(num_steps):
        ax = plt.subplot(num_steps // columns + 1, columns, step + 1)
        attn = attns[step].reshape(8, 8)
        attn = zoom(attn, (256 / 8, 256 / 8))  # 7x7 -> 256x256

        ax.imshow(image)
        ax.imshow(attn, cmap='Greys_r', alpha=0.65)
        ax.set_title(f"{tokens[step]}", fontsize=12, pad=6)
        ax.axis('off')
    plt.suptitle(caption, fontsize=16)
    plt.tight_layout()
    if save_name:
        i = 0
        name = os.path.splitext(save_name)[0]  # keep the name without extension
        while save_name in os.listdir(os.path.join(ROOT, save_dir)):
            save_name = f"{name}_{i:03d}.png"
            i += 1
        plt.savefig(os.path.join(ROOT, save_dir, save_name), bbox_inches='tight', dpi=150)
    plt.show()
    return save_name


def preprocess_image(img_path: str, transform: v2.Compose) -> torch.Tensor:
    """
    Preprocess an image for the model.

    :param img_path: Path to the image file
    :param transform: Transform to apply to the image
    :return: Preprocessed image tensor of shape (1, 3, H, W)
    """
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension


def max_seq_len(train_df, val_df, test_df, vocab):
    """
    Calculate the maximum sequence length for the dataset.
    :param train_df: Dataframe containing training data.
    :param val_df: Dataframe containing validation data.
    :param test_df: Dataframe containing test data.
    :param vocab: Vocabulary of the dataset.
    :return: Integer representing the maximum sequence length.
    """
    max_len = max(train_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                  val_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                  test_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max())
    return max_len + 2  # add 2 for start and end tokens


def load_checkpoint(checkpoint_pth: str):
    """
    Load the model, configuration, and vocabulary from a checkpoint.

    :param checkpoint_pth: Path to the checkpoint file.
    :return: Tuple containing the model, configuration, and vocabulary.
    """
    checkpoint = torch.load(checkpoint_pth)
    config = checkpoint["config"]
    model, vocab = get_model_and_vocab(config)
    model.load_state_dict(checkpoint['model_state'])
    return model, config, vocab


def plot_pic_and_caption(img_pth, model_pth, config):
    if config:
        model, vocab = get_model_and_vocab(config)
        model.load_state_dict(torch.load(model_pth, weights_only=True))
    else:
        model, config, vocab = load_checkpoint(model_pth)

    try:
        transform_resize = config["transform_resize"]
    except KeyError:
        transform_resize = (256, 256)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize(transform_resize),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])

    img = preprocess_image(img_pth, TRANSFORM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions, _ = gen_caption(model, img, vocab, 20, device, None, 5, True, False)
    print(captions)
    # plot pic and add caption
    plt.figure(figsize=(8, 8))
    plt.imshow(Image.open(img_pth))
    plt.axis("off")
    plt.title(captions[0], fontsize=14, wrap=True)
    plt.show()


def get_model_and_vocab(config):
    ds_name = config["dataset"]["name"]
    match ds_name:
        case "flickr8k":
            train_df = pd.read_csv(os.path.join(ROOT, FLICKR_TRAIN_CSV))
            test_df = pd.read_csv(os.path.join(ROOT, FLICKR_TEST_CSV))
            val_df = pd.read_csv(os.path.join(ROOT, FLICKR_VAL_CSV))
            ds_dir = FLICKR8K_DIR
        case "coco":
            train_df = pd.read_pickle(os.path.join(ROOT, COCO_TRAIN_PKL))
            test_df = pd.read_pickle(os.path.join(ROOT, COCO_TEST_PKL))
            val_df = pd.read_pickle(os.path.join(ROOT, COCO_VAL_PKL))
            ds_dir = COCO_DIR
        case _:
            raise ValueError(f"Dataset '{ds_name}' not recognized")
    tokenizer = config["vocab"]["tokenizer"]
    freq_threshold = config["vocab"]["freq_threshold"]
    match tokenizer:
        case "word":
            vocab_file = f"vocab_freq-{freq_threshold}.pt"
            if vocab_file in os.listdir(os.path.join(ROOT, ds_dir)):  # check if vocab file exists
                vocab = Vocabulary(tokenizer, freq_threshold, sp_model_path=None)
                vocab.load_dict(torch.load(os.path.join(ROOT, ds_dir, vocab_file)))
            else:
                vocab = Vocabulary(tokenizer, freq_threshold, train_df["caption"], None)
        case "sp-bpe":
            sp_model = os.path.join(ROOT, ds_dir, f"{ds_name}_{config["vocab"]["vocab_size"]}.model")
            vocab = Vocabulary(tokenizer, None, text=None, sp_model_path=sp_model)
        case _:
            raise ValueError(f"Tokenizer '{tokenizer}' not recognized")
    fine_tune = config["fine_tune_encoder"]
    hidden_size = config["hidden_size"]
    decoder_dropout = config["dropout"]
    encoder_dropout = config["encoder_dropout"]
    num_layers = config["num_layers"]
    match config["encoder"]:
        case "resnet50":
            if config["model"] == "transformer":
                encoder = t_encoder.Encoder(hidden_size, encoder_dropout, fine_tune)
                model = transformer.ImageCaptioningTransformer(encoder, vocab, hidden_size, num_layers, config["num_heads"],
                                                               max_seq_len(train_df, test_df, val_df, vocab), decoder_dropout)
            else:
                raise ValueError(f"Model '{config['model']}' not recognized")
        case "swin":
            encoder = swin.SwinEncoder(hidden_size, encoder_dropout, fine_tune)
            if config["model"] == "transformer":
                model = transformer.ImageCaptioningTransformer(encoder, vocab, hidden_size, num_layers, config["num_heads"],
                                                               max_seq_len(train_df, test_df, val_df, vocab), decoder_dropout)
            else:
                raise ValueError(f"Model '{config['model']}' not recognized")
        case _:
            raise ValueError(f"Encoder '{config['encoder']}' not recognized")
    return model, vocab


def print_banner():
    """Print a stylized application banner"""
    click.secho("╭────────────────────────────────────────────╮", fg=COLOR_INFO)
    click.secho("│          Attention Plot Generator          │", fg=COLOR_INFO)
    click.secho("╰────────────────────────────────────────────╯", fg=COLOR_INFO)


def format_path(path: str) -> str:
    """Format a path for display in CLI output"""
    return click.style(f"'{click.format_filename(path)}'", fg=COLOR_HIGHLIGHT)


@click.command()
@click.argument("img_pth", type=click.Path(exists=True, dir_okay=False))
@click.argument("checkpoint_pth", type=click.Path(exists=True, dir_okay=False))
@click.option("--save-name", type=str, default="test.png", help="Name for the output plot file.")
@click.option("--save-dir", type=click.Path(file_okay=False), default=os.path.join(ROOT, "plots", "attention"),
              help="Directory to save the generated plot.", show_default=True)
def plot_attn_cli(img_pth: str, checkpoint_pth: str, save_name: str, save_dir: str):
    print_banner()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        click.secho(f"📁 Created output directory: {format_path(save_dir)}", fg=COLOR_INFO)
    click.secho("\n🚀 Starting image captioning pipeline:", fg=COLOR_INFO)

    try:
        click.secho("⏳ Loading model checkpoint...", fg=COLOR_INFO)
        model, config, vocab = load_checkpoint(checkpoint_pth)
        click.secho(f"✅ Successfully loaded {format_path(checkpoint_pth)}", fg=COLOR_SUCCESS)
    except Exception as e:
        click.secho(f"❌ Failed to load checkpoint: {str(e)}", fg=COLOR_ERROR)
        raise click.Abort()

    # Image processing
    try:
        click.secho("\n⏳ Processing image...", fg=COLOR_INFO)
        transform_resize = config["transform_resize"]
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        TRANSFORM = v2.Compose([
            v2.ToImage(),
            v2.Resize(transform_resize),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ])

        img = preprocess_image(img_pth, TRANSFORM)
        click.secho(f"✅ Processed image: {format_path(img_pth)}", fg=COLOR_SUCCESS)
    except Exception as e:
        click.secho(f"❌ Image processing failed: {str(e)}", fg=COLOR_ERROR)
        raise click.Abort()

    # Generate caption and attention
    try:
        click.secho("\n⏳ Generating caption and attention maps...", fg=COLOR_INFO)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        captions, _, attns = gen_caption(model, img, vocab, 20, device, None, 5, True, True)
        tokens = vocab.tokenize(captions[0])
        click.secho("✅ Caption generated successfully!", fg=COLOR_SUCCESS)
        click.secho(f"📝 Caption: {click.style(captions[0], fg=COLOR_HIGHLIGHT)}", fg=COLOR_INFO)
    except Exception as e:
        click.secho(f"❌ Caption generation failed: {str(e)}", fg=COLOR_ERROR)
        raise click.Abort()

    # Save and display results
    try:
        click.secho("\n⏳ Generating visualization...", fg=COLOR_INFO)
        save_name = plot_attention(img[0], captions[0], tokens, attns[0][:-1], MEAN, STD, 5, save_name, save_dir)
        final_path = os.path.join(save_dir, save_name)
        click.secho(f"💾 Saved attention plot to {format_path(final_path)}", fg=COLOR_SUCCESS)
    except Exception as e:
        click.secho(f"❌ Visualization failed: {str(e)}", fg=COLOR_ERROR)
        raise click.Abort()

    click.secho("\n✨ Process completed successfully! ✨\n", fg=COLOR_SUCCESS, bold=True)


if __name__ == "__main__":
    c = {
        "model": "transformer",
        "encoder": "resnet50",
        "decoder": "Attention",
        "criterion": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "batch_size": 64,
        "embed_size": 512,
        "hidden_size": 512,
        "num_layers": 2,
        "num_heads": 2,
        "encoder_dropout": 0.1,
        "dropout": 0.5,
        "fine_tune_encoder": "partial",
        "vocab": {
            "freq_threshold": 3,
            "tokenizer": "word",
            "vocab_size": 3500
        },
        "dataset": {
            "name": "flickr8k",
            "version": "2025-02-16",
            "split": {
                "train": 80,
                "val": 10,
                "test": 10
            }
        },
        "encoder_lr": 1e-5,
        "decoder_lr": 1e-4,
        "gradient_clip": 2.0,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 10
        },
        "max_caption_len": 40,
        "beam_size": 5,
        "temperature": 0.0,
    }

    # m_pth = "report/models/atomic-voice-25_best_val_2025-02-26_04-56_2-6236.pt"
    m_pth = "report/models/LAST_2025-03-26_04-05_2-1619.pt"
    i_pth = "data/mine/005_cropped.jpg"
    # i_pth = "data/flickr8k/images/1288909046_d2b2b62607.jpg"
    # 2857558098_98e9249284.jpg
    plot_pic_and_caption(i_pth, m_pth, None)
