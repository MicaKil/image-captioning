import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
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
        while save_name in os.listdir(os.path.join(ROOT, save_dir)):
            name = os.path.splitext(save_name)[0]
            save_name = f"{name}_{i:03d}.png"
            i += 1
        plt.savefig(os.path.join(ROOT, save_dir, save_name), bbox_inches='tight', dpi=150)
    plt.show()


def process_swin_attention(attentions: list, original_size=(256, 256)):
    """
    Process Swin's windowed attention into full image heatmaps

    :param attentions: List of attention weights from Swin blocks
    :param original_size: Original image size (H, W)
    :return: Aggregated attention maps (L, H, W)
    """
    maps = []
    for attn in attentions:
        # attn shape: [batch, num_heads, num_windows, window_size, window_size]
        batch, heads, windows, wh, ww = attn.shape
        window_size = int(wh ** 0.5)

        # Average across heads and windows
        attn = attn.mean(dim=[1, 2])  # [batch, wh*ww, wh*ww]

        # Reshape to spatial grid
        grid_size = int(windows ** 0.5)
        attn = attn.view(batch, grid_size, grid_size, window_size, window_size)
        attn = attn.permute(0, 1, 3, 2, 4).contiguous()
        attn = attn.view(batch,
                         grid_size * window_size,
                         grid_size * window_size)

        # Upsample to original size
        attn = f.interpolate(attn.unsqueeze(1),
                             size=original_size,
                             mode='bilinear').squeeze()
        maps.append(attn)

    # Average across layers
    return torch.stack(maps).mean(dim=0)


def preprocess_image(img_path: str, transform: v2.Compose) -> torch.Tensor:
    """
    Preprocess an image for the model.

    :param img_path: Path to the image file
    :param transform: Transform to apply to the image
    :return: Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension


def max_seq_len(train_df, val_df, test_df, vocab):
    max_len = max(train_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                  val_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max(),
                  test_df["caption"].apply(lambda x: len(vocab.tokenize(x))).max())
    return max_len + 2  # add 2 for start and end tokens


def load_checkpoint(checkpoint_pth: str):
    """
    Load the model from a checkpoint.

    :param checkpoint_pth: Path to the checkpoint file
    :return: Loaded model
    """
    checkpoint = torch.load(checkpoint_pth)
    config = checkpoint["config"]

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

    model.load_state_dict(checkpoint['model_state'])
    return model, config, vocab


def plot_app(img_pth: str, checkpoint_pth: str, save_name="test.png", save_dir="plots/attention/"):
    model, config, vocab = load_checkpoint(checkpoint_pth)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions, _, attns = gen_caption(model, img, vocab, 20, device, None, 5, True, True)
    tokens = vocab.tokenize(captions[0])
    plot_attention(img[0], captions[0], tokens, attns[0][:-1], MEAN, STD, 5, save_name, save_dir)


if __name__ == "__main__":
    pass
