import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import torch
from torchvision.transforms import v2
import torch.nn.functional as f


def plot_attention(img_tensor: torch.Tensor, caption: str, tokens: list[str], attns: list, mean: list[float], std: list[float], columns: int,
                   save_name: str = None, save_dir: str = None):
    """
    Plot attention maps over the image for each step in the caption generation process.

    :param save_dir:
    :param img_tensor: Original image tensor (after normalization)
    :param caption: Generated caption
    :param tokens: List of tokens in the caption
    :param attns: List of attention maps (steps x layers x 49)
    :param mean: Mean values for normalization
    :param std: Standard deviation values for normalization
    :param columns: The number of columns to display the attention maps
    :param save_name: Path to save the plot (optional)
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


if __name__ == "__main__":
    import os.path

    import pandas as pd
    from einops import rearrange

    from constants import ROOT, COCO_TRAIN_PKL, COCO_DIR
    from dataset.vocabulary import Vocabulary
    from models.transformer import ImageCaptioningTransformer
    from models.encoders.transformer import Encoder

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_pth = os.path.join(ROOT, "checkpoints/transformer/BEST_2025-03-20_12-40_2-1685.pt")
    train_df = pd.read_pickle(os.path.join(ROOT, COCO_TRAIN_PKL))
    vocab = Vocabulary("word", 5, None, None)
    vocab.load_dict(torch.load(os.path.join(ROOT, f"{COCO_DIR}/vocab_freq-{vocab.freq_threshold}.pt")))
    encoder = Encoder(512, 0.2, "partial")
    model = ImageCaptioningTransformer(encoder, vocab, 512, 2, 4, 59, 0.5)
    checkpoint = torch.load(os.path.join(ROOT, checkpoint_pth))
    model.load_state_dict(checkpoint['model_state'])

    img_ = preprocess_image(str(os.path.join(ROOT, f"data/coco/images/COCO_val2014_000000119120.jpg")), TRANSFORM)  # (B, C, H, W)
    model = model.to(DEVICE)
    model.eval()
    img_ = img_.to(DEVICE)
    features = model.encoder(img_)
    features = rearrange(features, 'b c h w -> b (h w) c')
    captions, _, attns_ = model.beam_search(features, vocab, max_len=20, beam_size=5)

    tokens_ = vocab.tokenize(captions[0])
    plot_attention(img_[0], captions[0], tokens_, attns_[0][:-1], MEAN, STD, 5, "test.png", "plots/attention/")
