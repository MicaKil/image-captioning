import os.path

import pandas as pd
import torch
from einops import rearrange
from torchvision.transforms import v2

from caption import preprocess_image, plot_attention
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

if __name__ == "__main__":
    checkpoint_pth = os.path.join(ROOT, "checkpoints/transformer/BEST_2025-03-20_12-40_2-1685.pt")
    train_df = pd.read_pickle(os.path.join(ROOT, COCO_TRAIN_PKL))
    vocab = Vocabulary("word", 5, None, None)
    vocab.load_dict(torch.load(os.path.join(ROOT, f"{COCO_DIR}/vocab_freq-{vocab.freq_threshold}.pt")))
    encoder = Encoder(512, 0.2, "partial")
    model = ImageCaptioningTransformer(encoder, vocab, 512, 2, 4, 59, 0.5)
    checkpoint = torch.load(os.path.join(ROOT, checkpoint_pth))
    model.load_state_dict(checkpoint['model_state'])

    img = preprocess_image(str(os.path.join(ROOT, f"data/coco/images/COCO_val2014_000000119120.jpg")), TRANSFORM) # (B, C, H, W)
    model = model.to(DEVICE)
    model.eval()
    img = img.to(DEVICE)
    features = model.encoder(img)
    features = rearrange(features, 'b c h w -> b (h w) c')
    captions, _, attns = model.beam_search(features, vocab, max_len=20, beam_size=5)

    tokens = vocab.tokenize(captions[0])
    plot_attention(img[0], captions[0], tokens, attns[0][:-1], MEAN, STD, 5, "test.png", "plots/attention/")
