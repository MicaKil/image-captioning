import os.path

import torch
from einops import rearrange

from constants import ROOT, PATH_ALVARITO
from scripts.caption import preprocess_image, plot_attention
from scripts.dataset.vocabulary import Vocabulary
from scripts.models.transformer import ImageCaptioningTransformer
from scripts.runner.config import TRANSFORM, MEAN, STD

checkpoint_pth = os.path.join(ROOT, "checkpoints/transformer/BEST_2025-03-19_03-20_2-8243.pt")
sp_model = os.path.join(ROOT, "data/flickr8k/flickr8k.model")
vocab = Vocabulary("sp-bpe", None, None, sp_model)
model = ImageCaptioningTransformer(vocab, 512, 3, 4, 46, 0.2, 0.5, "full")
checkpoint = torch.load(os.path.join(ROOT, checkpoint_pth))
model.load_state_dict(checkpoint['model_state'])

img = preprocess_image(str(os.path.join(ROOT, PATH_ALVARITO)), TRANSFORM) # (B, C, H, W)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
model.eval()
img = img.to(DEVICE)
features = model.encoder(img)
features = rearrange(features, 'b c h w -> b (h w) c')
captions, _, attentions = model.temperature_sampling(features, vocab, max_length=60, temperature=0, collect_attn=True)

pieces = vocab.tokenize(captions[0])
plot_attention(img[0], pieces, attentions[:-1], MEAN, STD, None)
