import os.path

import torch
from einops import rearrange
from torchvision.transforms import v2

from constants import ROOT, PATH_ALVARITO
from dataset.vocabulary import Vocabulary
from scripts.caption import preprocess_image, plot_attention, gen_caption
from scripts.models.transformer import ImageCaptioningTransformer, Encoder

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    pass
    checkpoint_pth = os.path.join(ROOT, "checkpoints/transformer/BEST_2025-03-19_03-20_2-8243.pt")
    sp_model = os.path.join(ROOT, "data/flickr8k/flickr8k.model")
    vocab = Vocabulary("sp-bpe", None, None, sp_model)
    encoder = Encoder(512, 0.2, "full")
    model = ImageCaptioningTransformer(encoder, vocab, 512, 3, 4, 46, 0.5)
    checkpoint = torch.load(os.path.join(ROOT, checkpoint_pth))
    model.load_state_dict(checkpoint['model_state'])

    img = preprocess_image(str(os.path.join(ROOT, PATH_ALVARITO)), TRANSFORM) # (B, C, H, W)
    model = model.to(DEVICE)
    model.eval()
    img = img.to(DEVICE)
    features = model.encoder(img)
    features = rearrange(features, 'b c h w -> b (h w) c')
    captions, _, attentions = model.temperature_sampling(features, vocab, max_length=60, temperature=0, collect_attn=True)

    pieces = vocab.tokenize(captions[0])
    print(captions[0])
    plot_attention(img[0], pieces, attentions[:-1], MEAN, STD, None)

    c, _ = gen_caption(model, img, vocab, max_length=60, device=DEVICE, temperature=0, beam_size=5, no_grad=True)
    print(c)