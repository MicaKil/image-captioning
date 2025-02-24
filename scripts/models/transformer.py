from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import log_softmax, softmax

from constants import SOS, EOS, UNK, PAD
from scripts.dataset.vocabulary import Vocabulary


class SeqEmbedding(nn.Module):
    """
    Combines token embeddings with positional embeddings. In addition to a simple vector embedding for each token ID, the embedding layer also
    includes an embedding for each position in the sequence.
    """

    def __init__(self, vocab_size: int, max_length: int, depth: int, pad_idx: int):
        """
        :param vocab_size: Size of the vocabulary
        :param max_length: Maximum length of the sequence
        :param depth: Depth of the embedding
        :param pad_idx: Index of the padding token
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, depth, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_length, depth)

    def forward(self, seq):
        _, seq_len = seq.size()
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, depth)
        tok_emb = self.token_embedding(seq)  # (batch_size, seq_len, depth)
        return tok_emb + pos_emb  # (batch_size, seq_len, depth)


class CausalSelfAttention(nn.Module):
    """
    Implements masked self-attention for autoregressive generation.
    """

    def __init__(self, units, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(units)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x, is_causal=True)
        x = x + attn_output  # Residual connection
        return self.layer_norm(x)

    # TODO: Could be redundant. Possibly use is_causal=True in MultiheadAttention instead.
    @staticmethod
    def causal_mask(x):
        """
        Creates triangular mask to prevent looking at future tokens
        :param x:
        :return:
        """
        sz = x.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(x.device)


class CrossAttention(nn.Module):
    """
    Handles image-text attention, connecting the encoder and decoder.
    """

    def __init__(self, units, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(units)
        self.attention_scores = None

    def forward(self, x, y):
        attn_output, attn_weights = self.mha(x, y, y)  # Query: x (Decoder query), Key: y (Image features), Value: y (Image features)
        self.attention_scores = attn_weights  # Save attention scores for visualization
        x = x + attn_output
        return self.layer_norm(x)


class FeedForward(nn.Module):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(units, 2 * units),
            nn.ReLU(),
            nn.Linear(2 * units, units),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(units)

    def forward(self, x):
        x = x + self.seq(x)  # Residual connection
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.self_attention = CausalSelfAttention(units, num_heads)  # Handles image-text attention
        self.cross_attention = CrossAttention(units, num_heads)  # Aligns text with image features
        self.ff = FeedForward(units, dropout_rate)  # Further processes representation

    def forward(self, img_features, txt_emb):
        txt_emb = self.self_attention(txt_emb)
        txt_emb = self.cross_attention(txt_emb, img_features)
        txt_emb = self.ff(txt_emb)
        return txt_emb


class Output(nn.Module):
    def __init__(self, hidden_size, vocab_size, banned_indices):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.banned_indices = banned_indices
        # Initialize bias to zeros
        self.linear.bias.data.zero_()
        # Set banned tokens to -1e9 initially
        self.linear.bias.data[self.banned_indices] = -1e9

    def adapt(self, counts):
        """
        Adjusts the bias based on token frequencies in the dataset.
        :param counts: Counter containing token indices and their frequencies (excluding PAD)
        """
        total = sum(counts.values())
        if total == 0:
            return
        log_probs = torch.full_like(self.linear.bias.data, -1e9, dtype=torch.float32)
        for idx, count in counts.items():
            if count == 0:
                continue
            prob = count / total
            log_probs[idx] = torch.log(torch.tensor(prob))
        # Ensure banned tokens remain blocked
        log_probs[self.banned_indices] = -1e9
        self.linear.bias.data = log_probs

    def forward(self, x):
        return self.linear(x)


class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab, encoder, hidden_size=256, num_layers=2, num_heads=2, max_length=50, dropout=0.1, pad_idx=0):
        super().__init__()
        self.vocab = vocab
        vocab_size = len(vocab)

        self.encoder = encoder

        # Text embedding
        self.seq_embedding = SeqEmbedding(vocab_size, max_length, hidden_size, pad_idx)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output layer with smart initialization
        self.output_layer = Output(hidden_size, vocab_size, [vocab.to_idx(t) for t in [PAD, UNK, SOS]])

    def _init_output_bias(self):
        """
        The model will be generating text. It should never generate a pad, unknown, or start token.
        So set the bias for these to a large negative value.
        :return:
        """
        banned = [self.vocab.to_idx(t) for t in [PAD, UNK, SOS]]
        with torch.no_grad():
            self.output_layer.bias[banned] = -1e9

    def forward(self, images, captions):
        # Extract image features
        img_features = self.encoder(images)
        # Encode image to features: (batch, channels, height, width) → (batch, h*w, features)
        img_features = rearrange(img_features, 'b c h w -> b (h w) c')

        # Embed text
        txt_emb = self.seq_embedding(captions)

        # Process through decoder layers
        for layer in self.decoder_layers:
            txt_emb = layer(img_features, txt_emb)

        # Final output
        logits = self.output_layer(txt_emb)
        return logits

    def generate(self, image: torch.Tensor, vocab: Vocabulary, max_length: int = 30, device: torch.device = torch.device("cpu"),
                 temperature: Optional[float] = None, beam_size: int = 1) -> str:
        self.eval()
        with torch.no_grad():
            image = image.to(device)
            # Encode image
            img_features = self.encoder(image.unsqueeze(0))
            img_features = rearrange(img_features, 'b c h w -> b (h w) c')

            if beam_size > 1:
                return self.beam_search(img_features, vocab, max_length, beam_size)
            else:
                return self.temperature_sampling(img_features, vocab, max_length, temperature)

    def temperature_sampling(self, img_features, vocab, max_length, temperature):
        tokens = torch.tensor([[vocab.to_idx(SOS)]], device=img_features.device)

        for _ in range(max_length):
            txt_emb = self.seq_embedding(tokens)
            for layer in self.decoder_layers:
                txt_emb = layer(img_features, txt_emb)
            logits = self.output_layer(txt_emb[:, -1, :])

            if temperature is None or temperature == 0:
                next_token = logits.argmax(-1)
            else:
                probs = softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == vocab.to_idx(EOS):
                break

        # return tokens.squeeze().tolist()[1:-1]
        return vocab.to_text(tokens.squeeze().tolist())

    def beam_search(self, img_features, vocab, max_length, beam_size):
        # Initialize beam search
        sos_idx = vocab.to_idx(SOS)
        eos_idx = vocab.to_idx(EOS)
        # vocab_size = len(vocab)

        # Expand features for beam search
        img_features = img_features.repeat(beam_size, 1, 1)

        # Initialize beams (log_prob, sequence)
        beams = [(0.0, [sos_idx])]

        for _ in range(max_length):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_idx:
                    candidates.append((score, seq))
                    continue

                # Prepare input
                tokens = torch.tensor([seq], device=img_features.device)

                # Forward pass
                txt_emb = self.seq_embedding(tokens)
                for layer in self.decoder_layers:
                    txt_emb = layer(img_features[:len(tokens)], txt_emb)
                logits = self.output_layer(txt_emb[:, -1, :])

                # Get probabilities
                log_probs = log_softmax(logits, dim=-1)
                top_probs, top_indices = log_probs.topk(beam_size)

                # Add candidates
                for i in range(beam_size):
                    candidates.append((
                        score + top_probs[0, i].item(),
                        seq + [top_indices[0, i].item()]
                    ))

            # Keep top-k candidates
            candidates.sort(reverse=True, key=lambda x: x[0] / (len(x[1]) ** 0.5))  # Length normalization
            beams = candidates[:beam_size]

            # Check if all beams are complete
            if all(seq[-1] == eos_idx for _, seq in beams):
                break

        # Return best sequence
        best_seq = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5))[1]
        return vocab.to_text(best_seq)
