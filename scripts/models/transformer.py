import os.path
from collections import Counter
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import log_softmax, softmax

from configs.runner_config import TRANSFORM
from constants import SOS, EOS, UNK, PAD, TRAIN_CSV, ROOT, FLICKR8K_IMG_DIR
from scripts.dataset.flickr_dataset import FlickrDataset
from scripts.dataset.vocabulary import Vocabulary
from scripts.models import intermediate


class SeqEmbedding(nn.Module):
    """
    Combines token embeddings with positional embeddings to provide contextualized token representations.
    """

    def __init__(self, vocab_size: int, max_length: int, embed_dim: int, pad_idx: int):
        """
        Initializes a token embedding (with padding support) and a positional embedding layer for a fixed maximum sequence length.
        :param vocab_size: Size of the vocabulary
        :param max_length: Maximum length of the sequence
        :param embed_dim: Depth of the embedding
        :param pad_idx: Index of the padding token
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, seq: torch.Tensor):
        """
        Given an input sequence of token IDs, it computes the corresponding token and positional embeddings, and returns their sum.
        :param seq: Input sequence of token indices (shape: [batch_size, seq_len])
        :return: Embedded sequence with positional encoding
        """
        _, seq_len = seq.size()
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, depth)
        tok_emb = self.token_embedding(seq)  # (batch_size, seq_len, depth)
        return tok_emb + pos_emb  # (batch_size, seq_len, depth)


class CausalSelfAttention(nn.Module):
    """
    Implements masked self-attention for autoregressive generation.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        """
        Initializes multi head self-attention and layer normalization.
        :param hidden_size: The size of the hidden dimension
        :param num_heads: The number of attention heads
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        """
        Computes self-attention using x as query, key, and value. It, then adds a residual connection and applies layer normalization.
        :param x:
        :return:
        """
        attn_output, _ = self.mha(x, x, x, is_causal=True)
        x = x + attn_output  # Residual connection
        return self.layer_norm(x)

    # TODO: Could be redundant. Possibly use is_causal=True in MultiheadAttention instead.
    # @staticmethod
    # def causal_mask(x):
    #     """
    #     Creates triangular mask to prevent looking at future tokens
    #     :param x:
    #     :return:
    #     """
    #     sz = x.size(0)
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask.to(x.device)


class CrossAttention(nn.Module):
    """
    Performs cross-modal attention (e.g., aligning text with image features) by computing attention between the decoder’s text representations and the
    encoder’s image features.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        """
        Sets up multi head attention and layer normalization. It also stores attention weights for potential visualization.
        :param hidden_size: The size of the hidden dimension
        :param num_heads: The number of attention heads
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention_scores = None

    def forward(self, x, y):
        """
        Uses x as the decoder query and y as both key and value (image features), adds the attention output to x (residual connection), and normalizes
        the result.
        :param x:
        :param y:
        :return:
        """
        attn_output, attn_weights = self.mha(x, y, y)  # Query: x (Decoder query), Key: y (Image features), Value: y (Image features)
        self.attention_scores = attn_weights  # Save attention scores for visualization
        x = x + attn_output
        return self.layer_norm(x)


class FeedForward(nn.Module):
    """
    Implements a two-layer feedforward network used in transformer blocks.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        Constructs a sequential model with an intermediate ReLU activation, dropout, and a residual connection with layer normalization.
        :param hidden_size: The size of the hidden dimension
        :param dropout: The dropout rate
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Applies the feedforward network to x, adds the input (residual connection), and normalizes the output.
        :param x:
        :return:
        """
        x = x + self.seq(x)  # Residual connection
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    """
    Represents a single decoder block combining causal self-attention, cross-attention, and feedforward operations.
    """

    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        """
        Initializes the three sub-modules: self-attention, cross-attention, and feedforward network.
        :param hidden_size:
        :param num_heads:
        :param dropout:
        """
        super().__init__()
        self.self_attention = CausalSelfAttention(hidden_size, num_heads)  # Handles image-text attention
        self.cross_attention = CrossAttention(hidden_size, num_heads)  # Aligns text with image features
        self.ff = FeedForward(hidden_size, dropout)  # Further processes representation

    def forward(self, img_features: torch.Tensor, txt_emb: torch.Tensor):
        """
        Processes the text embeddings (txt_emb) first with self-attention, then aligns them with image features via cross-attention, and finally
        passes them through the feedforward network.
        :param img_features:
        :param txt_emb:
        :return:
        """
        txt_emb = self.self_attention(txt_emb)
        txt_emb = self.cross_attention(txt_emb, img_features)
        txt_emb = self.ff(txt_emb)
        return txt_emb


class Output(nn.Module):
    """
    Maps the final hidden representations to vocabulary logits for caption generation while ensuring that banned tokens (PAD, UNK, SOS) are not generated.
    """

    def __init__(self, hidden_size: int, vocab: Vocabulary, banned_indices: list[int]):
        """
        Initializes a linear layer with output dimensions equal to the vocabulary size. Also sets the bias for banned tokens to a large negative
        value (-1e9) to effectively block their generation.
        :param hidden_size:
        :param vocab:
        :param banned_indices:
        """
        super().__init__()
        self.vocab = vocab
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.banned_indices = banned_indices
        # Initialize bias to zeros
        self.linear.bias.data.zero_()
        # Set banned tokens to -1e9 initially
        self.linear.bias.data[self.banned_indices] = -1e9

    def adapt(self):
        """
        Adjusts the output layer’s bias based on the word frequency counts from the vocabulary. Transforms raw word counts into log probabilities and
        resets the bias for banned tokens.
        :return:
        """
        # Transform word counts to indices counts
        idx_counts = Counter()
        for word, count in self.vocab.word_counts.items():
            idx_counts[self.vocab.to_idx(word)] = count

        # Ensure banned tokens are not counted
        for idx in self.banned_indices:
            idx_counts[idx] = 0

        # Compute log probabilities
        total = sum(idx_counts.values())
        if total == 0:
            return
        log_probs = torch.full_like(self.linear.bias.data, -1e9, dtype=torch.float32)
        for idx, count in idx_counts.items():
            if count == 0:
                continue
            prob = count / total
            log_probs[idx] = torch.log(torch.tensor(prob))

        # Ensure banned tokens remain blocked
        log_probs[self.banned_indices] = -1e9
        self.linear.bias.data = log_probs

    def forward(self, x):
        """
        Applies the linear transformation to x, producing logits over the vocabulary.
        :param x:
        :return:
        """
        return self.linear(x)


class ImageCaptioningTransformer(nn.Module):
    """
    The main transformer model that integrates an image encoder, a text embedding layer, several decoder layers, and an output layer to generate
    captions for input images.
    """

    def __init__(self, vocab: Vocabulary, encoder: nn.Module, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 2, max_length: int = 50,
                 dropout: float = 0.1, pad_idx: int = 0):
        """
        Sets up the vocabulary, assigns an external image encoder, creates the sequence embedding, a stack of decoder layers, and the output layer
        with banned token biases.
        :param vocab:
        :param encoder:
        :param hidden_size:
        :param num_layers:
        :param num_heads:
        :param max_length:
        :param dropout:
        :param pad_idx:
        """
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
        self.output_layer = Output(hidden_size, vocab, [vocab.to_idx(t) for t in [PAD, UNK, SOS]])

    # def _init_output_bias(self):
    #     """
    #     The model will be generating text. It should never generate a pad, unknown, or start token.
    #     So set the bias for these to a large negative value.
    #     :return:
    #     """
    #     banned = [self.vocab.to_idx(t) for t in [PAD, UNK, SOS]]
    #     with torch.no_grad():
    #         self.output_layer.bias[banned] = -1e9

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Encodes images using the encoder, reshapes the resulting feature maps, and embeds the captions.
        Processes the embedded text through all decoder layers and outputs the final logits via the output layer.
        :param images:
        :param captions:
        :return:
        """
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
        """
        Switches the model to evaluation mode and encodes the input image.
        Depending on the beam_size parameter, it either uses beam search or temperature sampling to generate captions.
        :param image:
        :param vocab:
        :param max_length:
        :param device:
        :param temperature:
        :param beam_size:
        :return:
        """
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

    def temperature_sampling(self, img_features: torch.Tensor, vocab: Vocabulary, max_length: int, temperature: Optional[float]) -> str:
        """
        Implements autoregressive generation using temperature sampling.
        Starts with the SOS token and iteratively appends tokens based on the output distribution, stopping when the EOS token is generated or
        max_length is reached.
        :param img_features:
        :param vocab:
        :param max_length:
        :param temperature:
        :return:
        """
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

    def beam_search(self, img_features: torch.Tensor, vocab: Vocabulary, max_length: int, beam_size: int) -> str:
        """
        Implements beam search to generate the most likely caption sequence.
        Repeats image features for each beam and, for each step, extends beams by considering the top probable next tokens, applying length
        normalization before choosing the best sequence
        :param img_features:
        :param vocab:
        :param max_length:
        :param beam_size:
        :return:
        """
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


if __name__ == '__main__':
    train_df = pd.read_csv(str(os.path.join(ROOT, TRAIN_CSV)))
    vocab_ = Vocabulary(3, train_df['caption'])
    img_dir = str(os.path.join(ROOT, FLICKR8K_IMG_DIR))
    train_ds = FlickrDataset(img_dir, train_df, vocab_, transform=TRANSFORM)
    encoder_ = intermediate.Encoder(256, 0.2, True)
    print(encoder_)
    model_ = ImageCaptioningTransformer(vocab_, encoder_, 256, 2, 2, 50, 0.1, vocab_.to_idx(PAD))
    print(model_)
