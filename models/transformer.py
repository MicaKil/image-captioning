from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
from torch import Tensor

from constants import SOS, EOS, UNK, PAD
from dataset.vocabulary import Vocabulary
from models.encoders import transformer as t_encoder, swin as swin


class SeqEmbedding(nn.Module):
    """
    Combines token embeddings with positional embeddings to provide contextualized token representations.
    """

    def __init__(self, vocab_size: int, max_len: int, embed_dim: int, pad_idx: int):
        """
        Initializes a token embedding (with padding support) and a positional embedding layer for a fixed maximum sequence length.
        :param vocab_size: Size of the vocabulary
        :param max_len: Maximum length of the sequence
        :param embed_dim: Depth of the embedding
        :param pad_idx: Index of the padding token
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, seq: torch.Tensor):
        """
        Given an input sequence of token IDs, it computes the corresponding token and positional embeddings, and returns their sum.
        :param seq: Input sequence of token indices (shape: [batch_size, seq_len])
        :return: Embedded sequence with positional encoding
        """
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, embed_dim)
        tok_emb = self.token_embedding(seq)  # (batch_size, seq_len, embed_dim)
        return tok_emb + pos_emb  # (batch_size, seq_len, embed_dim)


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
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, txt_emb: torch.Tensor):
        """
        Computes self-attention using x as query, key, and value. It, then adds a residual connection and applies layer normalization.
        :param txt_emb: (batch, seq, feature)
        :return:
        """
        # Generate causal mask on first forward pass
        attn_output, _ = self.mha(txt_emb, txt_emb, txt_emb, attn_mask=self.create_causal_mask(txt_emb))
        txt_emb = txt_emb + attn_output  # Residual connection
        return self.layer_norm(txt_emb)

    @staticmethod
    def create_causal_mask(txt_emb):
        """
        Creates triangular mask to prevent looking at future tokens
        :param txt_emb:
        :return:
        """
        seq_size = txt_emb.size(1)
        mask = torch.triu(
            torch.full((seq_size, seq_size), float('-inf')),
            diagonal=1
        ).to(txt_emb.device)
        return mask.to(txt_emb.device)


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
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention_scores = None

    def forward(self, txt_emb: torch.Tensor, img_features: torch.Tensor):
        """
        Uses x= txt_emb as the decoder query and y = image features as both key and value, adds the attention output to x (residual connection), and normalizes
        the result.
        :param txt_emb: Decoder query, shape: (batch, seq, embed_dim)
        :param img_features: Key and value, shape: (batch, h*w, feature)
        :return:
        """
        attn_output, attn_weights = self.mha(txt_emb, img_features, img_features)
        self.attention_scores = attn_weights  # Save attention scores for visualization
        txt_emb = txt_emb + attn_output
        return self.layer_norm(txt_emb)


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

    def forward(self, txt_emb: torch.Tensor):
        """
        Applies the feedforward network to x, adds the input (residual connection), and normalizes the output.
        :param txt_emb:
        :return:
        """
        txt_emb = txt_emb + self.seq(txt_emb)  # Residual connection
        return self.layer_norm(txt_emb)


class DecoderLayer(nn.Module):
    """
    Represents a single decoder block combining causal self-attention, cross-attention, and feedforward operations.
    """

    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        """
        Initializes the three submodules: self-attention, cross-attention, and feedforward network.
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
        self.banned_indices = banned_indices
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.linear.bias.data.zero_()  # Initialize bias to zeros
        self.linear.bias.data[self.banned_indices] = -1e9  # Set banned tokens to -1e9 initially

    def adapt(self):
        """
        Adjusts the output layer’s bias based on the word frequency counts from the vocabulary. Transforms raw word counts into log probabilities and
        resets the bias for banned tokens.
        :return:
        """
        # Transform word counts to indices counts
        idx_counts = Counter()
        for word, count in self.vocab.word_counts.items():
            idx_counts[self.vocab.str_to_idx(word)] = count
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

    def forward(self, txt_emb):
        """
        Applies the linear transformation to x, producing logits over the vocabulary.
        :param txt_emb:
        :return:
        """
        return self.linear(txt_emb)


class ImageCaptioningTransformer(nn.Module):
    """
    The main transformer model that integrates an image encoder, a text embedding layer, several decoder layers, and an output layer to generate
    captions for input images.
    """

    def __init__(self, encoder: t_encoder.Encoder | swin.Encoder, vocab: Vocabulary, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 2,
                 max_len: int = 50,
                 decoder_dropout: float = 0.5):
        """
        Sets up the vocabulary, assigns an external image encoder, creates the sequence embedding, a stack of decoder layers, and the output layer
        with banned token biases.
        :param encoder:
        :param vocab:
        :param hidden_size:
        :param num_layers:
        :param num_heads:
        :param max_len:
        :param decoder_dropout:
        """
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.encoder = encoder

        self.seq_embedding = SeqEmbedding(len(vocab), max_len, hidden_size, vocab.str_to_idx(PAD))

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, decoder_dropout)
            for _ in range(num_layers)
        ])

        # Output layer with smart initialization
        self.output_layer = Output(hidden_size, vocab, [vocab.str_to_idx(token) for token in [PAD, SOS, UNK]])
        self.output_layer.adapt()  # Initialize output layer bias

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
        # Encode image to features: (batch, features, height, width) → (batch, h*w, features)
        img_features = rearrange(img_features, 'b c h w -> b (h w) c')

        # Embed text
        txt_emb = self.seq_embedding(captions)

        # Process through decoder layers
        for layer in self.decoder_layers:
            txt_emb = layer(img_features, txt_emb)

        # Final output
        logits = self.output_layer(txt_emb)
        return logits

    @staticmethod
    def calc_loss(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Calculate the loss for the given outputs and targets.

        :param outputs: Predicted word indices (batch_size, padded_length, vocab_size)
        :param targets: Target word indices (batch_size, padded_length)
        :param criterion: Loss function
        :return: Loss value
        """
        # Reshape for loss calculation
        logits = outputs.reshape(-1, outputs.size(-1))  # (batch*(seq_len-1), vocab_size)
        targets_flat = targets.reshape(-1)  # (batch*(seq_len-1))

        # Calculate loss per token (without reduction)
        return criterion(logits, targets_flat)  # (batch*(seq_len-1))

    # INFERENCE --------------------------------------------------------------------------------------------------------------------------------------

    def generate(self, images: torch.Tensor, vocab: Vocabulary, max_len: int, device: torch.device, temp: Optional[float], beam_size: int,
                 no_grad: bool, return_attn=False) -> tuple:
        """
        Switches the model to evaluation mode and encodes the input image.
        Depending on the beam_size parameter, it either uses beam search or temperature sampling to generate captions.
        :param images:
        :param vocab:
        :param max_len:
        :param device:
        :param temp:
        :param beam_size:
        :param no_grad:
        :param return_attn:
        :return:
        """
        if max_len > self.max_len:
            max_len = self.max_len

        images = images.to(device)

        if no_grad:
            self.eval()
            with torch.no_grad():
                features = self.encoder(images)
                features = rearrange(features, 'b c h w -> b (h w) c')
                if beam_size > 1:
                    generated = self.beam_search(features, vocab, max_len, beam_size)
                else:
                    generated = self.temperature_sampling(features, vocab, max_len, temp)
        else:
            features = self.encoder(images)
            features = rearrange(features, 'b c h w -> b (h w) c')
            if beam_size > 1:
                generated = self.beam_search(features, vocab, max_len, beam_size)
            else:
                generated = self.temperature_sampling(features, vocab, max_len, temp)

        if return_attn:
            return generated
        return generated[:2]  # Return only captions and log_probs

    def temperature_sampling(self, features: torch.Tensor, vocab: Vocabulary, max_len: int,
                             temp: Optional[float]) -> tuple[list[str], torch.Tensor, list]:
        """
        Implements autoregressive generation using temperature sampling.
        Starts with the SOS token and iteratively appends tokens based on the output distribution, stopping when the EOS token is generated or
        max_length is reached.
        :param features:
        :param vocab:
        :param max_len:
        :param temp:
        :return:
        """
        batch_size = features.size(0)
        tokens = torch.full((batch_size, 1), vocab.str_to_idx(SOS), device=features.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        log_probs = torch.zeros(batch_size, device=features.device)
        all_attn = []

        for _ in range(max_len):
            txt_emb = self.seq_embedding(tokens)
            attn_step = []
            for layer in self.decoder_layers:
                txt_emb = layer(features, txt_emb)
                attn = layer.cross_attention.attention_scores
                attn = attn.mean(dim=1).squeeze(1).squeeze(1)  # Average attention over heads and remove singleton dimensions
                attn_step.append(attn.detach().cpu().numpy())
            all_attn.append(attn_step)

            logits = self.output_layer(txt_emb[:, -1, :])

            # Compute log probabilities (scaling logits if temperature is provided)
            if temp is not None and temp > 0:
                logits_scaled = logits / temp
            else:
                logits_scaled = logits
            log_probs_step = f.log_softmax(logits_scaled, dim=-1)

            # Sample from the probability distribution
            # Since multinomial works on probabilities, we exponentiate the log probabilities.
            probs = log_probs_step.exp()

            if temp is not None and temp > 0:
                next_tokens = torch.multinomial(probs, 1)
            else:
                next_tokens = logits_scaled.argmax(dim=-1, keepdim=True)

            # Gather the log probability of the sampled token.
            selected_log_prob = log_probs_step.gather(1, next_tokens).squeeze(1)

            # Only update log_probs for sequences not finished
            log_probs = log_probs + torch.where(finished, torch.zeros_like(selected_log_prob), selected_log_prob)

            # Replace tokens for finished sequences with PAD
            next_tokens = torch.where(finished.unsqueeze(-1), torch.tensor(vocab.str_to_idx(PAD), device=features.device), next_tokens)
            tokens = torch.cat([tokens, next_tokens], dim=1)

            # Update finished flags
            finished = finished | (next_tokens.squeeze(1) == vocab.str_to_idx(EOS))
            if finished.all():
                break

        captions = [vocab.encode_as_words(seq.tolist()) for seq in tokens]
        return captions, log_probs, all_attn

    def beam_search(self, images: torch.Tensor, vocab: Vocabulary, max_len: int, beam_size: int) -> tuple[list[str], Tensor, list]:
        """
        Implements beam search to generate the most likely caption sequence.
        Repeats image features for each beam and, for each step, extends beams by considering the top probable next tokens, applying length
        normalization before choosing the best sequence
        :param images: Batch of images features
        :param vocab:
        :param max_len:
        :param beam_size:
        :return:
        """

        batch_size = images.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)

        captions = []
        all_probs = []
        all_attn = []

        # Process each image in the batch separately
        for b in range(batch_size):
            image = images[b].unsqueeze(0).repeat(beam_size, 1, 1)
            beams = [(0.0, [sos_idx], [], [])]  # (score, sequence, log_probs)

            for t in range(max_len):
                # Prepare current beam sequences, skipping extension for beams that already ended
                active_beams = [beam for beam in beams if beam[1][-1] != eos_idx]
                # If no beams are active, break out of the loop
                if len(active_beams) == 0:
                    break

                # For candidates that are active, extend them
                seqs = [beam[1] for beam in active_beams]
                max_seq_length = max(len(seq) for seq in seqs)
                padded_seqs = [seq + [vocab.str_to_idx(PAD)] * (max_seq_length - len(seq)) for seq in seqs]
                seq_tensor = torch.tensor(padded_seqs, device=image.device)

                txt_emb = self.seq_embedding(seq_tensor)
                step_attn = []
                for layer in self.decoder_layers:
                    txt_emb = layer(image[:seq_tensor.size(0)], txt_emb)
                    attn = layer.cross_attention.attention_scores
                    attn = attn.mean(dim=1).squeeze(1).squeeze(1)
                    step_attn.append(attn.detach().cpu().numpy())
                logits = self.output_layer(txt_emb[:, -1, :])
                step_log_probs = f.log_softmax(logits, dim=-1)  # (active_beams, vocab_size)
                avg_attn = np.mean(step_attn, axis=0)  # Average attention over layers

                candidates = []
                for i, beam in enumerate(active_beams):
                    top_probs, top_indices = step_log_probs[i].topk(beam_size)
                    for j in range(beam_size):
                        new_score = beam[0] + top_probs[j].item()
                        new_seq = beam[1] + [top_indices[j].item()]
                        new_log_probs = beam[2] + [top_probs[j].item()]
                        new_attn = beam[3] + [avg_attn[i]]
                        candidates.append((new_score, new_seq, new_log_probs, new_attn))

                # Add beams that have already finished (without extension)
                finished_beams = [beam for beam in beams if beam[1][-1] == eos_idx]
                # Combine and sort candidates
                all_candidates = candidates + finished_beams
                all_candidates.sort(reverse=True, key=lambda x: x[0] / (len(x[1]) ** 0.5))
                beams = all_candidates[:beam_size]

                # If all beams have ended, break early
                if all(beam[1][-1] == eos_idx for beam in beams):
                    break

            # After max_len iterations, choose the best beam
            best_beam = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5))
            captions.append(vocab.encode_as_words(best_beam[1]))
            all_probs.append(sum(best_beam[2]))
            all_attn.append(best_beam[3])

        return captions, torch.tensor(all_probs, device=images.device), all_attn
