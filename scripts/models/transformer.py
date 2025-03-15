from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from torch import Tensor
from torch.nn.functional import log_softmax
from torchvision.models import ResNet50_Weights

from constants import SOS, EOS, UNK, PAD
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.vocabulary import Vocabulary
from scripts.test import test
from scripts.train import train


class Encoder(nn.Module):
    """
    Encoder class that uses a pretrained ResNet-50 model to extract features from images.
    """

    def __init__(self, embed_dim: int, dropout: float, fine_tune: str):
        """
        Constructor for the EncoderResnet class

        :param embed_dim: Size of the embedding vector
        :param dropout: Dropout probability
        :param fine_tune: Whether to fine-tune the last two layers of the ResNet-50 model
        """
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1]  # Remove the last FC layer (Classification layer)
        )  # Output shape: (batch, feature_dim, 1, 1)

        self.set_requires_grad(fine_tune)

        self.projection = nn.Sequential(
            nn.Conv2d(in_features, embed_dim, kernel_size=1),  # Reduce to embed_dim (Shape: (batch, embed_dim, 1, 1))
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def set_requires_grad(self, fine_tune: str) -> None:
        """
        Set the requires_grad attribute of the parameters based on the fine_tune argument.
        :param fine_tune: String indicating the fine-tuning strategy. Can be "full", "partial", or "none".
        :return:
        """
        match fine_tune:
            case "full":
                return
            case "partial":
                # Freeze all layers except the last two layers of the ResNet-50 model
                for param in self.resnet.parameters():
                    param.requires_grad = False
                if fine_tune:
                    # Unfreeze the last two layers of the ResNet-50 model
                    for layer in list(self.resnet.children())[-2:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                return
            case _:
                for param in self.resnet.parameters():
                    param.requires_grad = False
                return

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder

        :param image: Input image tensor of shape (batch_size, 3, 224, 224)
        :return:
        """
        features = self.resnet(image)  # Shape: (batch_size, feature_dim, 1, 1)
        features = self.projection(features)  # Shape: (batch_size, embed_size, 1, 1)
        return features


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
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
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
        Uses x as the decoder query and y as both key and value (image features), adds the attention output to x (residual connection), and normalizes
        the result.
        :param txt_emb: Decoder query, shape: (batch, seq, feature)
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

    def __init__(self, vocab: Vocabulary, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 2, max_length: int = 50,
                 encoder_dropout=0.1, decoder_dropout: float = 0.5, fine_tune_encoder=None):
        """
        Sets up the vocabulary, assigns an external image encoder, creates the sequence embedding, a stack of decoder layers, and the output layer
        with banned token biases.
        :param vocab:
        :param hidden_size:
        :param num_layers:
        :param num_heads:
        :param max_length:
        :param encoder_dropout:
        :param decoder_dropout:
        :param fine_tune_encoder:
        """
        super().__init__()
        self.vocab = vocab
        self.max_length = max_length
        self.encoder = Encoder(hidden_size, encoder_dropout, fine_tune_encoder)

        self.seq_embedding = SeqEmbedding(len(vocab), max_length, hidden_size, vocab.str_to_idx(PAD))

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

    def generate(self, images: torch.Tensor, vocab: Vocabulary, max_length: int, device: torch.device, temperature: Optional[float], beam_size: int,
                 no_grad: bool) -> tuple[list[str], torch.Tensor]:
        """
        Switches the model to evaluation mode and encodes the input image.
        Depending on the beam_size parameter, it either uses beam search or temperature sampling to generate captions.
        :param images:
        :param vocab:
        :param max_length:
        :param device:
        :param temperature:
        :param beam_size:
        :param no_grad:
        :return:
        """
        if max_length > self.max_length:
            # logger.warning(f"Max sequence length ({max_length}) is greater than model's ({self.max_length}). Using model's max length.")
            max_length = self.max_length

        images = images.to(device)

        if no_grad:
            self.eval()
            with torch.no_grad():
                # Encode image
                features = self.encoder(images)
                features = rearrange(features, 'b c h w -> b (h w) c')

                if beam_size > 1:
                    return self.beam_search(features, vocab, max_length, beam_size)
                else:
                    return self.temperature_sampling(features, vocab, max_length, temperature)

        # Encode image
        features = self.encoder(images)
        features = rearrange(features, 'b c h w -> b (h w) c')
        if beam_size > 1:
            return self.beam_search(features, vocab, max_length, beam_size)
        else:
            return self.temperature_sampling(features, vocab, max_length, temperature)

    def temperature_sampling(self, features: torch.Tensor, vocab: Vocabulary, max_length: int,
                             temperature: Optional[float]) -> tuple[list[str], Tensor]:
        """
        Implements autoregressive generation using temperature sampling.
        Starts with the SOS token and iteratively appends tokens based on the output distribution, stopping when the EOS token is generated or
        max_length is reached.
        :param features:
        :param vocab:
        :param max_length:
        :param temperature:
        :return:
        """
        batch_size = features.size(0)
        tokens = torch.full((batch_size, 1), vocab.str_to_idx(SOS), device=features.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        log_probs = []  # Track log probabilities for each step

        for _ in range(max_length):
            txt_emb = self.seq_embedding(tokens)
            for layer in self.decoder_layers:
                txt_emb = layer(features, txt_emb)
            logits = self.output_layer(txt_emb[:, -1, :])

            if tokens.size(1) > 1:
                last_token = tokens[:, -1]
                # Penalize the logits for the last generated token (per batch element)
                logits[torch.arange(batch_size), last_token] -= 1.0  # Reduce probability

            if temperature is not None and temperature != 0:
                # Apply temperature scaling
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                # Greedy sampling if temperature=0
                probs = torch.softmax(logits, dim=-1)

            # Sample tokens and compute log probabilities
            sampler = torch.distributions.Categorical(probs)
            next_tokens = sampler.sample()
            log_probs_step = sampler.log_prob(next_tokens)  # (batch_size,)

            # Mask finished sequences (no further probability accumulation)
            log_probs_step = torch.where(finished, 0.0, log_probs_step)
            log_probs.append(log_probs_step)
            # Update tokens
            next_tokens = torch.where(finished.unsqueeze(-1), vocab.str_to_idx(PAD), next_tokens.unsqueeze(-1))
            tokens = torch.cat([tokens, next_tokens], dim=1)

            # Update finished flags
            finished = finished | (next_tokens.squeeze() == vocab.str_to_idx(EOS))

            if finished.all():
                break

        log_probs = torch.stack(log_probs, dim=0).sum(dim=0)  # Sum log probabilities across steps
        captions = [vocab.encode_as_words(seq.tolist()) for seq in tokens]
        return captions, log_probs

    def beam_search(self, features: torch.Tensor, vocab: Vocabulary, max_length: int, beam_size: int) -> tuple[list[str], Tensor]:
        """
        Implements beam search to generate the most likely caption sequence.
        Repeats image features for each beam and, for each step, extends beams by considering the top probable next tokens, applying length
        normalization before choosing the best sequence
        :param features:
        :param vocab:
        :param max_length:
        :param beam_size:
        :return:
        """
        batch_size = features.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)

        captions = []
        all_probs = []  # To store log probabilities for each sequence

        # Process each image in batch separately
        for i in range(batch_size):
            image = features[i].unsqueeze(0)
            beams = [(0.0, [sos_idx], [])]  # (score, sequence, log_probs)

            for _ in range(max_length):
                candidates = []
                for score, seq, log_probs in beams:
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq, log_probs))
                        continue

                    tokens = torch.tensor([seq], device=image.device)
                    txt_emb = self.seq_embedding(tokens)
                    for layer in self.decoder_layers:
                        txt_emb = layer(image, txt_emb)
                    logits = self.output_layer(txt_emb[:, -1, :])

                    step_probs = log_softmax(logits, dim=-1)
                    top_probs, top_indices = step_probs.topk(beam_size)

                    for j in range(beam_size):
                        candidates.append((
                            score + top_probs[0, j].item(),
                            seq + [top_indices[0, j].item()],
                            log_probs + [top_probs[0, j].item()]
                        ))

                # Keep top-k candidates with length normalization
                candidates.sort(reverse=True, key=lambda x: x[0] / (len(x[1]) ** 0.5))
                beams = candidates[:beam_size]

                if all(seq[-1] == eos_idx for _, seq, _ in beams):
                    break

            # Select best beam
            best_beam = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5))
            captions.append(vocab.encode_as_words(best_beam[1]))
            all_probs.append(sum(best_beam[2]))  # Sum of log probabilities

        return captions, torch.tensor(all_probs, device=features.device)

    # TRAINING ---------------------------------------------------------------------------------------------------------------------------------------

    def train_model(self, train_loader: CaptionLoader, val_loader: CaptionLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim,
                    scheduler: torch.optim.lr_scheduler, checkpoint_dir: str, use_wandb: bool, run_config: dict,
                    resume_checkpoint: str) -> tuple:
        """
        Training loop for the model.

        :param resume_checkpoint:
        :param train_loader: DataLoader for the training set
        :param val_loader: DataLoader for the validation set
        :param device: Device to run the training on
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param scheduler: Learning rate scheduler
        :param checkpoint_dir: Directory to save the best model
        :param use_wandb: Whether to use Weights & Biases for logging
        :param run_config: Configuration for the run
        :return: Path to the best model
        """
        return train(self, train_loader, val_loader, device, criterion, optimizer, scheduler, checkpoint_dir, use_wandb, run_config,
                     resume_checkpoint)

    def test_model(self, test_loader: CaptionLoader, device: torch.device, save_dir: str, tag: str, use_wandb: bool, run_config: dict) -> tuple:
        """
        Evaluate model on test set and log results
        :param test_loader: Test data loader to use
        :param device: Device to use (cpu or cuda)
        :param save_dir: If not None, save results to this directory
        :param tag: Tag to use for saving results
        :param use_wandb: Whether to use Weights & Biases for logging
        :param run_config: Configuration for the run if not using wandb
        :return:
        """
        return test(self, test_loader, device, save_dir, tag, use_wandb, run_config)
