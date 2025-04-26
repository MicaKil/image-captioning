from collections import Counter
from typing import Optional, List, Union  # Corrected typing imports for clarity

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Renamed import to F for convention
from einops import rearrange

from constants import SOS, EOS, UNK, PAD
from dataset.vocabulary import Vocabulary
from models.encoders import transformer as t_encoder, swin as swin


class SeqEmbedding(nn.Module):
    """
    Combines token embeddings with positional embeddings.

    This layer takes a sequence of token indices and produces a sequence of embeddings where each embedding is the sum of the token's embedding and
    its positional embedding. It supports padding by ignoring the gradient updates for the padding token index.
    """

    def __init__(self, vocab_size: int, max_len: int, embed_dim: int, pad_idx: int):
        """
        Initializes the token and positional embedding layers.

        :param vocab_size: The total number of unique tokens in the vocabulary.
        :param max_len: The maximum sequence length that this model can handle. Determines the size of the positional embedding table.
        :param embed_dim: The dimensionality of the embedding vectors for both tokens and positions.
        :param pad_idx: The index of the padding token in the vocabulary. This index will be ignored during embedding lookup and gradient computation.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Positional embeddings for sequences up to max_len
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined token and positional embeddings for the input sequence.

        :param seq: A tensor of token indices with shape (batch_size, seq_len).
        :returns: The combined embeddings tensor with shape (batch_size, seq_len, embed_dim).
        """
        seq_len = seq.size(1)
        # Create position indices (0, 1, ..., seq_len-1)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)  # Shape: (1, seq_len)
        pos_emb = self.pos_embedding(positions)  # Shape: (1, seq_len, embed_dim)
        tok_emb = self.token_embedding(seq)  # Shape: (batch_size, seq_len, embed_dim)
        # Broadcasting adds positional embeddings to token embeddings for each item in the batch
        return tok_emb + pos_emb


class CausalSelfAttention(nn.Module):
    """
    Implements masked multi-head self-attention for autoregressive tasks.

    This layer applies self-attention to the input sequence, but uses a causal mask to prevent positions from attending to subsequent positions. This
    is essential for sequence generation tasks where the prediction of the next token should only depend on the preceding tokens. Includes a residual
    connection and layer normalization.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        """
        Initializes the multi-head attention mechanism and layer normalization.

        :param hidden_size: The dimensionality of the input and output features. Must be divisible by num_heads.
        :param num_heads: The number of attention heads to use.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Applies causal self-attention to the input text embeddings.

        :param txt_emb: Input tensor with shape (batch_size, seq_len, hidden_size).
        :returns: Output tensor after attention, residual connection, and layer normalization, with shape (batch_size, seq_len, hidden_size).
        """
        # Generate causal mask to prevent attending to future positions
        causal_mask = self.create_causal_mask(txt_emb)
        # Compute self-attention (query, key, value are all txt_emb)
        attn_output, _ = self.mha(txt_emb, txt_emb, txt_emb, attn_mask=causal_mask)
        # Residual connection and layer normalization
        txt_emb = txt_emb + attn_output
        return self.layer_norm(txt_emb)

    @staticmethod
    def create_causal_mask(txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Creates a causal mask for self-attention. The mask ensures that a position i cannot attend to positions j > i. It's a square matrix where the
        upper triangle (above the diagonal) is masked.

        :param txt_emb: Input tensor, used to determine the sequence length and device. Shape (batch_size, seq_len, hidden_size).
        :returns: A causal mask tensor of shape (seq_len, seq_len) with -inf in masked positions and 0.0 otherwise.
        """
        seq_size = txt_emb.size(1)
        # Create an upper triangular matrix filled with -inf
        mask = torch.triu(
            torch.full((seq_size, seq_size), float('-inf'), device=txt_emb.device),
            diagonal=1  # Start masking from the first diagonal above the main diagonal
        )
        return mask


class CrossAttention(nn.Module):
    """
    Implements multi-head cross-attention between text and image features.

    This layer allows the text representations (query) to attend to the image features (key and value). It's crucial for grounding the generated text
    in the visual content. Includes a residual connection and layer normalization. Stores attention weights for potential analysis or visualization.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        """
        Initializes the multi-head attention mechanism and layer normalization.

        :param hidden_size: The dimensionality of the text embeddings and image features. Must be divisible by num_heads.
        :param num_heads: The number of attention heads to use.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention_scores = None  # To store attention weights

    def forward(self, txt_emb: torch.Tensor, img_features: torch.Tensor) -> torch.Tensor:
        """
        Applies cross-attention from text embeddings to image features.

        :param txt_emb: Text embeddings (query) with shape (batch_size, text_seq_len, hidden_size).
        :param img_features: Image features (key and value) with shape (batch_size, img_seq_len, hidden_size), where img_seq_len is typically
                             height * width of the feature map.
        :returns: Output tensor after attention, residual connection, and layer normalization, with shape (batch_size, text_seq_len, hidden_size).
        """
        # Compute cross-attention: query=txt_emb, key=img_features, value=img_features
        attn_output, attn_weights = self.mha(txt_emb, img_features, img_features)
        self.attention_scores = attn_weights  # Store attention weights (batch, num_heads, text_seq_len, img_seq_len)
        # Residual connection and layer normalization
        txt_emb = txt_emb + attn_output
        return self.layer_norm(txt_emb)


class FeedForward(nn.Module):
    """
    Implements the position-wise Feed-Forward Network (FFN) of a Transformer block.

    Consists of two linear transformations with a ReLU activation in between, followed by dropout. Includes a residual connection and layer
    normalization.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        Initializes the feed-forward network layers.

        :param hidden_size: The dimensionality of the input and output features.
        :param dropout (float): The dropout probability to apply after the second linear layer. Defaults to 0.1.
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),  # Expansion layer
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),  # Contraction layer
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward network to the input tensor.

        :param txt_emb: Input tensor with shape (batch_size, seq_len, hidden_size).
        :returns: Output tensor after the FFN, residual connection, and layer normalization, with shape (batch_size, seq_len, hidden_size).
        """
        # Apply feed-forward, add residual connection, and normalize
        processed_emb = self.seq(txt_emb)
        txt_emb = txt_emb + processed_emb
        return self.layer_norm(txt_emb)


class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer decoder.

    This layer combines causal self-attention, cross-attention (with encoder outputs), and a feed-forward network. Each sub-layer (attention, FFN)
    includes residual connections and layer normalization.
    """

    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        """
        Initializes the components of the decoder layer.

        :param hidden_size: The dimensionality of the features used throughout the layer.
        :param num_heads: The number of attention heads for both self-attention and cross-attention. Defaults to 1.
        :param dropout: The dropout rate for the feed-forward network. Defaults to 0.1.
        """
        super().__init__()
        # Masked self-attention for the decoder inputs
        self.self_attention = CausalSelfAttention(hidden_size, num_heads)
        # Cross-attention between decoder inputs and encoder outputs (image features)
        self.cross_attention = CrossAttention(hidden_size, num_heads)
        # Position-wise feed-forward network
        self.ff = FeedForward(hidden_size, dropout)

    def forward(self, img_features: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Passes the text embeddings through the decoder layer, interacting with image features.

        :param img_features: Image features from the encoder, used as key/value in cross-attention. Shape (batch, img_seq_len, hidden_size).
        :param txt_emb: Text embeddings from the previous layer or embedding layer. Shape (batch, txt_seq_len, hidden_size).

        :returns: The processed text embeddings after passing through the layer. Shape (batch, txt_seq_len, hidden_size).
        """
        # 1. Causal Self-Attention on text embeddings
        txt_emb = self.self_attention(txt_emb)
        # 2. Cross-Attention with image features
        txt_emb = self.cross_attention(txt_emb, img_features)
        # 3. Feed-Forward Network
        txt_emb = self.ff(txt_emb)
        return txt_emb


class Output(nn.Module):
    """
    Maps the final decoder hidden states to vocabulary logits.

    Includes functionality to prevent the generation of specified banned tokens (like PAD, UNK, SOS) by setting their corresponding biases to a large
    negative value. Also offers an adapt method to initialize biases based on vocabulary frequency, promoting more common words initially.
    """

    def __init__(self, hidden_size: int, vocab: Vocabulary, banned_indices: List[int]):
        """
        Initializes the final linear layer and sets biases for banned tokens.

        :param hidden_size: The dimensionality of the input hidden states from the decoder.
        :param vocab: The vocabulary object, used to determine the output size (vocabulary size) and access word counts for bias adaptation.
        :param banned_indices: A list of token indices that should not be generated. Their corresponding biases will be set to -infinity.
        """
        super().__init__()
        self.vocab = vocab
        self.banned_indices = banned_indices
        # Linear layer mapping hidden state to vocabulary size
        self.linear = nn.Linear(hidden_size, len(vocab))
        # Initialize biases to zero
        self.linear.bias.data.zero_()
        # Set biases for banned tokens to a large negative value to prevent their selection
        self.linear.bias.data[self.banned_indices] = -1e9

    def adapt(self):
        """
        Adapts the output layer's bias based on token frequencies in the vocabulary.

        This method calculates the log probability of each token based on its count in the vocab.word_counts and sets the linear layer's bias
        accordingly. This can help the model start generating more plausible (common) words early in training. Banned tokens have their biases reset
        to -infinity after calculation.
        """
        # Count occurrences of each index in the vocabulary
        idx_counts = Counter()
        for word, count in self.vocab.word_counts.items():
            idx_counts[self.vocab.str_to_idx(word)] = count
        # Ensure banned tokens are not counted
        for idx in self.banned_indices:
            idx_counts[idx] = 0

        total = sum(idx_counts.values())
        if total == 0:
            # Ensure biases are initialized reasonably even if adaptation fails
            self.linear.bias.data.zero_()
            self.linear.bias.data[self.banned_indices] = -1e9
            return

        # Initialize log probabilities to -infinity
        log_probs = torch.full_like(self.linear.bias.data, -1e9, dtype=torch.float32)

        # Calculate log probability for each token index
        for idx, count in idx_counts.items():
            if count > 0:  # Ensure count is positive to avoid log(0)
                prob = count / total
                log_probs[idx] = torch.log(torch.tensor(prob, dtype=torch.float32))

        # Ensure banned tokens remain strongly discouraged
        log_probs[self.banned_indices] = -1e9

        # Set the computed log probabilities as the bias
        self.linear.bias.data = log_probs

    def forward(self, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to produce vocabulary logits.

        :param txt_emb: The final hidden states from the decoder. Shape (batch_size, seq_len, hidden_size).
        :returns: Logits over the vocabulary. Shape (batch_size, seq_len, vocab_size).
        """
        return self.linear(txt_emb)


class ImageCaptioner(nn.Module):
    """
    An Image Captioning model using a Transformer architecture.

    This model integrates an image encoder (like a CNN or Vision Transformer), a text embedding layer, a stack of Transformer decoder layers, and an
    output layer to generate captions for input images. It supports both training (via forward and calc_loss) and inference (via generate, using
    either temperature sampling or beam search).
    """

    def __init__(self, encoder: Union[t_encoder.Encoder, swin.Encoder], vocab: Vocabulary, hidden_size: int = 256, num_layers: int = 2,
                 num_heads: int = 2, max_len: int = 50, decoder_dropout: float = 0.5):
        """
        Initializes the Image Captioning Transformer model.

        :param encoder: An instantiated image encoder module (e.g., a pre-trained ResNet or Swin Transformer)
               that outputs image features.
        :param vocab: The vocabulary object containing token-to-index mappings and vice-versa.
        :param hidden_size: The dimensionality of embeddings and hidden states used throughout the transformer decoder. Defaults to 256.
        :param num_layers: The number of decoder layers to stack. Defaults to 2.
        :param num_heads: The number of attention heads in each decoder layer's self-attention and cross-attention mechanisms. Defaults to 2.
        :param max_len: The maximum length of the generated captions, including special tokens. Also used for positional embeddings. Defaults to 50.
        :param decoder_dropout: The dropout rate applied within the feed-forward networks of the decoder layers. Defaults to 0.5.
        """
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.encoder = encoder  # Image feature extractor

        # Text embedding layer (token + position)
        self.seq_embedding = SeqEmbedding(len(vocab), max_len, hidden_size, vocab.str_to_idx(PAD))

        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, decoder_dropout)
            for _ in range(num_layers)
        ])

        # Output layer to predict next token probabilities
        banned_tokens = [PAD, SOS, UNK]
        self.output_layer = Output(hidden_size, vocab, [vocab.str_to_idx(token) for token in banned_tokens])
        self.output_layer.adapt()  # Initialize output layer bias based on word frequency

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass for training or teacher-forcing evaluation.

        Encodes the images, embeds the input captions (target captions shifted right), and passes them through the decoder layers to predict the next
        token logits at each position.

        :param images: A batch of input images. Shape (batch_size, C, H, W).
        :param captions: A batch of target caption sequences (token indices), typically including SOS at the start. Shape (batch_size, seq_len).
        :returns: Logits predicting the next token for each position in the input captions.Shape (batch_size, seq_len, vocab_size).
        """
        # 1. Encode images to get features
        img_features = self.encoder(images)  # Shape: (batch, features, height, width)
        # Reshape image features for attention: (batch, h*w, features)
        img_features = rearrange(img_features, 'b c h w -> b (h w) c')

        # 2. Embed captions (token + position)
        txt_emb = self.seq_embedding(captions)  # Shape: (batch, seq_len, hidden_size)

        # 3. Process through decoder layers
        for layer in self.decoder_layers:
            txt_emb = layer(img_features, txt_emb)  # Shape remains (batch, seq_len, hidden_size)

        # 4. Generate logits over vocabulary
        logits = self.output_layer(txt_emb)  # Shape: (batch, seq_len, vocab_size)
        return logits

    @staticmethod
    def calc_loss(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Calculates the loss between predicted logits and target sequences.

        Typically used with CrossEntropyLoss. Reshapes the outputs and targets
        to be compatible with standard loss functions expecting (N, C) and (N) shapes.

        :param outputs: The logits predicted by the model's forward pass. Shape (batch_size, seq_len, vocab_size).
        :param targets: The ground truth target sequences (token indices). Should correspond to the outputs but shifted, e.g., excluding the SOS
                        token and including an EOS token. Shape (batch_size, seq_len).
        :param criterion: The loss function (e.g., nn.CrossEntropyLoss). Note: If using CrossEntropyLoss, ensure it's initialized with
                          ignore_index=PAD_IDX if applicable.
        :returns: The computed loss value (scalar if reduction is 'mean' or 'sum').
        """
        # Reshape for loss calculation:
        # Logits: (batch * seq_len, vocab_size)
        # Targets: (batch * seq_len)
        logits_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)

        # Calculate loss
        loss = criterion(logits_flat, targets_flat)
        return loss

    # -----------------------------------------------------------------------------------
    # INFERENCE METHODS
    # -----------------------------------------------------------------------------------

    def generate(self, images: torch.Tensor, vocab: Vocabulary, max_len: int, device: torch.device, temp: Optional[float], beam_size: int,
                 no_grad: bool, return_attn: bool = False) -> tuple:
        """
        Generates captions for a batch of images using either temperature sampling or beam search.

        Switches the model to evaluation mode (eval()).

        :param images: A batch of input images. Shape (batch_size, C, H, W).
        :param vocab: The vocabulary object.
        :param max_len: The maximum length for the generated caption.
        :param device: The device to perform computation on (e.g., 'cuda', 'cpu').
        :param temp: Temperature for sampling. If None or 0, uses argmax (greedy). If > 0, scales logits before sampling. Only used if beam_size=1.
        :param beam_size: The beam width for beam search. If 1, uses temperature sampling (or greedy decoding if temp is None/0).
        :param no_grad: Whether to disable gradient computation. If True, the model will not track gradients.
        :param return_attn: Whether to return attention weights along with captions. Defaults to False.

        :returns: tuple containing:
            - captions (List[List[str]]): A list of generated captions, where each caption is a list of word strings.
            - log_probs (torch.Tensor): A tensor of total log probabilities for each generated caption. Shape (batch_size,).
            - all_attns (Optional[List]): If return_attn is True, a list (per batch item) of attention weights collected during generation. The
                                          structure depends on the generation method. Otherwise, this element is omitted.
        """
        # Ensure max_len does not exceed model's capability
        effective_max_len = min(max_len, self.max_len)
        images = images.to(device)

        if no_grad:
            self.eval()
            with torch.no_grad():
                generated = self.generate_common(beam_size, effective_max_len, images, temp, vocab)
        else:
            generated = self.generate_common(beam_size, effective_max_len, images, temp, vocab)

        # 3. Return based on return_attn flag
        if return_attn:
            return generated  # (captions, log_probs, all_attn)
        # Return only captions and log probabilities
        return generated[0], generated[1]

    def generate_common(self, beam_size: int, effective_max_len: int, images: torch.Tensor, temp: float, vocab: Vocabulary) -> tuple:
        """
        Encodes the images and generates captions using either beam search or temperature sampling.
        :param images: A batch of input images. Shape (batch_size, C, H, W).
        :param vocab: The vocabulary object.
        :param effective_max_len: The maximum length for the generated caption.
        :param temp: Temperature for sampling. If None or 0, uses argmax (greedy). If > 0, scales logits before sampling. Only used if beam_size=1.
        :param beam_size: The beam width for beam search. If 1, uses temperature sampling (or greedy decoding if temp is None/0).
        :returns: tuple containing:
            - captions (List[List[str]]): A list of generated captions, where each caption is a list of word strings.
            - log_probs (torch.Tensor): A tensor of total log probabilities for each generated caption. Shape (batch_size,).
            - all_attns (Optional[List]): If return_attn is True, a list (per batch item) of attention weights collected during generation. The
                                          structure depends on the generation method. Otherwise, this element is omitted.
        """
        # 1. Encode images
        features = self.encoder(images)  # Shape: (batch, features, height, width)
        # Reshape features for decoder: (batch, h*w, features)
        features = rearrange(features, 'b c h w -> b (h w) c')
        # 2. Generate captions using the selected method
        if beam_size > 1:
            generated = self.beam_search(features, vocab, effective_max_len, beam_size)
        else:
            generated = self.temperature_sampling(features, vocab, effective_max_len, temp)
        return generated

    def temperature_sampling(self, features: torch.Tensor, vocab: Vocabulary, max_len: int, temp: Optional[float]) -> tuple:
        """
        Generates captions autoregressively using temperature sampling or greedy decoding.

        Starts with the SOS token and iteratively predicts the next token based on the model's output distribution, optionally scaled by temperature.
        Stops when EOS is generated or max_len is reached for all sequences in the batch.

        :param features: Encoded image features. Shape (batch_size, img_seq_len, hidden_size).
        :param vocab: The vocabulary object.
        :param max_len: The maximum generation length.
        :param temp: Temperature for sampling. If None or 0, uses argmax (greedy).
        :returns: Tuple containing:
                - captions: List of generated word lists for the batch.
                - log_probs: Total log probability for each caption. Shape (batch_size,).
                - all_attn: List containing attention weights for each step. Outer list length is num_steps, inner list length is num_decoder_layers.
                            Each element is a numpy array (batch_size, img_seq_len) averaged over heads.
        """
        batch_size = features.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)
        pad_idx = vocab.str_to_idx(PAD)
        device = features.device

        # Initialize sequences with SOS token
        tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        # Keep track of finished sequences (initially all False)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Accumulate log probabilities for each sequence
        total_log_probs = torch.zeros(batch_size, device=device)
        # Store attention weights per step
        all_attn_steps = []

        for _ in range(max_len - 1):  # Max length includes SOS, so loop max_len-1 times
            # 1. Embed current tokens
            txt_emb = self.seq_embedding(tokens)  # Shape: (batch, current_len, hidden_size)

            # 2. Pass through decoder layers
            attn_weights_current_step = []  # Store attention from layers at this step
            for layer in self.decoder_layers:
                txt_emb = layer(features, txt_emb)
                # Get cross-attention scores (batch, num_heads, txt_len, img_len)
                attn = layer.cross_attention.attention_scores
                # Average over heads and take the scores for the *last* token attending to image features
                # Shape: (batch, img_seq_len)
                avg_attn_last_token = attn[:, :, -1, :].mean(dim=1)  # attn = attn.mean(dim=1).squeeze(1).squeeze(1)
                attn_weights_current_step.append(avg_attn_last_token.detach().cpu().numpy())
            all_attn_steps.append(attn_weights_current_step)

            # 3. Get logits for the next token (only need the last token's output)
            logits = self.output_layer(txt_emb[:, -1, :])  # Shape: (batch, vocab_size)

            # 4. Apply temperature scaling if specified
            if temp is not None and temp > 0:
                logits_scaled = logits / temp
            else:
                logits_scaled = logits  # Use original logits for greedy (temp=0 or None)

            # 5. Calculate log probabilities for the next token
            log_probs_step = F.log_softmax(logits_scaled, dim=-1)  # Shape: (batch, vocab_size)

            # 6. Sample the next token
            if temp is not None and temp > 0:
                # Sample using probabilities (exp(log_probs))
                probs = log_probs_step.exp()
                next_tokens = torch.multinomial(probs, num_samples=1)  # Shape: (batch, 1)
            else:
                # Greedy decoding: select the token with the highest log probability
                # next_tokens = logits_scaled.argmax(dim=-1, keepdim=True)
                next_tokens = log_probs_step.argmax(dim=-1, keepdim=True)  # Shape: (batch, 1)

            # 7. Gather the log probability of the chosen token
            # Shape: (batch,)
            selected_log_prob = log_probs_step.gather(dim=1, index=next_tokens).squeeze(1)

            # 8. Update total log probabilities, only for sequences that are not yet finished
            # Use torch.where to conditionally add: add 0 if finished, else add selected_log_prob
            total_log_probs += torch.where(finished, torch.zeros_like(selected_log_prob), selected_log_prob)

            # 9. Mask out updates for finished sequences: replace next token with PAD
            next_tokens = torch.where(finished.unsqueeze(-1), torch.tensor(pad_idx, device=device), next_tokens)

            # 10. Append the new token to the sequence
            tokens = torch.cat([tokens, next_tokens], dim=1)  # Shape: (batch, current_len + 1)

            # 11. Update the finished status: mark as True if EOS was generated
            finished = finished | (next_tokens.squeeze(1) == eos_idx)

            # 12. Check stopping condition: if all sequences are finished
            if finished.all():
                break

        # Convert token indices to lists of words
        captions = [vocab.encode_as_words(seq.tolist()) for seq in tokens]

        # Restructure attention: List[num_layers] of List[num_steps] of np.ndarray(batch, img_seq_len)
        num_layers = len(self.decoder_layers)
        restructured_attn = [[step_attn[layer_idx] for step_attn in all_attn_steps] for layer_idx in range(num_layers)]

        return captions, total_log_probs, restructured_attn

    def beam_search(self, features: torch.Tensor, vocab: Vocabulary, max_len: int, beam_size: int) -> tuple:
        """
        Generates captions using beam search.

        Maintains beam_size candidate sequences at each step, expanding them and selecting the top beam_size based on cumulative log probability,
        often normalized by length.

        :param features: Encoded image features. Shape (batch_size, img_seq_len, hidden_size). Note: This implementation processes
                                        batch_size=1 implicitly by looping. For batch > 1, features should be (1, img_seq_len, hidden_size).
        :param vocab: The vocabulary object.
        :param max_len: The maximum generation length.
        :param beam_size: The number of beams to maintain.
        :returns: Tuple containing:
                - captions: List containing the single best caption (as a list of words).
                - log_probs: Total log probability for the best caption. Shape (1,).
                - all_attn: List (one per step) of attention weights for the final chosen beam. Each element is the averaged attention (img_seq_len,).
        """

        batch_size = features.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)

        captions = []
        all_probs = []
        all_attn = []

        # Process each image in the batch separately
        for b in range(batch_size):
            feature = features[b].unsqueeze(0).repeat(beam_size, 1, 1)
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
                seq_tensor = torch.tensor(padded_seqs, device=feature.device)

                txt_emb = self.seq_embedding(seq_tensor)
                step_attn = []
                for layer in self.decoder_layers:
                    txt_emb = layer(feature[:seq_tensor.size(0)], txt_emb)
                    attn = layer.cross_attention.attention_scores
                    attn = attn.mean(dim=1).squeeze(1).squeeze(1)
                    step_attn.append(attn.detach().cpu().numpy())
                logits = self.output_layer(txt_emb[:, -1, :])
                step_log_probs = F.log_softmax(logits, dim=-1)  # (active_beams, vocab_size)
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

        return captions, torch.tensor(all_probs, device=features.device), all_attn

        # batch_size = features.size(0)
        #
        # sos_idx = vocab.str_to_idx(SOS)
        # eos_idx = vocab.str_to_idx(EOS)
        # pad_idx = vocab.str_to_idx(PAD)
        # device = features.device
        #
        # final_captions = []
        # final_log_probs = []
        # final_attentions = []
        #
        # for b_idx in range(batch_size):
        #     img_features = features[b_idx].unsqueeze(0)  # Shape: (1, img_seq_len, hidden_size)
        #
        #     # Start with SOS token, score 0. Structure: (cumulative_log_prob, sequence_indices, list_of_step_log_probs, list_of_step_attentions)
        #     beams = [(0.0, [sos_idx], [], [])]
        #     completed_beams = []
        #
        #     for t in range(max_len - 1):
        #         candidates = []
        #         active_beams_indices = []  # Indices of beams in beams that are not finished
        #         current_seqs = []  # Token sequences of active beams
        #
        #         # Prepare inputs for batch processing of active beams
        #         for i, (score, seq, log_probs_list, attn_list) in enumerate(beams):
        #             if seq[-1] == eos_idx:
        #                 # Beam finished in previous step, add to completed list if score is competitive
        #                 # Length penalty applied here for ranking completed beams
        #                 length_penalty = (len(seq) ** 0.5)  # Example penalty, adjust as needed
        #                 completed_beams.append((score / length_penalty, seq, log_probs_list, attn_list))
        #             else:
        #                 active_beams_indices.append(i)
        #                 current_seqs.append(seq)
        #
        #         if not active_beams_indices:  # All beams finished
        #             break
        #
        #         # Pad sequences for batch processing
        #         max_current_len = max(len(s) for s in current_seqs)
        #         padded_seqs = [s + [pad_idx] * (max_current_len - len(s)) for s in current_seqs]
        #         seq_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=device)  # (num_active_beams, max_current_len)
        #
        #         # Repeat image features for each active beam
        #         # Shape: (num_active_beams, img_seq_len, hidden_size)
        #         current_img_features = img_features.repeat(len(active_beams_indices), 1, 1)
        #
        #         # Run decoder for all active beams
        #         txt_emb = self.seq_embedding(seq_tensor)  # (num_active_beams, max_current_len, hidden_size)
        #         attn_weights_step = []
        #         for layer in self.decoder_layers:
        #             txt_emb = layer(current_img_features, txt_emb)
        #             attn = layer.cross_attention.attention_scores  # (num_active, heads, txt_len, img_len)
        #             avg_attn_last = attn[:, :, -1, :].mean(dim=1)  # (num_active, img_len)
        #             attn_weights_step.append(avg_attn_last.detach().cpu().numpy())
        #
        #         # Average attention over layers for this step: (num_active_beams, img_seq_len)
        #         avg_attn_step = np.mean(attn_weights_step, axis=0)
        #
        #         logits = self.output_layer(txt_emb[:, -1, :])  # (num_active_beams, vocab_size)
        #         log_probs_step = F.log_softmax(logits, dim=-1)  # (num_active_beams, vocab_size)
        #
        #         # Expand each active beam
        #         for i, beam_idx in enumerate(active_beams_indices):
        #             original_score, original_seq, original_log_probs, original_attns = beams[beam_idx]
        #             # Get top k next tokens and their log probs for this beam
        #             # Shape: (beam_size,)
        #             topk_log_probs, topk_indices = log_probs_step[i].topk(beam_size)
        #
        #             for j in range(beam_size):
        #                 next_token_idx = topk_indices[j].item()
        #                 token_log_prob = topk_log_probs[j].item()
        #
        #                 new_score = original_score + token_log_prob
        #                 new_seq = original_seq + [next_token_idx]
        #                 new_log_probs = original_log_probs + [token_log_prob]
        #                 new_attns = original_attns + [avg_attn_step[i]]  # Add attention for the *current* step
        #
        #                 candidates.append((new_score, new_seq, new_log_probs, new_attns))
        #
        #         # Prune candidates: Select top beam_size overall
        #         # Sort candidates by score (higher is better)
        #         candidates.sort(key=lambda x: x[0], reverse=True)
        #         beams = candidates[:beam_size]  # Keep only the best beams for the next step
        #
        #         # Check if all current best beams end in EOS
        #         if all(b[1][-1] == eos_idx for b in beams):
        #             completed_beams.extend([(b[0] / (len(b[1]) ** 0.5), b[1], b[2], b[3]) for b in beams])
        #             break  # Stop generation early if all top beams finished
        #
        #     # After loop finishes (max_len or all beams ended)
        #     # Add any remaining active beams to completed list (might not have reached EOS)
        #     for score, seq, log_probs_list, attn_list in beams:
        #         if seq[-1] != eos_idx:
        #             length_penalty = (len(seq) ** 0.5)
        #             completed_beams.append((score / length_penalty, seq, log_probs_list, attn_list))
        #
        #     # Select the best beam from completed beams based on normalized score
        #     if not completed_beams:
        #         # Handle cases where no beams completed (e.g., max_len too short)
        #         # Fallback to the best beam available, even if not finished
        #         best_beam = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5)) if beams else (0.0, [sos_idx, eos_idx], [], [])
        #     else:
        #         completed_beams.sort(key=lambda x: x[0], reverse=True)
        #         best_beam = completed_beams[0]  # (normalized_score, seq, log_probs_list, attn_list)
        #
        #     best_seq_indices = best_beam[1]
        #     best_caption_words = vocab.encode_as_words(best_seq_indices)
        #     total_log_prob = sum(best_beam[2])  # Sum of individual step log probs
        #     best_attentions = best_beam[3]  # List of np.array attentions per step
        #
        #     final_captions.append(best_caption_words)
        #     final_log_probs.append(total_log_prob)
        #     final_attentions.append(best_attentions)  # list (batch) of lists (steps) of arrays (img_seq_len)
        #
        # # Return results matching the batch size
        # return final_captions, torch.tensor(final_log_probs, device=device), final_attentions
