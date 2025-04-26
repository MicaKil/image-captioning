import torch
import torch.nn as nn

import models.image_captioner as image_captioner
from constants import PAD, SOS, UNK
from dataset.vocabulary import Vocabulary
from models.encoders import intermediate as inter_encoder


class Decoder(nn.Module):
    """
    Decoder class that uses an LSTM to generate captions for images. Slightly more advanced than the basic decoder.
    """

    def __init__(self, embed_dim: int, hidden_size: int, vocab: Vocabulary, dropout: float, num_layers: int, padding_idx: int) -> None:
        """
        Constructor for the Decoder class.

        :param embed_dim: Size of the word embeddings.
        :param hidden_size: Size of the hidden state of the LSTM.
        :param vocab: Vocabulary object containing mappings for tokens.
        :param dropout: Dropout probability for regularization.
        :param num_layers: Number of layers in the LSTM.
        :param padding_idx: Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)
        self.banned_indices = [vocab.str_to_idx(token) for token in [PAD, SOS, UNK]]

        # Project image features to initialize hidden and cell states
        self.init_h = nn.Linear(embed_dim, num_layers * hidden_size)
        self.init_c = nn.Linear(embed_dim, num_layers * hidden_size)

        self.embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.linear.bias.data.zero_()  # Initialize bias to zeros.
        self.linear.bias.data[self.banned_indices] = -1e9  # Set banned tokens' bias to -1e9 initially.

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the decoder.

        :param features: Image feature vectors of shape (batch_size, embed_dim).
        :param captions: Caption word indices of shape (batch_size, max_caption_length).
        :return: Predicted word indices of shape (batch_size, max_caption_length + 1, vocab_size).
        """
        # Initialize hidden and cell states using image features
        batch_size = features.size(0)
        h = self.init_h(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c = self.init_c(features).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()

        # Embed and pack sequences
        embeddings = self.embed(captions)  # Shape: (batch_size, max_caption_length, embed_dim).
        embeddings = self.dropout(embeddings)

        # LSTM forward
        lstm_out, _ = self.lstm(embeddings, (h, c))
        lstm_out = self.layer_norm(lstm_out)

        outputs = self.linear(lstm_out)  # Shape: (batch_size, max_caption_length + 1, vocab_size).
        return outputs


class ImageCaptioner(image_captioner.ImageCaptioner):
    """
    ImageCaptioner class that combines an encoder and a decoder to generate captions for images.
    """

    def __init__(self, encoder: inter_encoder.Encoder, decoder: Decoder) -> None:
        """
        Initialize the ImageCaptioner class.

        :param encoder: Encoder object to extract image features.
        :param decoder: Decoder object to generate captions from features.
        """
        super().__init__(encoder, decoder)

    def calc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Calculate the loss for the given outputs and targets.

        :param outputs: Predicted word indices of shape (batch_size, padded_length, vocab_size).
        :param targets: Target word indices of shape (batch_size, padded_length).
        :param criterion: Loss function to compute the loss.
        :return: Loss value as a scalar tensor.
        """
        return criterion(
            outputs.reshape(-1, outputs.size(-1)),  # Reshape outputs to (batch_size * (seq_len - 1), vocab_size).
            targets.reshape(-1)  # Reshape targets to (batch_size * (seq_len - 1)).
        )
