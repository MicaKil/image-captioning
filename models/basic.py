import torch
import torch.nn as nn

import models.image_captioner as image_captioner
from models.encoders.basic import Encoder


class Decoder(nn.Module):
    """
    Decoder class that uses an LSTM to generate captions for images.
    """

    def __init__(self, embed_dim: int, hidden_size: int, vocab_size: int, dropout: float, num_layers: int, pad_idx: int) -> None:
        """
        Initialize the Decoder class.

        :param embed_dim: Size of the word embeddings.
        :param hidden_size: Size of the hidden state of the LSTM.
        :param vocab_size: Size of the vocabulary.
        :param dropout: Dropout probability.
        :param num_layers: Number of layers in the LSTM.
        :param pad_idx: Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)  # Embedding layer for word indices.
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Apply dropout only if num_layers > 1.
        )
        self.linear = nn.Linear(hidden_size, vocab_size)  # Linear layer to project LSTM outputs to vocabulary size.
        self.linear.bias.data.zero_()  # Initialize bias to zeros.
        self.linear.bias.data[self.banned_indices] = -1e9  # Set banned tokens' bias to -1e9 initially.
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization.

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the decoder.

        :param features: Image feature vectors of shape (batch_size, embed_dim).
        :param captions: Caption word indices of shape (batch_size, padded_seq_len).
        :return: Predicted word indices of shape (batch_size, padded_seq_len + 1, vocab_size).
        """
        embeddings = self.embed(captions)  # Embed the captions. Shape: (batch_size, padded_seq_len, embed_dim).
        embeddings = self.dropout(embeddings)  # Apply dropout to embeddings.
        features = features.unsqueeze(1)  # Add a time dimension to features. Shape: (batch_size, 1, embed_dim).
        combined = torch.cat([features, embeddings],
                             dim=1)  # Concatenate features and embeddings along time. Shape: (batch_size, padded_seq_len + 1, embed_dim).
        lstm_out, _ = self.lstm(combined)  # Pass through LSTM. Shape: (batch_size, padded_seq_len + 1, hidden_size).
        outputs = self.linear(lstm_out)  # Project LSTM outputs to vocabulary size. Shape: (batch_size, padded_seq_len + 1, vocab_size).
        return outputs


class ImageCaptioner(image_captioner.ImageCaptioner):
    """
    ImageCaptioner class that combines an encoder and a decoder to generate captions for images.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
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
            outputs[:, 1:].reshape(-1, outputs.size(-1)),  # Reshape outputs to (batch_size * (seq_len - 1), vocab_size).
            targets.reshape(-1)  # Reshape targets to (batch_size * (seq_len - 1)).
        )
