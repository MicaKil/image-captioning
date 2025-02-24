from typing import Optional

import torch
from torch import nn as nn

from constants import SOS, EOS
from scripts.dataset.vocabulary import Vocabulary


class ImageCaptioner(nn.Module):
    """
    Image captioning model that combines an Encoder and Decoder.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Constructor for the ImageCaptioning class
        :param encoder: Encoder model
        :param decoder: Decoder model
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ImageCaptioning model

        :param images: Input image tensors
        :param captions: Caption word indices
        :return: Predicted word indices
        """
        features = self.encoder(images)  # Shape: (batch_size, embed_size)
        outputs = self.decoder(features, captions)  # Shape: (batch_size, max_caption_length + 1, vocab_size)
        return outputs

    def generate(self, image: torch.Tensor, vocab: Vocabulary, max_length: int = 30, device: torch.device = torch.device("cpu"),
                 temperature: Optional[float] = None, beam_size: int = 1) -> str:
        self.eval()
        with torch.no_grad():
            image = image.to(device)
            features = self.encoder(image)  # Encode the image (1, embed_size)
            if beam_size > 1:
                return self.beam_search(vocab, device, features, max_length, beam_size)
            else:
                return self.temperature_sampling(vocab, device, features, max_length, temperature)

    def temperature_sampling(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int,
                             temperature: Optional[float]) -> str:
        """
        Generate a caption using temperature-based sampling if temperature is not None, otherwise use greedy search.

        :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
        :param device: Device to use
        :param features: Encoded image features
        :param max_length: Maximum caption length
        :param temperature: Temperature for sampling (None for greedy search)
        :return: Generated caption as a list of token indices
        """
        caption = [vocab.to_idx(SOS)]  # Initialize caption with start token
        for _ in range(max_length):
            caption_tensor = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
            outputs = self.decoder(features, caption_tensor)  # Get predictions (batch_size, seq_len+1, vocab_size)
            logits = outputs[:, -1, :]  # Get last predicted token (batch_size, vocab_size)

            # Choose next token
            if temperature is None or temperature == 0.0:
                # Greedy search
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                # Temperature sampling
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Stop if we predict the end token
            if next_token == vocab.to_idx(EOS):
                break

            caption.append(next_token)
        return vocab.to_text(caption)

    def beam_search(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int, beam_size: int) -> str:
        """
        Generate a caption using beam search.

        :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
        :param device: Device to use
        :param features: Encoded image features
        :param max_length: Maximum caption length
        :param beam_size: Beam size for beam search
        :return: Generated caption as a list of token indices
        """
        sos_idx = vocab.to_idx(SOS)
        eos_idx = vocab.to_idx(EOS)
        vocab_size = len(vocab)
        # The image features are replicated beam_size times (one copy per beam/hypothesis)
        features = features.expand(beam_size, -1)  # (beam_size, embed_size)
        # Initialize beam
        beam_scores = torch.zeros(beam_size).to(device)  # log probabilities
        beam_sequences = torch.tensor([[sos_idx]] * beam_size, dtype=torch.long).to(device)  # all beams start with SOS
        completed_sequences = []
        completed_scores = []
        for _ in range(max_length):
            # Use the decoder to predict logits for the next token:
            outputs = self.decoder(features, beam_sequences)  # (beam_size, seq_len, vocab_size)
            # Extract the last token's logits and compute log probabilities
            logits = outputs[:, -1, :]  # (beam_size, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Combine scores
            scores = log_probs + beam_scores.unsqueeze(1)  # (beam_size, vocab_size)
            scores_flat = scores.view(-1)  # (beam_size * vocab_size)

            # Get top candidates
            top_scores, top_indices = torch.topk(scores_flat, k=beam_size)
            beam_indices = top_indices // vocab_size  # Which beam does this come from?
            token_indices = top_indices % vocab_size  # Which token does this predict?

            # Update sequences
            beam_sequences = torch.cat([beam_sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)
            beam_scores = top_scores

            # Check for completed sequences
            eos_mask = token_indices == eos_idx
            if eos_mask.any():
                # If a beam predicts the EOS token, move it to the completed_sequences list and remove it from active beams
                completed_sequences.extend(beam_sequences[eos_mask].tolist())
                completed_scores.extend(beam_scores[eos_mask].tolist())

                # Remove completed sequences from beam
                keep_mask = ~eos_mask
                beam_sequences = beam_sequences[keep_mask]
                beam_scores = beam_scores[keep_mask]
                features = features[keep_mask]

                if beam_sequences.size(0) == 0:
                    break  # All sequences completed
        # Select best sequence
        if completed_sequences:
            best_idx = torch.argmax(torch.tensor(completed_scores)).item()
            best_sequence = completed_sequences[best_idx]
        else:
            best_idx = torch.argmax(beam_scores).item()
            best_sequence = beam_sequences[best_idx].tolist()
        return vocab.to_text(best_sequence)

    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError("calculate_loss method must be implemented in the subclass")
