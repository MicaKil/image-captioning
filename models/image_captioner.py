from typing import Optional

import torch
from torch import nn as nn

from constants import SOS, EOS, PAD
from dataset.vocabulary import Vocabulary
from models import basic as basic_decoder, intermediate as inter_decoder
from models.encoders import basic as basic_encoder, intermediate as inter_encoder


class ImageCaptioner(nn.Module):
    """
    Common base class for the ImageCaptioner. This class is inherited by different image captioning implementations.
    """

    def __init__(self, encoder: basic_encoder.Encoder | inter_encoder.Encoder, decoder: basic_decoder.Decoder | inter_decoder.Decoder) -> None:
        """
        Constructor for the ImageCaptioner class.

        :param encoder: Encoder model to extract image features.
        :param decoder: Decoder model to generate captions from features.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ImageCaptioner model.

        :param images: Input image tensors of shape (batch_size, 3, height, width).
        :param captions: Caption word indices of shape (batch_size, seq_len).
        :return: Predicted word indices of shape (batch_size, seq_len + 1, vocab_size).
        """
        features = self.encoder(images)  # Extract image features. Shape: (batch_size, embed_size).
        outputs = self.decoder(features, captions)  # Generate captions. Shape: (batch_size, seq_len + 1, vocab_size).
        return outputs

    def generate(self, images: torch.Tensor, vocab: Vocabulary, max_len: int, temp: Optional[float], beam_size: int, device: torch.device,
                 no_grad: bool, return_attn=None) -> tuple:
        """
        Generate captions for the given images using either temperature sampling or beam search.

        :param images: Input image tensors of shape (batch_size, 3, height, width).
        :param vocab: Vocabulary object with mappings for tokens.
        :param max_len: Maximum length of the generated captions.
        :param temp: Temperature for sampling (None for greedy search).
        :param beam_size: Beam size for beam search.
        :param device: Device to use for computation.
        :param no_grad: Whether to disable gradient computation.
        :param return_attn: Placeholder for attention return (not implemented).
        :return: Tuple containing generated captions and attention weights (if applicable).
        """
        self.eval()
        images = images.to(device)
        if no_grad:
            with torch.no_grad():
                features = self.encoder(images)  # Encode the image. Shape: (batch_size, embed_size).
                if beam_size > 1:
                    return self.beam_search(vocab, device, features, max_len, beam_size), None
                else:
                    return self.temperature_sampling(vocab, device, features, max_len, temp), None

        features = self.encoder(images)  # Encode the image. Shape: (batch_size, embed_size).
        if beam_size > 1:
            return self.beam_search(vocab, device, features, max_len, beam_size), None
        else:
            return self.temperature_sampling(vocab, device, features, max_len, temp), None

    def temperature_sampling(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_len: int, temp: Optional[float]) -> list[str]:
        """
        Generate a caption using temperature-based sampling if temperature is not None, otherwise use greedy search.

        :param vocab: Vocabulary object with mappings for tokens.
        :param device: Device to use for computation.
        :param features: Encoded image features of shape (batch_size, embed_size).
        :param max_len: Maximum length of the generated captions.
        :param temp: Temperature for sampling (None for greedy search).
        :return: Generated captions as a list of strings.
        """
        batch_size = features.size(0)
        captions = torch.full((batch_size, 1), vocab.str_to_idx(SOS), dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            outputs = self.decoder(features, captions)  # Shape: (batch_size, seq_len, vocab_size).
            logits = outputs[:, -1, :]  # Get logits for the last predicted token. Shape: (batch_size, vocab_size).

            # Choose the next token
            if temp is None or temp == 0.0:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy search.
            else:
                probs = torch.softmax(logits / temp, dim=-1)  # Apply temperature scaling.
                next_tokens = torch.multinomial(probs, 1)  # Sample from the probability distribution.

            # Mask finished sequences
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, vocab.str_to_idx(PAD)), next_tokens)
            captions = torch.cat([captions, next_tokens], dim=1)

            # Update finished flags
            finished = finished | (next_tokens.squeeze() == vocab.str_to_idx(EOS))
            if finished.all():
                break
        return [vocab.encode_as_words(caption.tolist()) for caption in captions]

    def beam_search(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_len: int, beam_size: int) -> list[str]:
        """
        Generate a caption using beam search.

        :param vocab: Vocabulary object with mappings for tokens.
        :param device: Device to use for computation.
        :param features: Encoded image features of shape (batch_size, embed_size).
        :param max_len: Maximum length of the generated captions.
        :param beam_size: Beam size for beam search.
        :return: Generated captions as a list of strings.
        """
        batch_size = features.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)
        captions = []

        for i in range(batch_size):
            img_features = features[i].unsqueeze(0)  # Replicate image features for beam search. Shape: (beam_size, embed_size).
            beams = [(0.0, [sos_idx])]  # Initialize beams with the start token.

            for _ in range(max_len):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq))  # Keep completed sequences.
                        continue

                    tokens = torch.tensor([seq], device=device)
                    outputs = self.decoder(img_features, tokens)  # Shape: (1, seq_len, vocab_size).
                    logits = outputs[:, -1, :]  # Get logits for the last token. Shape: (1, vocab_size).

                    log_probs = torch.log_softmax(logits, dim=-1)
                    top_probs, top_indices = log_probs.topk(beam_size)  # Get top-k probabilities and indices.

                    for j in range(beam_size):
                        candidates.append((
                            score + top_probs[0, j].item(),
                            seq + [top_indices[0, j].item()]
                        ))

                # Keep top-k candidates
                candidates.sort(reverse=True, key=lambda x: x[0] / (len(x[1]) ** 0.5))  # Length normalization.
                beams = candidates[:beam_size]

                # Check if all beams are complete
                if all(seq[-1] == eos_idx for _, seq in beams):
                    break

            # Get the best sequence
            best_seq = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5))[1]
            captions.append(vocab.encode_as_words(best_seq))

        return captions

    def calc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Calculate the loss for the given outputs and targets.

        :param outputs: Predicted word indices of shape (batch_size, seq_len, vocab_size).
        :param targets: Target word indices of shape (batch_size, seq_len).
        :param criterion: Loss function to compute the loss.
        :return: Loss value as a scalar tensor.
        """
        raise NotImplementedError("calculate_loss method must be implemented in the subclass")
