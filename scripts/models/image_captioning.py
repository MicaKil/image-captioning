from typing import Optional

import torch
from torch import nn as nn

from constants import SOS, EOS, PAD
from dataset.vocabulary import Vocabulary


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

    def generate(self, images: torch.Tensor, vocab: Vocabulary, max_length: int, temperature: Optional[float], beam_size: int, device: torch.device,
                 no_grad: bool) -> list[str]:
        self.eval()
        images = images.to(device)
        if no_grad:
            with torch.no_grad():
                features = self.encoder(images)  # Encode the image (batch_size, embed_size)
                if beam_size > 1:
                    return self.beam_search(vocab, device, features, max_length, beam_size)
                else:
                    return self.temperature_sampling(vocab, device, features, max_length, temperature)

        features = self.encoder(images)  # Encode the image (batch_size, embed_size)
        if beam_size > 1:
            return self.beam_search(vocab, device, features, max_length, beam_size)
        else:
            return self.temperature_sampling(vocab, device, features, max_length, temperature)

    def temperature_sampling(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int,
                             temperature: Optional[float]) -> list[str]:
        """
        Generate a caption using temperature-based sampling if temperature is not None, otherwise use greedy search.

        :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
        :param device: Device to use
        :param features: Encoded image features
        :param max_length: Maximum caption length
        :param temperature: Temperature for sampling (None for greedy search)
        :return: Generated caption as a list of token indices
        """
        batch_size = features.size(0)
        captions = torch.full((batch_size, 1), vocab.str_to_idx(SOS), dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            outputs = self.decoder(features, captions)  # (batch_size, seq_len, vocab_size)
            logits = outputs[:, -1, :]  # Get last predicted token (batch_size, vocab_size)

            # Choose next token
            if temperature is None or temperature == 0.0:
                # Greedy search
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature sampling
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, 1)

            # Mask finished sequences
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, vocab.str_to_idx(PAD)), next_tokens)
            captions = torch.cat([captions, next_tokens], dim=1)

            # Update finished flags
            finished = finished | (next_tokens.squeeze() == vocab.str_to_idx(EOS))
            if finished.all():
                break
        return [vocab.encode_as_words(caption.tolist()) for caption in captions]

    def beam_search(self, vocab: Vocabulary, device: torch.device, features: torch.Tensor, max_length: int, beam_size: int) -> list[str]:
        """
        Generate a caption using beam search.

        :param vocab: Vocabulary object with str_to_idx and idx_to_str mappings
        :param device: Device to use
        :param features: Encoded image features
        :param max_length: Maximum caption length
        :param beam_size: Beam size for beam search
        :return: Generated caption as a list of token indices
        """
        batch_size = features.size(0)
        sos_idx = vocab.str_to_idx(SOS)
        eos_idx = vocab.str_to_idx(EOS)
        captions = []

        for i in range(batch_size):
            # The image features are replicated beam_size times (one copy per beam/hypothesis)
            img_features = features[i].unsqueeze(0)  # (beam_size, embed_size)
            beams = [(0.0, [sos_idx])]

            for _ in range(max_length):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq))
                        continue

                    tokens = torch.tensor([seq], device=device)
                    outputs = self.decoder(img_features, tokens)  # (1, seq_len, vocab_size)
                    logits = outputs[:, -1, :]  # (1, vocab_size)

                    log_probs = torch.log_softmax(logits, dim=-1)
                    top_probs, top_indices = log_probs.topk(beam_size)

                    for j in range(beam_size):
                        candidates.append((
                            score + top_probs[0, j].item(),
                            seq + [top_indices[0, j].item()]
                        ))

                # Keep top-k candidates
                candidates.sort(reverse=True, key=lambda x: x[0] / (len(x[1]) ** 0.5))
                beams = candidates[:beam_size]

                # Check if all beams are complete
                if all(seq[-1] == eos_idx for _, seq in beams):
                    break

            # Get best sequence
            best_seq = max(beams, key=lambda x: x[0] / (len(x[1]) ** 0.5))[1]
            captions.append(vocab.encode_as_words(best_seq))

        return captions

    def calc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError("calculate_loss method must be implemented in the subclass")