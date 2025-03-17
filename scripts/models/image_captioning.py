from typing import Optional

import torch
from torch import nn as nn

from constants import SOS, EOS, PAD
from scripts.dataset.dataloader import CaptionLoader
from scripts.dataset.vocabulary import Vocabulary
from scripts.test import test
from scripts.train import train


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
                 temperature: Optional[float] = None, beam_size: int = 1) -> list[str]:
        self.eval()
        with torch.no_grad():
            image = image.to(device)
            features = self.encoder(image)  # Encode the image (batch_size, embed_size)
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
            img_features = features[i].expand(beam_size, -1)  # (beam_size, embed_size)
            beams = [(0.0, [sos_idx])]

            for _ in range(max_length):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq))
                        continue

                    tokens = torch.tensor(seq, device=device).unsqueeze(0)
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

    def train_model(self, train_loader: CaptionLoader, val_loader: CaptionLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim,
                    scheduler: torch.optim.lr_scheduler, checkpoint_dir: str, use_wandb: bool, run_config: dict, resume_checkpoint) -> tuple:
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
