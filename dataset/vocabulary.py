from collections import Counter
from typing import Optional

import sentencepiece as spm
from nltk import word_tokenize, TreebankWordDetokenizer

from config.config import logger
from constants import PAD, SOS, EOS, UNK


class Vocabulary:
    """
    Vocabulary class that builds a vocabulary from a list of texts.
    Supports both word-based tokenization and SentencePiece-based subword tokenization.
    """

    def __init__(self, tokenizer: str, freq_threshold: Optional[int] = None, text: list[str] = None, sp_model_path: str = None):
        """
        Initialize the Vocabulary object.

        :param tokenizer: Type of tokenizer to use ("word" or "sp-bpe").
        :param freq_threshold: Minimum frequency of a word to be included in the vocabulary (used for "word" tokenizer).
        :param text: List of texts to build the vocabulary from (used for "word" tokenizer).
        :param sp_model_path: Path to the SentencePiece model file (used for "sp-bpe" tokenizer).
        """
        self.tokenizer = tokenizer
        self.freq_threshold = freq_threshold
        self.stoi_dict = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}  # String-to-index mapping.
        self.itos_dict = {0: PAD, 1: SOS, 2: EOS, 3: UNK}  # Index-to-string mapping.
        self.word_counts = Counter()  # Counter to store word frequencies.
        self.sp = None  # SentencePiece processor.

        if tokenizer == "word" and text is not None:
            self.build_vocab(text)  # Build vocabulary for word-based tokenization.
        if tokenizer == "sp-bpe" and sp_model_path is not None:
            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)  # Load SentencePiece model.
            self.sp = sp

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize a given text based on the tokenizer type.

        :param text: Input text to tokenize.
        :return: List of tokens.
        """
        match self.tokenizer:
            case "word":
                return word_tokenize(text.lower())  # Tokenize using NLTK word tokenizer.
            case "sp-bpe":
                return self.sp.encode_as_pieces(text)  # Tokenize using SentencePiece.
            case _:
                raise ValueError("Invalid tokenizer type.")

    def build_vocab(self, text_list: list[str]):
        """
        Build the vocabulary from a list of texts for word-based tokenization.

        :param text_list: List of texts to build the vocabulary from.
        :return: None
        """
        logger.info("Building vocabulary.")
        for text in text_list:
            self.word_counts.update(self.tokenize(text))  # Update word frequencies.

        idx = 4  # Start indexing from 4 (0-3 are reserved for special tokens).
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:  # Include words that meet the frequency threshold.
                self.stoi_dict[word] = idx
                self.itos_dict[idx] = word
                idx += 1

        self.word_counts[SOS] = len(text_list)  # Add SOS token frequency.
        self.word_counts[EOS] = len(text_list)  # Add EOS token frequency.

    def encode_as_ids(self, text: str) -> list[int]:
        """
        Convert a text to a list of token indices.

        :param text: Input text to encode.
        :return: List of token indices.
        """
        match self.tokenizer:
            case "word":
                return [self.str_to_idx(word) for word in self.tokenize(text)]  # Encode using word-based vocabulary.
            case "sp-bpe":
                return self.sp.encode_as_ids(text)  # Encode using SentencePiece.
            case _:
                raise ValueError("Invalid tokenizer type.")

    def str_to_idx(self, word: str) -> int:
        """
        Convert a word to its corresponding index.

        :param word: Word to convert.
        :return: Index of the word, or the index of UNK if the word is not in the vocabulary.
        """
        match self.tokenizer:
            case "word":
                return self.stoi_dict.get(word, self.stoi_dict[UNK])  # Return UNK index if word is not found.
            case "sp-bpe":
                return self.sp.piece_to_id(word)  # Get index from SentencePiece.
            case _:
                raise ValueError("Invalid tokenizer type.")

    def encode_as_words(self, idxs: list[int]) -> str:
        """
        Convert a list of token indices back to text.

        :param idxs: List of token indices.
        :return: Decoded text.
        """
        match self.tokenizer:
            case "word":
                return TreebankWordDetokenizer().detokenize(
                    [self.idx_to_str(int(idx)) for idx in idxs if int(idx) not in [0, 1, 2]]
                )  # Detokenize while ignoring special tokens.
            case "sp-bpe":
                return self.sp.decode_ids(idxs)  # Decode using SentencePiece.
            case _:
                raise ValueError("Invalid tokenizer type.")

    def idx_to_str(self, idx: int) -> str:
        """
        Convert an index to its corresponding word.

        :param idx: Index to convert.
        :return: Word corresponding to the index, or UNK if the index is not in the vocabulary.
        """
        match self.tokenizer:
            case "word":
                return self.itos_dict.get(idx, UNK)  # Return UNK if index is not found.
            case "sp-bpe":
                return self.sp.id_to_piece(idx)  # Get word from SentencePiece.
            case _:
                raise ValueError("Invalid tokenizer type.")

    def get_state_dict(self) -> dict:
        """
        Get the state dictionary of the vocabulary (only for word-based tokenization).

        :return: State dictionary containing vocabulary mappings and word counts.
        """
        if self.tokenizer == "word":
            return {
                "str_to_idx": self.stoi_dict,
                "idx_to_str": self.itos_dict,
                "word_counts": self.word_counts,
                "freq_threshold": self.freq_threshold
            }
        raise ValueError("Invalid tokenizer type.")

    def load_dict(self, state_dict: dict):
        """
        Load the state dictionary into the vocabulary (only for word-based tokenization).

        :param state_dict: State dictionary to load.
        :return: None
        """
        if self.tokenizer != "word":
            raise ValueError("Invalid tokenizer type.")

        self.stoi_dict = state_dict["str_to_idx"]
        self.itos_dict = state_dict["idx_to_str"]
        self.word_counts = state_dict["word_counts"]
        self.freq_threshold = state_dict["freq_threshold"]

    def __str__(self):
        """
        Return a string representation of the vocabulary.

        :return: String representation of the vocabulary with word frequencies.
        """
        return str({word: self.word_counts[word] for _, word in self.itos_dict.items()})

    def __len__(self):
        """
        Return the size of the vocabulary.

        :return: Size of the vocabulary.
        """
        match self.tokenizer:
            case "word":
                return len(self.stoi_dict)  # Return size of word-based vocabulary.
            case "sp-bpe":
                return self.sp.get_piece_size()  # Return size of SentencePiece vocabulary.
            case _:
                raise ValueError("Invalid tokenizer type.")
