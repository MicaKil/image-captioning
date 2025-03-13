from collections import Counter

import sentencepiece as spm
from nltk import word_tokenize, TreebankWordDetokenizer

from config.config import logger
from constants import PAD, SOS, EOS, UNK


class Vocabulary:
    """
    Vocabulary class that builds a vocabulary from a list of texts.
    """

    def __init__(self, tokenizer: str, freq_threshold: int, text: list[str] = None, sp_model_path: str = None):
        """
        :param tokenizer:
        :param freq_threshold: Minimum frequency of a word to be included in the vocabulary
        :param text: List of texts to build the vocabulary from
        :param sp_model_path:
        """
        self.tokenizer = tokenizer
        self.freq_threshold = freq_threshold
        self.stoi_dict = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}  # string to index
        self.itos_dict = {0: PAD, 1: SOS, 2: EOS, 3: UNK}  # index to string
        self.word_counts = Counter()

        if tokenizer == "word" and text is not None:
            self.build_vocab(text)
        if tokenizer == "sp-bpe" and sp_model_path is not None:
            sp = spm.SentencePieceProcessor()
            self.sp_model = sp.load(sp_model_path)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize an English text.
        :param text: English text
        :return: List of word tokens
        """
        match self.tokenizer:
            case "word":
                return word_tokenize(text.lower())
            case "sp-bpe":
                return self.sp_model.encode_as_pieces(text)
            case _:
                raise ValueError("Invalid tokenizer type.")

    def build_vocab(self, text_list: list[str]):
        """
        Build the vocabulary from a list of texts.
        :param text_list: List of texts
        :return: None
        """
        logger.info("Building vocabulary.")
        for text in text_list:
            self.word_counts.update(self.tokenize(text))

        idx = 4
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                self.stoi_dict[word] = idx
                self.itos_dict[idx] = word
                idx += 1

        self.word_counts[SOS] = len(text_list)
        self.word_counts[EOS] = len(text_list)

    def encode_as_ids(self, text: str) -> list[int]:
        """
        Convert a text to a list of word indices.
        :param text: Input text
        :return: A list of word indices
        """
        match self.tokenizer:
            case "word":
                return [self.str_to_idx(word) for word in self.tokenize(text)]
            case "sp-bpe":
                return self.sp_model.encode_as_ids(text)
            case _:
                raise ValueError("Invalid tokenizer type.")

    def str_to_idx(self, word: str) -> int:
        """
        Convert a string to its index.
        :param word: Word to convert
        :return: Index of the word or the index of UNK if the word is not in the vocabulary
        """
        match self.tokenizer:
            case "word":
                return self.stoi_dict.get(word, self.stoi_dict[UNK])
            case "sp-bpe":
                return self.sp_model.piece_to_id(word)
            case _:
                raise ValueError("Invalid tokenizer type.")

    def encode_as_words(self, idxs: list[int]) -> str:
        """
        Convert a list of indices to text.
        :param idxs: List of indices to convert
        :return: Text corresponding to the indices
        """
        match self.tokenizer:
            case "word":
                return TreebankWordDetokenizer().detokenize([self.idx_to_str(int(idx)) for idx in idxs if int(idx) not in [0, 1, 2]])
            case "sp-bpe":
                return self.sp_model.decode_ids(idxs)
            case _:
                raise ValueError("Invalid tokenizer type.")

    def idx_to_str(self, idx: int) -> str:
        """
        Convert an index to its word.
        :param idx: Index to convert
        :return: Word corresponding to the index or UNK if the index is not in the vocabulary
        """
        match self.tokenizer:
            case "word":
                return self.itos_dict.get(idx, UNK)
            case "sp-bpe":
                return self.sp_model.id_to_piece(idx)
            case _:
                raise ValueError("Invalid tokenizer type.")

    def get_state_dict(self) -> dict:
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
        Load the state dictionary.
        :param state_dict: State dictionary
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
        :return: String representation of the vocabulary
        """
        return str({word: self.word_counts[word] for _, word in self.itos_dict.items()})

    def __len__(self):
        """
        Return the size of the vocabulary.
        :return: Size of the vocabulary
        """
        return len(self.stoi_dict)
