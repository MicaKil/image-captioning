from collections import Counter

from nltk import word_tokenize, TreebankWordDetokenizer

from configs.config import logger
from constants import PAD, SOS, EOS, UNK


class Vocabulary:
	"""
	Vocabulary class that builds a vocabulary from a list of texts.
	"""

	def __init__(self, freq_threshold: int, text_list: list[str] = None):
		"""
		:param freq_threshold: Minimum frequency of a word to be included in the vocabulary
		:param text_list: List of texts to build the vocabulary from
		"""
		self.freq_threshold = freq_threshold
		self.str_to_idx = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
		self.idx_to_str = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
		self.word_counts = Counter()

		if text_list is not None:
			self.build_vocab(text_list)

	@staticmethod
	def tokenize_eng(text: str) -> list[str]:
		"""
		Tokenize an English text.
		:param text: English text
		:return: List of word tokens
		"""
		return word_tokenize(text.lower())

	def build_vocab(self, text_list: list[str]):
		"""
		Build the vocabulary from a list of texts.
		:param text_list: List of texts
		:return: None
		"""
		logger.info("Building vocabulary.")
		for text in text_list:
			self.word_counts.update(self.tokenize_eng(text))

		idx = 4
		for word, count in self.word_counts.items():
			if count >= self.freq_threshold:
				self.str_to_idx[word] = idx
				self.idx_to_str[idx] = word
				idx += 1

	def to_idx_list(self, text: str) -> list[int]:
		"""
		Convert a text to a list of word indices.
		:param text: Input text
		:return: A list of word indices
		"""
		return [self.to_idx(word) for word in self.tokenize_eng(text)]

	def to_idx(self, word: str) -> int:
		"""
		Convert a string to its index.
		:param word: Word to convert
		:return: Index of the word or the index of UNK if the word is not in the vocabulary
		"""
		return self.str_to_idx.get(word, self.str_to_idx[UNK])

	def to_text(self, idxs: list[int]) -> str:
		"""
		Convert a list of indices to text.
		:param idxs: List of indices to convert
		:return: Text corresponding to the indices
		"""
		return TreebankWordDetokenizer().detokenize(
			[self.to_word(int(idx)) for idx in idxs if int(idx) not in [0, 1, 2]])

	def to_word(self, idx: int) -> str:
		"""
		Convert an index to its word.
		:param idx: Index to convert
		:return: Word corresponding to the index or UNK if the index is not in the vocabulary
		"""
		return self.idx_to_str.get(idx, UNK)

	def __str__(self):
		"""
		Return a string representation of the vocabulary.
		:return: String representation of the vocabulary
		"""
		return str({word: self.word_counts[word] for _, word in self.idx_to_str.items()})

	def __len__(self):
		"""
		Return the size of the vocabulary.
		:return: Size of the vocabulary
		"""
		return len(self.str_to_idx)
