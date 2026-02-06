"""Tokenizer for converting text into processable tokens."""

import string
from nltk.stem import PorterStemmer


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Filter out stopwords from token list."""
    return [token for token in tokens if token not in stopwords]


class Tokenizer:
    """Handles tokenization with consistent stemming across queries and documents."""

    def __init__(self, stopwords: set[str] | None = None):
        """
        Initialize tokenizer.

        Args:
            stopwords: Optional set of stopwords to remove from tokens
        """
        self._stemmer = PorterStemmer()
        self._stopwords = stopwords if stopwords is not None else set()

    def _normalize(self, text: str) -> str:
        """
        Normalize text to lowercase and remove punctuation.

        Args:
            text: Raw text string

        Returns:
            Normalized text
        """
        lowercase_text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        return lowercase_text.translate(translator)

    def _split_into_words(self, text: str) -> list[str]:
        """
        Split normalized text into individual word tokens.

        Args:
            text: Normalized text string

        Returns:
            List of word tokens
        """
        return text.split()

    def _stem(self, tokens: list[str]) -> list[str]:
        """
        Apply Porter stemming algorithm to tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of stemmed tokens
        """
        return [self._stemmer.stem(token) for token in tokens]

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text with full processing pipeline.

        Pipeline: normalize → split → remove stopwords → stem

        This is the main tokenization method used consistently for:
        - Search queries
        - Document indexing
        - Title processing

        Args:
            text: Raw text string

        Returns:
            List of stemmed tokens with stopwords removed
        """
        normalized = self._normalize(text)
        tokens = self._split_into_words(normalized)
        tokens = remove_stopwords(tokens, self._stopwords)
        return self._stem(tokens)
