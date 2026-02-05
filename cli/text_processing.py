"""Text processing utilities for search operations."""

import string
from nltk.stem import PorterStemmer


def normalize_text(text: str) -> str:
    """Convert text to lowercase and remove punctuation."""
    lowercase_text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    return lowercase_text.translate(translator)


def tokenize(text: str) -> list[str]:
    """Split text into tokens (words)."""
    return text.split()


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Filter out stopwords from token list."""
    return [token for token in tokens if token not in stopwords]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
