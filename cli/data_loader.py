"""Data loading utilities."""

import json
from pathlib import Path


def load_movies(filepath: str = "data/movies.json") -> list[dict]:
    """Load movies from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords(filepath: str = "data/stopwords.txt") -> set[str]:
    """Load stopwords from text file."""
    with open(filepath) as f:
        return set(f.read().splitlines())
