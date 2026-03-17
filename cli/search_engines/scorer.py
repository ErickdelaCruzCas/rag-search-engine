"""Relevance scoring functions for keyword search."""

import math
from collections import Counter
from cli.search_engines.inverted_index import InvertedIndex
from cli.constants import BM25_K1, BM25_B


def tf(index: InvertedIndex, doc_id: int, stemmed_term: str) -> int:
    """Term frequency: how many times stemmed_term appears in doc_id."""
    return index.term_frequencies.get(doc_id, Counter()).get(stemmed_term, 0)


def idf(index: InvertedIndex, stemmed_term: str) -> float:
    """Inverse document frequency using Laplace smoothing: log((N+1) / (df+1))."""
    N = len(index.docmap)
    df = len(index.index.get(stemmed_term, set()))
    return math.log((N + 1) / (df + 1))


def tfidf(index: InvertedIndex, doc_id: int, stemmed_term: str) -> float:
    """TF-IDF score: TF × IDF."""
    return tf(index, doc_id, stemmed_term) * idf(index, stemmed_term)


def bm25_tf(index: InvertedIndex, doc_id: int, stemmed_term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """BM25 TF with saturation and document length normalization.

    Formula: (tf * (k1+1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    """
    raw_tf = tf(index, doc_id, stemmed_term)
    if raw_tf == 0:
        return 0.0
    doc_len = index.doc_lengths.get(doc_id, 0)
    avg_doc_len = sum(index.doc_lengths.values()) / len(index.doc_lengths) if index.doc_lengths else 1.0
    length_norm = 1 - b + b * (doc_len / avg_doc_len)
    return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)


def bm25_idf(index: InvertedIndex, stemmed_term: str) -> float:
    """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
    N = len(index.docmap)
    df = len(index.index.get(stemmed_term, set()))
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def bm25(index: InvertedIndex, doc_id: int, stemmed_term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Full BM25 score for a term in a document: bm25_tf × bm25_idf."""
    return bm25_tf(index, doc_id, stemmed_term, k1, b) * bm25_idf(index, stemmed_term)
