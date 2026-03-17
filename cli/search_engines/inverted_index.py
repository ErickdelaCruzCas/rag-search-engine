import os
import math
import pickle
from typing import Dict, Set, List, Any, Tuple
from cli.tokenizer import Tokenizer
from cli.constants import CACHE_DIR, BM25_K1, BM25_B
from collections import Counter
from tqdm import tqdm


class InvertedIndex:
    """
    A simple inverted index that maps tokens to document IDs
    and stores the full document objects in a separate map.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        """
        Initialize an empty inverted index.

        Args:
            tokenizer: Tokenizer instance for processing text.

        Attributes:
            index: Dictionary mapping tokens (str) to sets of document IDs (int).
            docmap: Dictionary mapping document IDs (int) to full document objects.
            term_frequencies: Dictionary mapping document IDs (int) to Counter objects
                            that track term frequencies within each document.
        """
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}
        self.term_frequencies: Dict[int, Counter] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.tokenizer = tokenizer

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Add a document to the inverted index.

        The text is tokenized and each token is associated
        with the provided document ID. Also tracks term frequencies
        for each document.

        Args:
            doc_id: Unique identifier of the document.
            text: Text content used for indexing.
        """
        # Doc length = raw word count after normalize+split, before stopword removal and stemming.
        # This matches the BM25 convention: document length is the number of word occurrences
        # in the original text, not the reduced token set.
        normalized = self.tokenizer._normalize(text)
        raw_words = self.tokenizer._split_into_words(normalized)
        self.doc_lengths[doc_id] = len(raw_words)

        tokens = self.tokenizer.tokenize(text)

        # Initialize Counter for this document if it doesn't exist
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            # Add to inverted index
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            # Update term frequency for this document
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> List[int]:
        """
        Retrieve document IDs containing a given term.

        Args:
            term: A single search term (will be tokenized and stemmed).

        Returns:
            A list of document IDs sorted in ascending order.
            Returns an empty list if the term is not found.
        """
        tokens = self.tokenizer.tokenize(term)
        if not tokens:
            return []

        # Use the first token (should only be one for single term)
        stemmed_term = tokens[0]
        docs = self.index.get(stemmed_term, set())
        return sorted(docs)

    def build(self, movies: List[Dict[str, Any]]) -> None:
        """
        Build the inverted index from a list of movie objects.

        Each movie is added to the document map and indexed
        using the concatenation of its title and description.

        Args:
            movies: List of movie dictionaries. Each dictionary
                    must contain at least 'id', 'title', and 'description'.
        """
        for m in tqdm(movies, desc="Building index", unit="movies"):
            doc_id: int = m["id"]
            self.docmap[doc_id] = m

            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def __get_avg_doc_length(self) -> float:
        """Return the average document length across all indexed documents."""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def save(self) -> None:
        """
        Persist the index, document map, term frequencies and document lengths to disk.

        Files created:
            cache/index.pkl            - serialized inverted index
            cache/docmap.pkl           - serialized document map
            cache/term_frequencies.pkl - serialized term frequency counters
            cache/doc_lengths.pkl      - serialized document length map

        The cache directory is created automatically if it does not exist.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """
        Load the index, document map, term frequencies and document lengths from disk.

        Files read:
            cache/index.pkl            - serialized inverted index
            cache/docmap.pkl           - serialized document map
            cache/term_frequencies.pkl - serialized term frequency counters
            cache/doc_lengths.pkl      - serialized document length map
        """
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)

        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "rb") as f:
            self.doc_lengths = pickle.load(f)

    def bm25_search(self, query: str, limit: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using full BM25 scoring (BM25_TF × BM25_IDF) across all query tokens.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of (movie, score) tuples sorted by score descending.
        """
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Use tokenized term counts for length normalization in search,
        # consistent with how term_frequencies are built.
        avg_doc_len = (
            sum(sum(tf.values()) for tf in self.term_frequencies.values()) / len(self.term_frequencies)
            if self.term_frequencies else 1.0
        )
        scores: Dict[int, float] = {}

        for token in query_tokens:
            # BM25 IDF for this token
            N = len(self.docmap)
            df = len(self.index.get(token, set()))
            token_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # Accumulate BM25 score for each document containing this token
            for doc_id in self.index.get(token, set()):
                tf_counter = self.term_frequencies.get(doc_id, Counter())
                raw_tf = tf_counter.get(token, 0)
                doc_len = sum(tf_counter.values())
                length_norm = 1 - BM25_B + BM25_B * (doc_len / avg_doc_len)
                token_bm25_tf = (raw_tf * (BM25_K1 + 1)) / (raw_tf + BM25_K1 * length_norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + token_bm25_tf * token_idf

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in sorted_docs[:limit]]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for movies matching the query using the inverted index.

        This is much more efficient than linear search as it uses
        the inverted index to quickly find matching documents.

        Args:
            query: Search query sentence

        Returns:
            List of first 5 matching movie dictionaries sorted by ID
        """
        # Tokenize the query
        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        # Collect all document IDs that contain any query token (OR operation)
        matching_doc_ids: Set[int] = set()
        for token in query_tokens:
            doc_ids = self.index.get(token, set())
            matching_doc_ids.update(doc_ids)

        # Retrieve the actual movie documents
        matching_movies = [self.docmap[doc_id] for doc_id in matching_doc_ids]

        # Sort by ID and return first 5 results
        sorted_movies = sorted(matching_movies, key=lambda m: m["id"])
        return sorted_movies[:5]

