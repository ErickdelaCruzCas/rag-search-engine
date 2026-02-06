import os
import pickle
from typing import Dict, Set, List, Any
from cli.tokenizer import Tokenizer


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
        """
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}
        self.tokenizer = tokenizer

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Add a document to the inverted index.

        The text is tokenized and each token is associated
        with the provided document ID.

        Args:
            doc_id: Unique identifier of the document.
            text: Text content used for indexing.
        """
        tokens = self.tokenizer.tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

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
        for m in movies:
            doc_id: int = m["id"]
            self.docmap[doc_id] = m

            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        """
        Persist the index and document map to disk using pickle.

        Files created:
            cache/index.pkl  - serialized inverted index
            cache/docmap.pkl - serialized document map

        The cache directory is created automatically if it does not exist.
        """
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        """
        Load the index and document map from disk.

        Files read:
            cache/index.pkl  - serialized inverted index
            cache/docmap.pkl - serialized document map
        """
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

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

