import re
import os
import pickle
from typing import Dict, Set, List, Any


class InvertedIndex:
    """
    A simple inverted index that maps tokens to document IDs
    and stores the full document objects in a separate map.
    """

    def __init__(self) -> None:
        """
        Initialize an empty inverted index.

        Attributes:
            index: Dictionary mapping tokens (str) to sets of document IDs (int).
            docmap: Dictionary mapping document IDs (int) to full document objects.
        """
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into lowercase word tokens.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of lowercase tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Add a document to the inverted index.

        The text is tokenized and each token is associated
        with the provided document ID.

        Args:
            doc_id: Unique identifier of the document.
            text: Text content used for indexing.
        """
        tokens = self._tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        """
        Retrieve document IDs containing a given term.

        Args:
            term: A single search token.

        Returns:
            A list of document IDs sorted in ascending order.
            Returns an empty list if the term is not found.
        """
        term = term.lower()
        docs = self.index.get(term, set())
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

