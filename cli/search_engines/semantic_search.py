"""Semantic search engine using sentence-transformers embeddings."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = Path("cache/movie_embeddings.npy")


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text must not be empty or whitespace.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(strings, show_progress_bar=True)
        np.save(EMBEDDINGS_CACHE, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if EMBEDDINGS_CACHE.exists():
            self.embeddings = np.load(EMBEDDINGS_CACHE)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        results = [
            (cosine_similarity(query_embedding, self.embeddings[i]), self.documents[i])
            for i in range(len(self.documents))
        ]
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in results[:limit]
        ]


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings() -> None:
    from cli.data_loader import load_movies

    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
