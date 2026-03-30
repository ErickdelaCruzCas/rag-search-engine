"""Semantic search engine using sentence-transformers embeddings."""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from cli.lib.search_utils import SCORE_PRECISION, semantic_chunk

MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = Path("cache/movie_embeddings.npy")
CHUNK_EMBEDDINGS_CACHE = Path("cache/chunk_embeddings.npy")
CHUNK_METADATA_CACHE = Path("cache/chunk_metadata.json")


class SemanticSearch:
    def __init__(self, model_name: str = MODEL) -> None:
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        total = len(documents)
        for movie_idx, doc in enumerate(documents):
            if movie_idx % 500 == 0:
                print(f"  Chunking documents... {movie_idx}/{total}", flush=True)

            description = doc.get("description", "")
            if not description or not description.strip():
                continue

            chunks = semantic_chunk(description, max_chunk_size=4, overlap=1)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })

        print(f"  Chunking documents... {total}/{total} — done")

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(CHUNK_EMBEDDINGS_CACHE, self.chunk_embeddings)
        with open(CHUNK_METADATA_CACHE, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if CHUNK_EMBEDDINGS_CACHE.exists() and CHUNK_METADATA_CACHE.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_CACHE)
            with open(CHUNK_METADATA_CACHE) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        # Vectorized cosine similarity: (N, D) @ (D,) / norms
        norms = np.linalg.norm(self.chunk_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        scores = (self.chunk_embeddings @ query_embedding) / (norms * query_norm + 1e-10)

        chunk_scores = []
        for i, score in enumerate(scores):
            meta = self.chunk_metadata[i]
            chunk_scores.append({
                "global_idx": i,
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": float(score),
            })

        movie_scores = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in movie_scores or cs["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = cs

        sorted_scores = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        top = sorted_scores[:limit]

        results = []
        for cs in top:
            doc = self.documents[cs["movie_idx"]]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(cs["score"], SCORE_PRECISION),
                "metadata": self.chunk_metadata[cs["global_idx"]] if self.chunk_metadata else {},
            })
        return results


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
