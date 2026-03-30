import os

from cli.search_engines.inverted_index import InvertedIndex
from cli.search_engines.semantic_search import ChunkedSemanticSearch
from cli.tokenizer import Tokenizer
from cli.data_loader import load_stopwords
from cli.lib.search_utils import normalize

INDEX_CACHE = os.path.join("cache", "index.pkl")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        tokenizer = Tokenizer(load_stopwords())
        self.idx = InvertedIndex(tokenizer)
        if not os.path.exists(INDEX_CACHE):
            self.idx.build(documents)
            self.idx.save()
        self.idx.load()

    def _bm25_search(self, query, limit):
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        fetch = limit * 500

        bm25_results = self._bm25_search(query, fetch)       # [(doc, score), ...]
        semantic_results = self.semantic_search.search_chunks(query, fetch)  # [{"id", "title", "document", "score"}, ...]

        # Normalize BM25 scores
        bm25_scores_raw = [score for _, score in bm25_results]
        bm25_scores_norm = normalize(bm25_scores_raw)

        # Normalize semantic scores
        sem_scores_raw = [r["score"] for r in semantic_results]
        sem_scores_norm = normalize(sem_scores_raw)

        # Build combined dict keyed by doc id
        combined = {}

        for (doc, _), norm_score in zip(bm25_results, bm25_scores_norm):
            doc_id = doc["id"]
            combined[doc_id] = {
                "doc": doc,
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        for result, norm_score in zip(semantic_results, sem_scores_norm):
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = norm_score
            else:
                combined[doc_id] = {
                    "doc": {"id": doc_id, "title": result["title"], "description": result["document"]},
                    "bm25_score": 0.0,
                    "semantic_score": norm_score,
                }

        # Compute hybrid score and sort
        for entry in combined.values():
            entry["hybrid_score"] = alpha * entry["bm25_score"] + (1 - alpha) * entry["semantic_score"]

        sorted_results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)

        return sorted_results[:limit]

    def rrf_search(self, query, k=60, limit=10):
        fetch = limit * 500

        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        combined = {}

        for rank, (doc, _) in enumerate(bm25_results, 1):
            doc_id = doc["id"]
            combined[doc_id] = {
                "doc": doc,
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": 1 / (k + rank),
            }

        for rank, result in enumerate(semantic_results, 1):
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf_score"] += 1 / (k + rank)
            else:
                combined[doc_id] = {
                    "doc": {"id": doc_id, "title": result["title"], "description": result["document"]},
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": 1 / (k + rank),
                }

        sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:limit]
