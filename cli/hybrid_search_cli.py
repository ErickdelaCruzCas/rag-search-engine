#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import os
import time
from dotenv import load_dotenv
from cli.lib.search_utils import normalize
from cli.data_loader import load_movies
from cli.search_engines.hybrid_search import HybridSearch

load_dotenv()


def enhance_query(query: str, method: str) -> str:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    if method == "spell":
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        return response.text.strip()
    if method == "rewrite":
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        return response.text.strip()
    if method == "expand":
        prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
        response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        expansion = response.text.strip()
        return f"{query} {expansion}"
    return query


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    for entry in results:
        doc = entry["doc"]
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", doc.get("description", ""))}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        try:
            entry["rerank_score"] = float(response.text.strip())
        except ValueError:
            entry["rerank_score"] = 0.0
        time.sleep(3)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    doc_lines = []
    id_to_entry = {}
    for entry in results:
        doc = entry["doc"]
        doc_id = doc["id"]
        description = doc.get("description", doc.get("document", ""))[:100]
        doc_lines.append(f"ID {doc_id}: {doc.get('title', '')} - {description}")
        id_to_entry[doc_id] = entry

    doc_list_str = "\n".join(doc_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    ranked_ids = json.loads(response.text.strip())

    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in id_to_entry:
            id_to_entry[doc_id]["rerank_rank"] = rank

    for entry in results:
        if "rerank_rank" not in entry:
            entry["rerank_rank"] = len(results) + 1

    return sorted(results, key=lambda x: x["rerank_rank"])


def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    from sentence_transformers import CrossEncoder

    pairs = []
    for entry in results:
        doc = entry["doc"]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', doc.get('description', ''))}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for entry, score in zip(results, scores):
        entry["cross_encoder_score"] = float(score)

    return sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)


def evaluate_results(query: str, results: list[dict]) -> None:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    formatted_results = []
    for i, entry in enumerate(results, 1):
        doc = entry["doc"]
        description = doc.get("description", doc.get("document", ""))[:100]
        formatted_results.append(f"{i}. {doc.get('title', '')} - {description}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    scores = json.loads(response.text.strip())

    print()
    for i, (entry, score) in enumerate(zip(results, scores), 1):
        doc = entry["doc"]
        print(f"{i}. {doc.get('title', '')}: {score}/3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search with alpha weighting")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (default: 0.5)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Re-ranking method")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Evaluate results with LLM after search")
    rrf_parser.add_argument("--debug", action="store_true", help="Print debug info at each pipeline stage")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return
            for score in normalize(scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, entry in enumerate(results, 1):
                doc = entry["doc"]
                description = doc.get("description", "")
                print(f"{i}. {doc['title']}")
                print(f"  Hybrid Score: {entry['hybrid_score']:.3f}")
                print(f"  BM25: {entry['bm25_score']:.3f}, Semantic: {entry['semantic_score']:.3f}")
                print(f"  {description[:100]}...")
        case "rrf-search":
            query = args.query
            debug = args.debug

            if debug:
                print(f"[debug] Original query: '{query}'")

            if args.enhance:
                enhanced = enhance_query(query, args.enhance)
                if enhanced != query:
                    print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced

            if debug:
                print(f"[debug] Query after enhancement: '{query}'")

            documents = load_movies()
            hs = HybridSearch(documents)
            fetch_limit = args.limit * 5 if args.rerank_method else args.limit
            results = hs.rrf_search(query, args.k, fetch_limit)

            if debug:
                print(f"[debug] RRF results (top {fetch_limit}):")
                for i, entry in enumerate(results, 1):
                    print(f"  {i}. {entry['doc']['title']} (rrf={entry['rrf_score']:.4f}, bm25_rank={entry['bm25_rank']}, sem_rank={entry['semantic_rank']})")
                print()

            if args.rerank_method == "individual":
                print(f"Re-ranking top {fetch_limit} results using individual method...")
                results = rerank_individual(query, results)
                results = results[:args.limit]
            elif args.rerank_method == "batch":
                print(f"Re-ranking top {fetch_limit} results using batch method...")
                results = rerank_batch(query, results)
                results = results[:args.limit]
            elif args.rerank_method == "cross_encoder":
                print(f"Re-ranking top {fetch_limit} results using cross_encoder method...")
                results = rerank_cross_encoder(query, results)
                results = results[:args.limit]

            if debug and args.rerank_method:
                print(f"[debug] After re-ranking (top {args.limit}):")
                for i, entry in enumerate(results, 1):
                    score_info = ""
                    if args.rerank_method == "individual":
                        score_info = f", rerank_score={entry.get('rerank_score', 'N/A'):.3f}"
                    elif args.rerank_method == "batch":
                        score_info = f", rerank_rank={entry.get('rerank_rank', 'N/A')}"
                    elif args.rerank_method == "cross_encoder":
                        score_info = f", ce_score={entry.get('cross_encoder_score', 'N/A'):.4f}"
                    print(f"  {i}. {entry['doc']['title']} (rrf={entry['rrf_score']:.4f}{score_info})")
                print()
            print(f"\nReciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, entry in enumerate(results, 1):
                doc = entry["doc"]
                description = doc.get("description", doc.get("document", ""))
                bm25_rank = entry["bm25_rank"] if entry["bm25_rank"] is not None else "N/A"
                semantic_rank = entry["semantic_rank"] if entry["semantic_rank"] is not None else "N/A"
                print(f"{i}. {doc['title']}")
                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {entry['rerank_score']:.3f}/10")
                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {entry['rerank_rank']}")
                elif args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {entry['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {entry['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {description[:100]}...")
                print()
            if args.evaluate:
                evaluate_results(query, results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
