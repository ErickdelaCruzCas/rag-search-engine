#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from cli.search_engines.semantic_search import verify_embeddings, embed_text, embed_query_text, SemanticSearch
from cli.data_loader import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify_embeddings", help="Print model information")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text input")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search over movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            ss = SemanticSearch()
            documents = load_movies()
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()