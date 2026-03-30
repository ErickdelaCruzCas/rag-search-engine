#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from cli.search_engines.semantic_search import verify_embeddings, embed_text, embed_query_text, SemanticSearch, ChunkedSemanticSearch
from cli.data_loader import load_movies
from cli.lib.search_utils import semantic_chunk


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

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlapping words between chunks (default: 0)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text into sentence-based chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlapping sentences between chunks (default: 0)")

    subparsers.add_parser("embed_chunks", help="Build or load chunked embeddings for all movies")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunk embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

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
        case "chunk":
            if args.overlap >= args.chunk_size:
                print(f"Error: overlap ({args.overlap}) must be less than chunk-size ({args.chunk_size})")
                return
            words = args.text.split()
            chunks = []
            i = 0
            while i < len(words):
                chunks.append(" ".join(words[i:i + args.chunk_size]))
                i += args.chunk_size - args.overlap
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            if args.overlap >= args.max_chunk_size:
                print(f"Error: overlap ({args.overlap}) must be less than max-chunk-size ({args.max_chunk_size})")
                return
            chunks = semantic_chunk(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()