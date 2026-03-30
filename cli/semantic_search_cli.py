#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from cli.search_engines.semantic_search import verify_embeddings, embed_text, embed_query_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify_embeddings", help="Print model information")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text input")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()