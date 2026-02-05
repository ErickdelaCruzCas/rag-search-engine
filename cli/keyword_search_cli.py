#!/usr/bin/env python3
"""Command-line interface for movie keyword search."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from cli.search_engine import MovieSearchEngine
from cli.inverted_index import InvertedIndex
from cli.data_loader import load_movies


def display_results(movies: list[dict]) -> None:
    """Display search results in numbered format."""
    for i, movie in enumerate(movies, start=1):
        print(f"{i}. {movie['title']}")


def main() -> None:
    """Run the CLI application."""
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keyword matching")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        
        case "build":
            print("Building inverted index...")
            # 1. Cargar películas

            movies = load_movies()

            # 2. Construir índice
            index = InvertedIndex()
            index.build(movies)

            # 3. Guardar en disco
            index.save()

            # 4. Verificar token 'merida'
            docs = index.get_documents("merida")
            if docs:
                print(f"First document ID for token 'merida': {docs[0]}")
            else:
                print("Token 'merida' not found")

        case "search":
            print(f"Searching for: {args.query}")
            search_engine = MovieSearchEngine()
            results = search_engine.search(args.query)
            display_results(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
