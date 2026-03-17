#!/usr/bin/env python3
"""Command-line interface for movie keyword search."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Optional
from cli.search_engines.inverted_index import InvertedIndex
from cli.search_engines import scorer
from cli.data_loader import load_movies, load_stopwords
from cli.tokenizer import Tokenizer
from cli.constants import BM25_K1, BM25_B


def display_results(movies: list[dict]) -> None:
    """Display search results in numbered format."""
    for i, movie in enumerate(movies, start=1):
        print(f"{i}. {movie['title']}")


def load_index(tokenizer: Tokenizer) -> InvertedIndex:
    """Load the inverted index from disk."""
    index = InvertedIndex(tokenizer)
    index.load()
    return index


def get_stemmed_term(tokenizer: Tokenizer, term: str) -> Optional[str]:
    """
    Tokenize and return the stemmed version of a term.

    Returns:
        The first stemmed token, or None if no tokens produced.
    """
    tokens = tokenizer.tokenize(term)
    return tokens[0] if tokens else None


def handle_bm25search(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    """Handle the BM25 search command."""
    index = load_index(tokenizer)
    results = index.bm25_search(args.query, args.limit)
    for i, (movie, score) in enumerate(results, start=1):
        print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Load the index and return the BM25 TF score for a term in a document."""
    stopwords = load_stopwords()
    tokenizer = Tokenizer(stopwords)
    index = load_index(tokenizer)
    stemmed_term = get_stemmed_term(tokenizer, term)
    bm25tf = scorer.bm25_tf(index, doc_id, stemmed_term or "", k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    return bm25tf


def bm25_idf_command(term: str) -> float:
    """Load the index and return the BM25 IDF score for a term across the corpus."""
    stopwords = load_stopwords()
    tokenizer = Tokenizer(stopwords)
    index = load_index(tokenizer)
    stemmed_term = get_stemmed_term(tokenizer, term)
    score = scorer.bm25_idf(index, stemmed_term or "")
    print(f"BM25 IDF score of '{term}': {score:.2f}")
    return score


def handle_build(tokenizer: Tokenizer) -> None:
    """Handle the build command."""
    print("Building inverted index...")
    movies = load_movies()
    index = InvertedIndex(tokenizer)
    index.build(movies)
    index.save()
    print("Index built and saved successfully!")


def handle_search(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    """Handle the search command."""
    print(f"Searching for: {args.query}")
    index = load_index(tokenizer)
    results = index.search(args.query)
    display_results(results)


def handle_tf(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    """Handle the term frequency command."""
    index = load_index(tokenizer)
    stemmed_term = get_stemmed_term(tokenizer, args.term)
    print(scorer.tf(index, args.doc_id, stemmed_term or ""))


def handle_idf(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    """Handle the inverse document frequency command."""
    index = load_index(tokenizer)
    stemmed_term = get_stemmed_term(tokenizer, args.term)
    score = scorer.idf(index, stemmed_term or "")
    print(f"Inverse document frequency of '{args.term}': {score:.2f}")


def handle_tfidf(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    """Handle the TF-IDF calculation command."""
    index = load_index(tokenizer)
    stemmed_term = get_stemmed_term(tokenizer, args.term)
    score = scorer.tfidf(index, args.doc_id, stemmed_term or "")
    print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {score:.2f}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    subparsers.add_parser("build", help="Build inverted index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search movies using keyword matching")
    search_parser.add_argument("query", type=str, help="Search query")

    # Term frequency command
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    # IDF command
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    # TF-IDF command
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

    # BM25 TF command
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # BM25 search command
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # BM25 IDF command
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a term across the corpus"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    return parser


def main() -> None:
    # print("Welcome to the Movie Keyword Search CLI!")
    """Run the CLI application."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load stopwords and create tokenizer
    stopwords = load_stopwords()
    tokenizer = Tokenizer(stopwords)
    # print(f"Command: {args.command}")
    # Route to appropriate handler
    match args.command:
        case "build":
            handle_build(tokenizer)
        case "search":
            handle_search(args, tokenizer)
        case "tf":
            handle_tf(args, tokenizer)
        case "idf":
            handle_idf(args, tokenizer)
        case "tfidf":
            handle_tfidf(args, tokenizer)
        case "bm25search":
            handle_bm25search(args, tokenizer)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25idf":
            bm25_idf_command(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
