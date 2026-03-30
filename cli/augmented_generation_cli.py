#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os

from dotenv import load_dotenv

from cli.data_loader import load_movies
from cli.search_engines.hybrid_search import HybridSearch

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    question_parser = subparsers.add_parser(
        "question", help="Answer a question conversationally based on search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with cited sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            print("Search Results:")
            doc_lines = []
            for entry in results:
                doc = entry["doc"]
                title = doc.get("title", "")
                description = doc.get("description", doc.get("document", ""))
                print(f"- {title}")
                doc_lines.append(f"{title}: {description}")

            docs = "\n".join(doc_lines)

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("\nRAG Response:")
            print(response.text.strip())

        case "question":
            question = args.question

            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.rrf_search(question, k=60, limit=args.limit)

            print("Search Results:")
            context_lines = []
            for entry in results:
                doc = entry["doc"]
                title = doc.get("title", "")
                description = doc.get("description", doc.get("document", ""))
                print(f"  - {title}")
                context_lines.append(f"{title}: {description}")

            context = "\n".join(context_lines)

            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("\nAnswer:")
            print(response.text.strip())

        case "citations":
            query = args.query

            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            doc_lines = []
            for i, entry in enumerate(results, 1):
                doc = entry["doc"]
                title = doc.get("title", "")
                description = doc.get("description", doc.get("document", ""))
                print(f"  - {title}")
                doc_lines.append(f"[{i}] {title}: {description}")

            documents = "\n".join(doc_lines)

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("\nLLM Answer:")
            print(response.text.strip())

        case "summarize":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            search_results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            result_lines = []
            for entry in search_results:
                doc = entry["doc"]
                title = doc.get("title", "")
                description = doc.get("description", doc.get("document", ""))
                print(f"  - {title}")
                result_lines.append(f"{title}: {description}")

            results = "\n".join(result_lines)

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("\nLLM Summary:")
            print(response.text.strip())

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
