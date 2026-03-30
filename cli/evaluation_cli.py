#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json

from cli.data_loader import load_movies
from cli.search_engines.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_titles = set(test_case["relevant_docs"])

        results = hs.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [entry["doc"]["title"] for entry in results]

        hits = sum(1 for title in retrieved_titles if title in relevant_titles)
        precision = hits / limit if limit > 0 else 0.0
        recall = hits / len(relevant_titles) if relevant_titles else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(test_case["relevant_docs"])

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}")
        print()


if __name__ == "__main__":
    main()
