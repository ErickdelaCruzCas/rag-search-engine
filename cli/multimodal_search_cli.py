#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from cli.lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding shape")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search movies by image")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['score']:.3f})")
                print(f"   {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
