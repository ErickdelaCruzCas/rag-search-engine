# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Build the inverted index (must be done before searching; run from project root)
keyword-search build

# Search
keyword-search search "space adventure"

# Relevance scoring
keyword-search tf <doc_id> <term>       # term frequency
keyword-search idf <term>               # inverse document frequency
keyword-search tfidf <doc_id> <term>    # TF-IDF score
keyword-search bm25idf <doc_id> <term>  # BM25 IDF score
```

**Important:** All CLI commands and file I/O use relative paths (`data/`, `cache/`), so they must be run from the project root.

## Architecture

This is an educational search engine built incrementally toward RAG. Python 3.14, managed with `uv`. The `keyword-search` binary is registered in `pyproject.toml` as `cli.keyword_search_cli:main`.

### Data flow

**Build phase:** `data/movies.json` → `data_loader.load_movies()` → `InvertedIndex.build()` (tokenizes title + description) → pickled to `cache/` (3 files: `index.pkl`, `docmap.pkl`, `term_frequencies.pkl`).

**Search phase:** cached index loaded from `cache/` → query tokenized with same pipeline → OR lookup across index → top 5 results sorted by doc ID.

### Key modules

- `cli/keyword_search_cli.py` — argparse CLI; routes subcommands to handlers; TF/IDF/TF-IDF math lives here as standalone functions
- `cli/search_engines/inverted_index.py` — `InvertedIndex` class; three internal structures: `index` (`dict[str, set[int]]`), `docmap` (`dict[int, movie]`), `term_frequencies` (`dict[int, Counter]`); also implements BM25-saturated TF via `get_bm25_tf()`
- `cli/tokenizer.py` — `Tokenizer` class; pipeline: normalize → split → remove stopwords → Porter stem (via nltk); same instance used for both indexing and querying
- `cli/data_loader.py` — loads `data/movies.json` and `data/stopwords.txt`
- `cli/constants.py` — `BM25_K1 = 1.5`
- `cli/search_engines/linear_search.py` — deprecated O(n) scan, kept for reference

### Tokenization consistency

The same `Tokenizer` instance (with identical stopwords and stemmer) must be used for both building the index and processing queries. Stems are not real words (e.g. `"movies"` → `"movi"`); what matters is that indexing and querying produce the same stems.

### Planned next steps (from README)

Dense vector search (embeddings + cosine similarity) → re-ranker → LLM prompt with top-K retrieved docs for answer generation.
