# RAG Search Engine

A from-scratch search engine built to understand the fundamentals behind Retrieval-Augmented Generation (RAG). The project is developed incrementally — each module introduces a new concept, from naive linear search all the way to TF-IDF scoring on top of an inverted index.

The dataset is a collection of movies (title + description). The engine lets you index them, search by keyword, and calculate relevance scores.

---

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py      # CLI entry point
│   ├── data_loader.py             # Loads movies and stopwords from disk
│   ├── tokenizer.py               # Text normalization, stemming pipeline
│   └── search_engines/
│       ├── scorer.py              # TF-IDF and BM25 scoring functions
│       ├── linear_search.py       # Naive O(n) search (deprecated)
│       └── inverted_index.py      # Efficient index-based search + BM25
├── data/
│   ├── movies.json                # Movie corpus (title + description)
│   └── stopwords.txt              # Words to ignore during indexing
└── cache/                         # Persisted index files (auto-generated)
    ├── index.pkl
    ├── docmap.pkl
    ├── term_frequencies.pkl
    └── doc_lengths.pkl
```

---

## Concepts Covered

### 1. Tokenization

Before any search can happen, raw text must be converted into a normalized, comparable form. The `Tokenizer` class ([cli/tokenizer.py](cli/tokenizer.py)) applies a four-step pipeline:

```
raw text → normalize → split → remove stopwords → stem
```

**Step 1 — Normalize:** Convert to lowercase and strip punctuation.
`"Spider-Man: No Way Home!" → "spiderman no way home"`

**Step 2 — Split:** Break into individual word tokens.
`["spiderman", "no", "way", "home"]`

**Step 3 — Remove stopwords:** Filter out common words that carry no meaningful signal (e.g. *the*, *a*, *no*, *is*). These are loaded from `data/stopwords.txt`.
`["spiderman", "way", "home"]`

**Step 4 — Stem:** Reduce words to their root form using the **Porter Stemming Algorithm**. This ensures that *running*, *runs*, and *ran* all match the same index entry (`run`).
`["spiderman", "wai", "home"]`

The same pipeline is applied identically to both documents at index time and to queries at search time. This consistency is what makes matches work across different word forms.

---

### 2. Stopwords

Stopwords are extremely common words (*the*, *and*, *of*, *in*, *a*…) that appear in almost every document. Including them in the index would:
- Waste memory (they map to nearly every document)
- Pollute results (every document would match a query like *"the matrix"* because of *"the"*)

By filtering them out early in the tokenization pipeline, the index stays lean and results stay relevant.

---

### 3. The Porter Stemming Algorithm

Stemming is the process of reducing a word to its base or root form. The **Porter Stemmer** (1980) applies a series of rule-based suffix-stripping steps:

| Original    | Stemmed  |
|-------------|----------|
| running     | run      |
| movies      | movi     |
| historical  | histor   |
| searching   | search   |

Stems are not always real words — what matters is that the same stem is produced for all related word forms, both when indexing and when querying.

---

### 4. Linear Search (the naive approach)

The first implementation ([cli/search_engines/linear_search.py](cli/search_engines/linear_search.py)) scans every movie in the dataset for each query:

```python
for movie in self.movies:
    title_tokens = self.tokenizer.tokenize(movie["title"])
    if any(q in t for q in query_tokens for t in title_tokens):
        results.append(movie)
```

**Time complexity: O(n)** — every query reads every document.
This works fine for small datasets, but becomes impractical at scale (millions of documents). It also only searches titles, not descriptions. This approach is marked as `deprecated` in the code.

---

### 5. Inverted Index

An **inverted index** is the core data structure behind every real search engine (Google, Elasticsearch, Lucene all use this concept).

Instead of asking *"does this document contain the word?"* for every document, we precompute the answer during an indexing phase and store it as:

```
token → {set of document IDs that contain it}
```

**Example:**

| Token    | Document IDs       |
|----------|--------------------|
| "spiderman" | {12, 47, 203}   |
| "histor"    | {5, 47, 98, 312} |
| "wai"       | {12, 98}         |

At query time, looking up a word is an O(1) dictionary lookup — no scanning needed.

The `InvertedIndex` class ([cli/search_engines/inverted_index.py](cli/search_engines/inverted_index.py)) stores four structures:

| Attribute          | Type                          | Purpose                                      |
|--------------------|-------------------------------|----------------------------------------------|
| `index`            | `dict[str, set[int]]`         | Token → set of matching doc IDs             |
| `docmap`           | `dict[int, dict]`             | Doc ID → full movie object                  |
| `term_frequencies` | `dict[int, Counter]`          | Doc ID → term count within that document    |
| `doc_lengths`      | `dict[int, int]`              | Doc ID → raw word count (for BM25 length normalization) |

**Build phase:** The entire corpus is processed once. Each movie's title and description are tokenized together and added to the index.

**Search phase:** Query tokens are looked up in the index (OR logic — any token match returns the document). Results are sorted by ID and capped at 5.

**Persistence:** The index is serialized to disk with `pickle` in the `cache/` directory, so it only needs to be built once.

---

### 6. TF-IDF (Term Frequency – Inverse Document Frequency)

Once we can retrieve documents, the next question is: **which results are most relevant?** TF-IDF is the classic statistical measure for this.

#### Term Frequency (TF)

How many times does the term appear in a specific document?

```
TF(term, doc) = count of term in doc
```

A term that appears 5 times in a movie description is more relevant to that movie than one that appears once.

#### Inverse Document Frequency (IDF)

How rare is the term across the entire corpus?

```
IDF(term) = log( (total_docs + 1) / (docs_containing_term + 1) )
```

- A term that appears in every document (like a stopword would) gets a near-zero IDF → low weight.
- A term that appears in only 2 out of 10,000 documents gets a high IDF → high weight.

The `+1` in numerator and denominator is **Laplace smoothing**, which prevents division by zero and avoids extreme values for very rare terms.

#### TF-IDF Score

```
TF-IDF(term, doc) = TF × IDF
```

This score is high when:
- The term appears **frequently in this document** (high TF), AND
- The term is **rare across the corpus** (high IDF)

It is low when the term is either absent from the document or so common it appears everywhere.

The CLI exposes all three metrics as individual commands so you can inspect the values step by step.

---

### 7. BM25 (Best Match 25)

TF-IDF has a flaw: raw TF is unbounded. A term that appears 100 times scores 100× higher than one that appears once, even though in practice the relevance gain tapers off long before that.

**BM25 fixes this with two improvements:**

#### Saturation (k1 parameter)

```
BM25_TF(term, doc) = (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |D| / avgdl))
```

The `k1` parameter (default `1.5`) controls how fast the score saturates. As `tf` grows, the result asymptotically approaches `k1 + 1` — it never exceeds it, no matter how many times the term appears.

#### Length normalization (b parameter)

The term `(1 - b + b × |D| / avgdl)` normalizes for document length, where:
- `|D|` = number of words in the document
- `avgdl` = average document length across the corpus
- `b` (default `0.75`) controls the strength of the normalization. `b=0` disables it; `b=1` applies full normalization.

This prevents long documents from having an unfair advantage: a term appearing 5 times in a short document should score higher than the same term appearing 5 times in a 10× longer one.

| tf | BM25_TF (k1=1.5, normalized) |
|----|------------------------------|
| 1  | ≤ 1.00                       |
| 2  | ≤ 1.40                       |
| 5  | ≤ 1.67                       |
| 10 | ≤ 1.77                       |
| 100| ≤ 1.97                       |

#### BM25 IDF

Uses a different formula than classic IDF, penalising very frequent terms more aggressively:

```
BM25_IDF(term) = log( (N - df + 0.5) / (df + 0.5) + 1 )
```

#### Full BM25 score

```
BM25(term, doc) = BM25_TF × BM25_IDF
```

The final document score for a multi-term query sums BM25 over all query tokens. The scoring functions live in [`cli/search_engines/scorer.py`](cli/search_engines/scorer.py) (`bm25_tf`, `bm25_idf`, `bm25`). Document lengths are stored in `doc_lengths` (raw word count per document, computed at index time and persisted to `cache/doc_lengths.pkl`).

---

## Setup

```bash
# Install dependencies with uv
uv sync

# Build the inverted index (run once)
keyword-search build
```

---

## Usage

```bash
# Search for movies (keyword, OR logic)
keyword-search search "space adventure"

# Search for movies ranked by BM25 score
keyword-search bm25search "love story"

# Get term frequency of "action" in document 42
keyword-search tf 42 action

# Get inverse document frequency of "robot"
keyword-search idf robot

# Get TF-IDF score of "war" in document 7
keyword-search tfidf 7 war

# Get BM25 TF score of "love" in document 1 (optional custom k1 and b)
keyword-search bm25tf 1 love
keyword-search bm25tf 1 love 2.0 0.5

# Get BM25 IDF score of "robot"
keyword-search bm25idf robot
```

---

## What's Next (RAG)

This project builds the **retrieval** half of a RAG system. The pipeline so far:

```
Query → Tokenize → Index Lookup → TF-IDF Ranking → Top-K Documents
```

In a full RAG setup, those top-K retrieved documents are passed as context to a language model (e.g. Claude, GPT), which then generates a grounded answer based on the retrieved evidence — rather than hallucinating from its parametric memory alone.

The next steps would be:
- Replace keyword matching with **dense vector search** (embeddings + cosine similarity)
- Add a **re-ranker** to improve result ordering
- Pipe the top results into an **LLM prompt** for answer generation
