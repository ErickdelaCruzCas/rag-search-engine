# RAG Search Engine

A from-scratch search engine built to understand the fundamentals behind Retrieval-Augmented Generation (RAG). The project is developed incrementally — each module introduces a new concept, from naive linear search all the way to TF-IDF scoring on top of an inverted index.

The dataset is a collection of movies (title + description). The engine lets you index them, search by keyword, and calculate relevance scores.

---

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py      # Keyword search CLI entry point
│   ├── semantic_search_cli.py     # Semantic search CLI entry point
│   ├── data_loader.py             # Loads movies and stopwords from disk
│   ├── tokenizer.py               # Text normalization, stemming pipeline
│   └── search_engines/
│       ├── scorer.py              # TF-IDF and BM25 scoring functions
│       ├── linear_search.py       # Naive O(n) search (deprecated)
│       ├── inverted_index.py      # Efficient index-based search + BM25
│       └── semantic_search.py     # Embedding-based semantic search
├── data/
│   ├── movies.json                # Movie corpus (title + description)
│   └── stopwords.txt              # Words to ignore during indexing
└── cache/                         # Persisted index files (auto-generated)
    ├── index.pkl
    ├── docmap.pkl
    ├── term_frequencies.pkl
    ├── doc_lengths.pkl
    └── movie_embeddings.npy       # Pre-computed movie embeddings
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

### 8. Embeddings

Keyword search only finds documents that contain the exact words in the query. Semantic search finds documents that have the same *meaning*, even if they use completely different words.

An **embedding** is a dense numerical representation of text as a high-dimensional vector (e.g. 384 dimensions for `all-MiniLM-L6-v2`). The model is trained so that semantically similar texts produce vectors that point in roughly the same direction in that space.

```
"space adventure"     → [0.12, -0.34, 0.87, ...]   ← 384 numbers
"interstellar voyage" → [0.11, -0.31, 0.85, ...]   ← nearby vector
"romantic comedy"     → [-0.42, 0.67, -0.21, ...]  ← far away vector
```

The `SemanticSearch` class ([cli/search_engines/semantic_search.py](cli/search_engines/semantic_search.py)) uses the `sentence-transformers` library with the `all-MiniLM-L6-v2` model. Embeddings for all 5,000 movies are computed once and cached to `cache/movie_embeddings.npy`. On subsequent runs they are loaded from disk, skipping the expensive generation step.

---

### 9. Cosine Similarity

To compare two embedding vectors we use **cosine similarity**, which measures the angle between them rather than their distance. This makes it invariant to vector magnitude — only direction matters.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

The result ranges from **-1.0 to 1.0**:

| Score | Meaning |
|-------|---------|
| 1.0   | Vectors point in the same direction (identical meaning) |
| 0.0   | Perpendicular (unrelated) |
| -1.0  | Opposite directions (opposite meaning) |

In practice, embedding models produce positive values, so most scores fall between 0 and 1.

---

### 10. Semantic Search

The full semantic search pipeline has five steps:

```
1. Embed documents (once)  →  store 5,000 movie vectors in cache
2. Embed query (per search) →  convert query to a single vector
3. Cosine similarity        →  compare query vector to every movie vector
4. Rank                     →  sort by similarity score (descending)
5. Return top-K             →  surface the most semantically relevant results
```

Unlike keyword search, this finds relevant movies even when the query uses completely different words than the document. A search for *"space adventure"* will surface films described as *"an interstellar voyage"* or *"exploring the cosmos"* because their embeddings are nearby in vector space.

The `search(query, limit)` method in `SemanticSearch` implements this pipeline. It raises a `ValueError` if embeddings have not been loaded first.

---

### 11. Approximate Nearest Neighbor (ANN) Search

The semantic search implemented here does a brute-force comparison: every query vector is compared against all 5,000 movie vectors. That's fine for 5,000 documents, but at millions of vectors it becomes O(N) and too slow for production.

**ANN algorithms** solve this by trading a small amount of accuracy for a massive speed gain. Instead of scanning everything, they navigate data structures (graphs, clusters, hash buckets) to jump directly to promising regions of the vector space.

All three main approaches share the same two properties:
- They are **approximate** — they don't always return the mathematically closest vector.
- They are **fast** — search complexity drops from O(N) to something sublinear (logarithmic or better).

| Algorithm | How it works | Trade-off |
|-----------|-------------|-----------|
| **HNSW** (Hierarchical Navigable Small World) | Multilevel graph. Search starts at the top (coarse, long jumps) and refines going down. | Best recall and speed. The default in most modern vector databases. |
| **IVF** (Inverted File Index) | k-means clustering. Only searches the nearest clusters, ignores the rest. | Controllable: more clusters → more precision, less speed. |
| **LSH** (Locality-Sensitive Hashing) | Hashes that preserve similarity — close vectors land in the same bucket. Only compares within the bucket. | Very fast, but usually lower recall than HNSW. |

**Vector databases** (Pinecone, Qdrant, Chroma, Weaviate, pgvector) are storage engines built around these ANN algorithms. They handle indexing, persistence, filtering, and ANN search so you don't have to implement it yourself.

---

## Setup

```bash
# Install dependencies with uv
uv sync

# Build the inverted index (run once)
keyword-search build

# Generate and cache movie embeddings (run once, takes ~20s)
python cli/semantic_search_cli.py verify_embeddings
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

```bash
# Semantic search — find movies by meaning
python cli/semantic_search_cli.py search "space adventure"
python cli/semantic_search_cli.py search "space adventure" --limit 10

# Embed a single query and inspect its vector
python cli/semantic_search_cli.py embedquery "space adventure"

# Verify embeddings cache (generates it on first run)
python cli/semantic_search_cli.py verify_embeddings
```

---

## What's Next (RAG)

This project builds the **retrieval** half of a RAG system. The pipeline so far:

```
Query → Embed → Cosine Similarity → Top-K Documents (semantic)
Query → Tokenize → Index Lookup → BM25 Ranking → Top-K Documents (keyword)
```

In a full RAG setup, those top-K retrieved documents are passed as context to a language model (e.g. Claude, GPT), which then generates a grounded answer based on the retrieved evidence — rather than hallucinating from its parametric memory alone.

The next steps would be:
- Combine keyword and semantic results with **hybrid search** (RRF fusion)
- Add a **re-ranker** to improve result ordering
- Pipe the top results into an **LLM prompt** for answer generation

---

## Estrategias de Chunking en RAG

El chunking es el proceso de dividir documentos largos en fragmentos más pequeños antes de indexarlos. Los modelos de embeddings tienen un límite de tokens (típicamente 256–512), y los LLMs tienen una ventana de contexto finita. Decidir *cómo* partir el texto es una de las decisiones de ingeniería con mayor impacto en la calidad final de un sistema RAG.

Esta sección describe las 8 estrategias principales, su base algorítmica, su implementación práctica y sus compromisos de rendimiento.

---

### 1. Chunking de tamaño fijo (Fixed-Size Chunking)

El chunking de tamaño fijo divide el texto en fragmentos de exactamente N tokens o palabras, sin considerar la estructura semántica del contenido. Es la estrategia más simple y la más rápida de implementar: se recorre la lista de palabras como un array y se agrupa cada N elementos.

El problema principal es que ignora completamente los límites naturales del lenguaje. Una frase puede quedar partida a la mitad, repartida entre dos chunks distintos. Si la información relevante para responder una pregunta cae exactamente en ese corte, ninguno de los dos chunks la contendrá completa, y el retrieval fallará aunque el documento correcto esté indexado.

Es apropiado cuando los documentos son homogéneos en estructura (logs, registros tabulares, texto continuo sin estructura narrativa) y cuando la velocidad de indexación prima sobre la calidad de retrieval.

**Concepto DSA subyacente:** particionado de array en bloques de tamaño fijo. Tiempo O(N), espacio O(N/k) chunks donde k es el tamaño. Sin estado, sin lookahead.

```python
def fixed_chunk(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

chunks = fixed_chunk(document, chunk_size=150)
```

**Complejidad:** O(N) tiempo, O(N) espacio. Índice de tamaño proporcional al corpus. Sin overhead de computo en indexación. El cuello de botella es el modelo de embeddings, no el chunking en sí.

---

### 2. Chunk con solapamiento (Sliding Window / Chunk Overlap)

El chunking con solapamiento es una extensión directa del fijo: en lugar de avanzar k palabras entre chunks, avanza `k - overlap`. Las últimas `overlap` palabras del chunk anterior se repiten al inicio del siguiente, creando una ventana deslizante sobre el texto.

El objetivo es mitigar el problema del corte: si una pieza de información relevante cae cerca de un límite, al menos uno de los dos chunks solapados la contendrá completa. Es especialmente útil cuando las preguntas del usuario hacen referencia a conceptos que se desarrollan a lo largo de varias frases.

El coste directo es el aumento del número de chunks indexados. Con un overlap del 50%, se generan aproximadamente el doble de chunks, lo que duplica el coste de embeddings y el tamaño del índice vectorial.

**Concepto DSA subyacente:** ventana deslizante (sliding window). Estructura clásica para problemas donde el contexto local entre elementos adyacentes es importante. El puntero avanza `chunk_size - overlap` posiciones en cada iteración en lugar de `chunk_size`.

```python
def overlapping_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

chunks = overlapping_chunk(document, chunk_size=150, overlap=30)
```

**Complejidad:** O(N · overlap/chunk_size) chunks adicionales respecto al fijo. Con overlap=0 equivale al fixed chunking. A mayor overlap, mayor redundancia en el índice y mayor coste de retrieval por aumento del número de candidatos.

---

### 3. Chunking semántico (Semantic Chunking / Similarity-Based Boundary Detection)

En lugar de partir por número de palabras, el chunking semántico parte por significado: detecta los puntos del texto donde el tema cambia y usa esos puntos como fronteras naturales entre chunks. La versión simple usa límites de frase (signos de puntuación); la versión avanzada embede cada frase y busca los puntos donde la similitud coseno entre frases consecutivas cae por debajo de un umbral.

El principio es que frases consecutivas dentro de un mismo párrafo o argumento tienen embeddings cercanos. Cuando el tema cambia, la similitud cae bruscamente. Ese punto de inflexión es la frontera del chunk. El resultado son chunks que contienen ideas completas, no fragmentos arbitrarios de texto.

La versión basada en frases (regex) es O(N) y no requiere embeddings. La versión basada en similitud embede cada frase individualmente, lo que tiene un coste alto en indexación pero produce chunks mucho más coherentes semánticamente.

**Concepto DSA subyacente:** detección de puntos de ruptura (breakpoint detection) sobre una secuencia. Similar a la detección de anomalías en series temporales: se busca el punto donde una métrica (similitud coseno) cae por debajo de un umbral dinámico o estático.

```python
import re
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(text: str, threshold: float = 0.5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Complejidad:** O(S) llamadas al modelo de embeddings donde S es el número de frases. Mucho más caro que el fixed chunking en indexación. El tamaño del índice resultante es menor porque los chunks son más coherentes y hay menos redundancia.

---

### 4. Embeddings por chunk (Chunked Semantic Embeddings / Vector Indexing per Chunk)

Una vez que el texto está dividido en chunks, cada uno se convierte en un vector de alta dimensión con un modelo de embeddings. Estos vectores se almacenan en un índice vectorial (FAISS, Qdrant, pgvector) junto con metadata que permite recuperar el texto original y su origen (documento, página, sección).

La diferencia respecto a embeder documentos completos es de granularidad: en lugar de un vector por documento, hay N vectores por documento (uno por chunk). El retrieval devuelve el chunk específico que contiene la información, no el documento entero. Esto es crítico cuando los documentos son largos: el LLM recibe solo el fragmento relevante, no todo el libro.

El diseño del índice es un problema de ingeniería no trivial: hay que decidir si normalizar los vectores (para usar dot product en lugar de coseno), qué algoritmo ANN usar (HNSW para alta recall, IVF para datasets masivos), y cómo almacenar la metadata de forma que el filtrado por fuente o fecha sea eficiente.

**Concepto DSA subyacente:** modelo de espacio vectorial (Vector Space Model) con índice ANN. Cada chunk es un punto en un espacio de alta dimensión. El retrieval es una búsqueda de vecinos más cercanos. HNSW construye un grafo de proximidad en múltiples niveles para hacer esa búsqueda en O(log N) en lugar de O(N).

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def build_chunk_index(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    # Normalizar para usar dot product como proxy de similitud coseno
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized, chunks

embeddings, chunk_texts = build_chunk_index(chunks)
np.save("cache/chunk_embeddings.npy", embeddings)
```

**Complejidad:** O(C) embeddings donde C es el número de chunks. El índice ocupa C × D floats (D = dimensiones del modelo, típicamente 384–1536). Con C=100k chunks y D=384, son ~150MB en float32. HNSW añade overhead de grafo (~400MB adicionales para 100k vectores con parámetros típicos).

---

### 5. Búsqueda semántica por chunks (Chunked Semantic Search / Top-K Retrieval)

El retrieval sobre un índice de chunks es conceptualmente idéntico al retrieval sobre documentos: se embede la query, se calcula similitud coseno contra todos los vectores del índice, y se devuelven los K más cercanos. La diferencia es que cada resultado es un chunk, no un documento completo.

Un problema específico del retrieval por chunks es la **redundancia**: si el mismo documento tiene varios chunks relevantes, todos pueden aparecer en el top-K, ocupando espacio de contexto del LLM con información repetida. La solución es aplicar **max-margin diversity** o **deduplicación por documento**: si ya hay un chunk del documento X en los resultados, los siguientes chunks del mismo documento reciben una penalización.

Otro problema es el **contexto perdido**: el chunk recuperado puede ser correcto pero carecer del contexto necesario para que el LLM lo entienda (por ejemplo, un chunk que empieza con "Sin embargo, esto implica que..."). Una solución es recuperar también los chunks adyacentes en el documento original.

**Concepto DSA subyacente:** búsqueda de vecinos más cercanos (KNN/ANN) en espacio vectorial de alta dimensión. El problema es equivalente a encontrar los K puntos más cercanos en un espacio métrico. La complejidad depende del algoritmo: O(N·D) fuerza bruta, O(log N · D) con HNSW, O(√N · D) con IVF.

```python
def search_chunks(
    query: str,
    embeddings: np.ndarray,
    chunks: list[str],
    model,
    top_k: int = 5
) -> list[dict]:
    query_vec = model.encode([query])[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    scores = embeddings @ query_vec  # dot product sobre vectores normalizados
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"score": float(scores[i]), "chunk": chunks[i]}
        for i in top_indices
    ]
```

**Complejidad:** O(C·D) fuerza bruta, donde C = número de chunks y D = dimensiones. En producción, ANN reduce esto a O(log C · D). El cuello de botella en latencia suele ser el embedding de la query (una llamada al modelo), no la búsqueda en sí.

---

### 6. Casos extremos de chunking (Chunked Edge Cases)

El chunking de texto narrativo es relativamente sencillo, pero los documentos reales contienen estructuras que el chunking por palabras o frases destruye: tablas, bloques de código, fórmulas matemáticas, listas numeradas, y documentos extremadamente cortos.

Una tabla partido a la mitad pierde su estructura relacional — las columnas de la segunda mitad no tienen cabecera. Un bloque de código partido en medio de una función es inútil para el LLM. Un documento de 50 palabras no debería partirse en absoluto. Cada uno de estos casos requiere una estrategia diferente.

El enfoque robusto en producción es un **chunker jerárquico**: primero detecta el tipo de cada bloque (texto, código, tabla, lista) usando heurísticas o un parser de Markdown/HTML, luego aplica la estrategia apropiada a cada bloque. Los bloques de código y tablas se indexan completos o con separadores especiales que preservan su estructura.

**Concepto DSA subyacente:** árbol de análisis sintáctico (parse tree) sobre la estructura del documento. El documento se modela como un árbol donde los nodos son secciones, párrafos, tablas y bloques de código. El chunking respeta los límites del árbol en lugar de ignorarlos.

```python
import re

def smart_chunk(text: str, chunk_size: int = 200) -> list[str]:
    # Extraer bloques de código antes de partir
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    protected = {}
    for i, block in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_{i}__"
        text = text.replace(block, placeholder)
        protected[placeholder] = block

    # Chunking normal sobre el texto sin código
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        # Restaurar bloques de código
        for placeholder, block in protected.items():
            chunk = chunk.replace(placeholder, block)
        chunks.append(chunk)
        i += chunk_size

    return chunks
```

**Complejidad:** O(N) para el chunking base más O(B) para detectar y restaurar B bloques especiales. El overhead es mínimo. El coste real es la complejidad de implementación y mantenimiento de las heurísticas de detección de tipos.

---

### 7. ColBERT (Late Interaction Retrieval)

ColBERT (Contextualized Late Interaction over BERT) es un enfoque radicalmente diferente al retrieval por embedding único. En lugar de comprimir cada documento en un solo vector, ColBERT produce **un vector por token** tanto para la query como para el documento. La puntuación final se calcula con una operación de "late interaction": para cada token de la query, se encuentra su token más similar en el documento (MaxSim), y se suman esas similitudes máximas.

La ventaja es precisión: un embedding único pierde información al comprimir cientos de tokens en un solo vector. ColBERT preserva la representación token a token y calcula la interacción en tiempo de retrieval, no de indexación. El resultado es una recall y precision significativamente superiores, especialmente en queries complejas con múltiples conceptos.

El coste es espacio: en lugar de 1 vector por chunk, hay T vectores por chunk (donde T es la longitud en tokens). Para un corpus de 1M chunks con longitud media de 100 tokens y 128 dimensiones, el índice ColBERT ocupa ~50GB frente a ~400MB de un bi-encoder estándar.

**Concepto DSA subyacente:** producto de matrices con reducción MaxSim. Para cada token de query qi, se calcula `max_j(qi · dj)` sobre todos los tokens dj del documento. El score final es `Σ_i max_j(qi · dj)`. En práctica se implementa como una multiplicación matricial Q×D^T seguida de un max por fila y una suma.

```python
import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

# Indexar documentos
with Run().context(RunConfig(nranks=1, experiment="rag")):
    config = ColBERTConfig(doc_maxlen=220, nbits=2)
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
    indexer.index(name="my_index", collection=chunks, overwrite=True)

# Buscar
with Run().context(RunConfig(experiment="rag")):
    searcher = Searcher(index="my_index")
    results = searcher.search("space adventure", k=5)
    for rank, (doc_id, _, score) in enumerate(zip(*results)):
        print(f"{rank+1}. [{score:.2f}] {chunks[doc_id]}")
```

**Complejidad:** indexación O(C·T·D) donde T es la longitud media en tokens. Retrieval: fase FAISS O(log C) para preselección, fase MaxSim O(K·T·T_q) sobre los K candidatos. El índice ocupa O(C·T·D) floats — típicamente 100× más que un bi-encoder. La librería `RAGatouille` simplifica la integración con pipelines existentes.

---

### 8. Late Chunking (Document-Level Encoding + Post-Pooling)

Late Chunking invierte el orden habitual: en lugar de partir el texto primero y embeder después, embede primero el documento completo y parte después. El modelo procesa el documento entero en un solo pase, produciendo un embedding contextualizado por token. Luego se definen los límites de chunk y se hace **mean pooling** sobre los tokens de cada chunk para obtener su vector representativo.

La ventaja clave es que cada token se representa teniendo en cuenta el contexto completo del documento. En el chunking tradicional, el primer token de cada chunk no "ve" el texto anterior porque fue truncado. Con late chunking, un pronombre como "él" en el chunk 3 puede estar correctamente asociado a la entidad que introdujo el chunk 1, porque el modelo procesó ambos juntos.

La limitación obvia es la ventana de contexto del modelo: si el documento supera el máximo de tokens soportado (típicamente 8k–128k según el modelo), no puede procesarse entero. Late chunking solo es aplicable cuando el documento cabe íntegro en el contexto del encoder.

**Concepto DSA subyacente:** pooling sobre particiones de una secuencia de vectores token. El modelo produce una matriz T×D (tokens × dimensiones). Los límites de chunk definen intervalos [a, b] sobre el eje T. El embedding de cada chunk es `mean(embeddings[a:b], axis=0)` — una operación de reducción sobre submatrices.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def late_chunk(text: str, chunk_boundaries: list[int], model_name: str = "BAAI/bge-m3"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: [1, T, D]
    token_embeddings = outputs.last_hidden_state[0]  # [T, D]

    chunk_embeddings = []
    prev = 0
    for boundary in chunk_boundaries + [token_embeddings.shape[0]]:
        chunk_vec = token_embeddings[prev:boundary].mean(dim=0)
        chunk_embeddings.append(chunk_vec.numpy())
        prev = boundary

    return np.array(chunk_embeddings)
```

**Complejidad:** O(T²) atención del transformer sobre el documento completo (el coste dominante). El pooling posterior es O(T·D). Para documentos largos, modelos con atención eficiente (FlashAttention, linear attention) reducen el cuadrático a O(T·log T) o O(T). El resultado es un número de embeddings igual al número de chunks, igual que el enfoque tradicional, pero con mejor calidad de representación.

---

### Tabla comparativa

| Estrategia | Coherencia del chunk | Impacto en tamaño del índice | Calidad de retrieval | Coste computacional |
|---|---|---|---|---|
| Fixed-size chunking | Bajo | Bajo | Bajo | Bajo |
| Chunk overlap | Bajo–Medio | Medio (redundancia) | Medio | Bajo |
| Semantic chunking | Alto | Bajo–Medio | Alto | Medio (embeddings por frase) |
| Chunked semantic embeddings | Medio | Medio | Medio–Alto | Medio |
| Chunked semantic search | Medio | Medio | Alto | Medio |
| Chunked edge cases | Alto | Medio | Medio–Alto | Bajo–Medio |
| ColBERT | Alto | Muy alto (100×) | Muy alto | Alto |
| Late chunking | Muy alto | Bajo (igual al fixed) | Alto | Alto (encoder full-doc) |
