# RAG Search Engine

Un motor de búsqueda construido desde cero para entender los fundamentos detrás de la Generación Aumentada por Recuperación (RAG). El proyecto se desarrolla de forma incremental — cada módulo introduce un concepto nuevo, desde una búsqueda lineal ingenua hasta puntuaciones TF-IDF sobre un índice invertido.

El dataset es una colección de películas (título + descripción). El motor permite indexarlas, buscar por palabras clave y calcular puntuaciones de relevancia.

---

## Estructura del proyecto

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py      # Punto de entrada del CLI
│   ├── data_loader.py             # Carga películas y stopwords desde disco
│   ├── tokenizer.py               # Pipeline de normalización y stemming
│   └── search_engines/
│       ├── linear_search.py       # Búsqueda ingenua O(n) (deprecada)
│       └── inverted_index.py      # Búsqueda por índice + TF-IDF
├── data/
│   ├── movies.json                # Corpus de películas (título + descripción)
│   └── stopwords.txt              # Palabras a ignorar durante la indexación
└── cache/                         # Archivos del índice persistido (auto-generado)
    ├── index.pkl
    ├── docmap.pkl
    └── term_frequencies.pkl
```

---

## Conceptos

### 1. Tokenización

Antes de poder buscar nada, el texto en crudo tiene que convertirse a una forma normalizada y comparable. La clase `Tokenizer` ([cli/tokenizer.py](cli/tokenizer.py)) aplica un pipeline de cuatro pasos:

```
texto crudo → normalizar → partir → eliminar stopwords → stemming
```

**Paso 1 — Normalizar:** Convierte a minúsculas y elimina la puntuación.
`"Spider-Man: No Way Home!" → "spiderman no way home"`

**Paso 2 — Partir:** Divide en tokens (palabras individuales).
`["spiderman", "no", "way", "home"]`

**Paso 3 — Eliminar stopwords:** Filtra las palabras tan comunes que no aportan significado (*the*, *a*, *no*, *is*…). Se cargan desde `data/stopwords.txt`.
`["spiderman", "way", "home"]`

**Paso 4 — Stemming:** Reduce cada palabra a su raíz. Así *running*, *runs* y *ran* producen el mismo token (`run`) y todos coinciden con la misma entrada en el índice.
`["spiderman", "wai", "home"]`

El mismo pipeline se aplica de forma idéntica a los documentos al indexar y a la consulta al buscar. Esa consistencia es lo que hace que las búsquedas funcionen aunque el usuario escriba la palabra en una forma distinta a la que aparece en el texto.

---

### 2. Stopwords

Las stopwords son palabras extremadamente frecuentes (*el*, *la*, *de*, *en*, *un*, *y*…) que aparecen en prácticamente cualquier documento. Incluirlas en el índice causaría dos problemas:

- **Desperdicio de memoria:** mapearían casi a todos los documentos.
- **Resultados ruidosos:** una búsqueda como *"the matrix"* devolvería todos los documentos solo por la palabra *"the"*.

Al filtrarlas en el pipeline de tokenización el índice queda compacto y los resultados son relevantes.

---

### 3. El algoritmo de Porter Stemming

El stemming es el proceso de reducir una palabra a su raíz o forma base. El **Porter Stemmer** (1980) aplica una serie de reglas para eliminar sufijos:

| Palabra original | Raíz (stem) |
|------------------|-------------|
| running          | run         |
| movies           | movi        |
| historical       | histor      |
| searching        | search      |

Las raíces no siempre son palabras reales — lo importante es que todas las formas relacionadas de una palabra produzcan la misma raíz, tanto al indexar como al buscar.

---

### 4. Búsqueda lineal (el enfoque ingenuo)

La primera implementación ([cli/search_engines/linear_search.py](cli/search_engines/linear_search.py)) recorre todas las películas del dataset para cada consulta:

```python
for movie in self.movies:
    title_tokens = self.tokenizer.tokenize(movie["title"])
    if any(q in t for q in query_tokens for t in title_tokens):
        results.append(movie)
```

**Complejidad temporal: O(n)** — cada búsqueda lee todos los documentos.

Funciona bien con datasets pequeños, pero es inviable a escala (millones de documentos). Además solo busca en títulos, no en descripciones. Por eso está marcada como `deprecated` en el código.

---

### 5. Índice invertido

El **índice invertido** es la estructura de datos central de cualquier motor de búsqueda real (Google, Elasticsearch y Lucene se basan en este mismo concepto).

En lugar de preguntar *"¿contiene este documento la palabra?"* para cada documento, precalculamos la respuesta en una fase de indexación y la guardamos como:

```
token → {conjunto de IDs de documentos que lo contienen}
```

**Ejemplo:**

| Token       | IDs de documentos     |
|-------------|-----------------------|
| "spiderman" | {12, 47, 203}         |
| "histor"    | {5, 47, 98, 312}      |
| "wai"       | {12, 98}              |

En tiempo de búsqueda, buscar una palabra es una consulta O(1) al diccionario — sin escaneos.

La clase `InvertedIndex` ([cli/search_engines/inverted_index.py](cli/search_engines/inverted_index.py)) mantiene tres estructuras internas:

| Atributo           | Tipo                          | Para qué sirve                                  |
|--------------------|-------------------------------|--------------------------------------------------|
| `index`            | `dict[str, set[int]]`         | Token → conjunto de IDs de documentos que lo contienen |
| `docmap`           | `dict[int, dict]`             | ID de documento → objeto película completo      |
| `term_frequencies` | `dict[int, Counter]`          | ID de documento → frecuencia de cada término    |

**Fase de construcción (build):** Se procesa el corpus entero una sola vez. El título y la descripción de cada película se tokenizan juntos y se añaden al índice.

**Fase de búsqueda (search):** Los tokens de la consulta se buscan en el índice (lógica OR — cualquier coincidencia devuelve el documento). Los resultados se ordenan por ID y se limitan a 5.

**Persistencia:** El índice se serializa a disco con `pickle` en la carpeta `cache/`, así solo hay que construirlo una vez.

---

### 6. TF-IDF (Frecuencia de Término – Frecuencia Inversa de Documento)

Una vez que podemos recuperar documentos, la siguiente pregunta es: **¿cuáles son más relevantes?** TF-IDF es la medida estadística clásica para responder esto.

#### Frecuencia de Término (TF)

¿Cuántas veces aparece el término en un documento concreto?

```
TF(término, doc) = número de veces que aparece el término en ese doc
```

Un término que aparece 5 veces en la descripción de una película es más relevante para esa película que uno que aparece solo una vez.

#### Frecuencia Inversa de Documento (IDF)

¿Qué tan rara es la palabra en todo el corpus?

```
IDF(término) = log( (total_docs + 1) / (docs_que_contienen_el_término + 1) )
```

- Una palabra que aparece en todos los documentos tiene un IDF casi cero → poco peso.
- Una palabra que aparece en solo 2 de 10.000 documentos tiene un IDF alto → mucho peso.

El `+1` en numerador y denominador es el **suavizado de Laplace**, que evita la división por cero y valores extremos para términos muy raros.

#### Puntuación TF-IDF

```
TF-IDF(término, doc) = TF × IDF
```

La puntuación es alta cuando:
- El término aparece **muchas veces en este documento** (TF alto), Y
- El término es **raro en el corpus** (IDF alto)

Es baja cuando el término está ausente del documento o es tan común que aparece en todos lados.

El CLI expone las tres métricas como comandos independientes para que puedas inspeccionar los valores paso a paso.

---

## Instalación

```bash
# Instalar dependencias con uv
uv sync

# Construir el índice invertido (solo una vez)
keyword-search build
```

---

## Uso

```bash
# Buscar películas
keyword-search search "space adventure"

# Frecuencia de "action" en el documento 42
keyword-search tf 42 action

# Frecuencia inversa de "robot" en el corpus
keyword-search idf robot

# Puntuación TF-IDF de "war" en el documento 7
keyword-search tfidf 7 war
```

---

## Siguientes pasos (hacia RAG completo)

Este proyecto construye la mitad de **recuperación** de un sistema RAG. El pipeline hasta ahora:

```
Consulta → Tokenizar → Búsqueda en índice → Ranking TF-IDF → Top-K documentos
```

En un RAG completo, esos top-K documentos recuperados se pasan como contexto a un modelo de lenguaje (Claude, GPT…), que genera una respuesta fundamentada en la evidencia recuperada, en lugar de inventarse información desde su memoria paramétrica.

Los pasos siguientes serían:

- Reemplazar la búsqueda por palabras clave con **búsqueda vectorial densa** (embeddings + similitud coseno)
- Añadir un **re-ranker** para mejorar el orden de los resultados
- Pasar los mejores resultados a un **prompt de LLM** para la generación de respuestas

---

## Apuntes del curso

Resumen de los conceptos clave de cada módulo. Sirve como referencia rápida para no tener que volver a los vídeos.

---

### Módulo 1 — Preprocesamiento

El preprocesamiento convierte texto crudo en una representación que el sistema puede comparar y analizar. Es el paso más fundamental: si el texto entra mal, todo lo que venga después falla.

**Conceptos clave:**

- **Normalización:** pasar a minúsculas, eliminar puntuación y caracteres especiales. Garantiza que `"Matrix"` y `"matrix"` se traten igual.
- **Tokenización:** dividir el texto en unidades mínimas (tokens). Puede ser a nivel de palabra, subpalabra o carácter.
- **Stopwords:** palabras tan frecuentes que no aportan señal (*el, la, de, en, un*…). Se eliminan para reducir ruido y tamaño del índice.
- **Stemming:** reducción de una palabra a su raíz mediante reglas de sufijos. Rápido pero produce raíces que no siempre son palabras reales (`"correr" → "corr"`).
- **Lematización:** reducción a la forma canónica del diccionario. Más lento que el stemming pero más preciso (`"corriendo" → "correr"`). Requiere conocimiento del idioma.
- **Porter Stemmer:** algoritmo de stemming más utilizado (1980). Aplica reglas en cascada para eliminar sufijos en inglés.

**Diferencia clave stemming vs lematización:**

| | Stemming | Lematización |
|---|---|---|
| Velocidad | Rápido | Más lento |
| Precisión | Aproximada | Exacta |
| Resultado | Raíz artificial | Forma del diccionario |
| Ejemplo | `"studies" → "studi"` | `"studies" → "study"` |

---

### Módulo 2 — TF-IDF

Mide la relevancia de un término para un documento dentro de un corpus. Es la base del ranking en búsqueda por palabras clave.

**Conceptos clave:**

- **TF (Term Frequency):** cuántas veces aparece el término en el documento. A más repeticiones, más relevante para ese documento.
- **IDF (Inverse Document Frequency):** qué tan raro es el término en todo el corpus. A más documentos lo contienen, menos discriminatorio es.
  ```
  IDF(t) = log( (N + 1) / (df(t) + 1) )
  ```
  donde `N` = total de documentos, `df(t)` = documentos que contienen el término.
- **Suavizado de Laplace (`+1`):** evita divisiones por cero y suaviza los valores extremos.
- **TF-IDF = TF × IDF:** el score final es alto cuando el término es frecuente en el documento pero raro en el corpus.
- **Limitación principal:** TF-IDF no entiende el significado. `"coche"` y `"automóvil"` son completamente distintos para él, aunque signifiquen lo mismo.

---

### Módulo 3 — Keyword Search (Búsqueda por Palabras Clave)

Sistemas de recuperación basados en coincidencia exacta de términos.

**Conceptos clave:**

- **Búsqueda lineal O(n):** recorre todos los documentos en cada consulta. Simple pero inescalable.
- **Índice invertido:** estructura `token → {doc_ids}` que convierte la búsqueda en una consulta O(1) al diccionario. Base de todos los motores de búsqueda reales.
- **Lógica booleana:** operaciones AND (intersección de conjuntos), OR (unión), NOT (diferencia) sobre los posting lists del índice.
- **BM25 (Best Match 25):** evolución de TF-IDF que resuelve un problema de TF clásico: el TF puro no tiene límite. Un término que aparece 100 veces puntúa 100× más que uno que aparece una, aunque la ganancia de relevancia real se agota mucho antes. BM25 añade **saturación de frecuencia** y normalización por longitud del documento. Es el estándar en búsqueda por palabras clave moderna.
  - **BM25 TF (saturación):** limita el crecimiento del score a medida que aumenta la frecuencia. El parámetro `k1` (por defecto `1.5`) controla la velocidad de saturación — el score se acerca asintóticamente a `k1 + 1` sin importar cuántas veces aparezca el término.
    ```
    BM25_TF(t, doc) = (tf × (k1 + 1)) / (tf + k1)
    ```
    | tf  | BM25_TF (k1=1.5) |
    |-----|-----------------|
    | 1   | 1.00            |
    | 2   | 1.40            |
    | 5   | 1.67            |
    | 10  | 1.77            |
    | 100 | 1.97            |
  - **BM25 IDF:** penaliza los términos frecuentes más agresivamente que el IDF clásico.
    ```
    BM25_IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
    ```
- **Posting list:** lista de documentos asociada a cada término en el índice invertido.

**Limitación:** solo encuentra lo que el usuario escribe textualmente. No entiende sinónimos ni contexto semántico.

---

### Módulo 4 — Búsqueda Semántica

En lugar de buscar palabras exactas, busca por **significado**. Dos frases con palabras distintas pero semántica similar se consideran cercanas.

**Conceptos clave:**

- **Embedding:** representación numérica de un texto como un vector de alta dimensión (ej. 768 o 1536 dimensiones). Textos semánticamente similares tienen vectores cercanos en ese espacio.
- **Modelo de embeddings:** red neuronal entrenada para producir estos vectores. Ejemplos: `text-embedding-ada-002` (OpenAI), `all-MiniLM-L6-v2` (sentence-transformers).
- **Similitud coseno:** medida de cercanía entre dos vectores. Mide el ángulo entre ellos (no la distancia). Valor entre -1 y 1, donde 1 = idénticos.
  ```
  cos(A, B) = (A · B) / (|A| × |B|)
  ```
- **Dense retrieval:** recuperación densa — todos los documentos están representados como vectores y la búsqueda es una comparación vectorial, no de palabras.
- **Vector store / base de datos vectorial:** base de datos optimizada para almacenar y consultar vectores eficientemente. Ejemplos: Pinecone, Qdrant, Chroma, Weaviate, pgvector.
- **ANN (Approximate Nearest Neighbors):** algoritmos como HNSW o FAISS que encuentran los vectores más cercanos de forma aproximada pero muy rápida, sin comparar contra todos los vectores.

**Ventaja sobre keyword search:** entiende sinónimos, paráfrasis y contexto. `"coche"` y `"automóvil"` quedan cerca en el espacio vectorial.

---

### Módulo 5 — Chunking

Los LLMs tienen una ventana de contexto limitada. No puedes meterle un libro entero. El chunking es la estrategia para dividir documentos grandes en trozos manejables antes de indexarlos.

**Conceptos clave:**

- **Chunk:** fragmento de texto que se indexa y recupera como unidad.
- **Chunk size:** número de tokens (no caracteres) por chunk. Valor típico: 256–512 tokens.
- **Overlap (solapamiento):** zona compartida entre chunks consecutivos para no perder contexto en los cortes.
  ```
  [--- chunk 1 ---]
              [--- chunk 2 ---]
                         [--- chunk 3 ---]
  ```
- **Estrategias de chunking:**
  - *Fixed-size:* corte cada N tokens. Simple pero puede partir frases a la mitad.
  - *Por oraciones / párrafos:* respeta la estructura natural del texto.
  - *Semántico:* agrupa frases similares. Más costoso pero produce chunks más coherentes.
  - *Recursivo:* intenta primero cortar por párrafo, luego por frase, luego por palabra — el empleado en LangChain por defecto.
- **El problema del chunk size:** chunks pequeños → más precisión en la recuperación, menos contexto para el LLM. Chunks grandes → más contexto, pero se recuperan cosas irrelevantes. No hay talla única.
- **Metadata:** guardar junto al chunk de dónde viene (título del documento, página, sección) para poder citarlo en la respuesta.

---

### Módulo 6 — Hybrid Search (Búsqueda Híbrida)

Combina búsqueda por palabras clave (sparse) con búsqueda semántica (dense) para aprovechar las fortalezas de ambas.

**Conceptos clave:**

- **Sparse retrieval:** búsqueda tradicional basada en palabras exactas (BM25, TF-IDF). Buena para términos técnicos, nombres propios, códigos.
- **Dense retrieval:** búsqueda semántica por embeddings. Buena para conceptos, sinónimos, preguntas en lenguaje natural.
- **Fusion:** mecanismo para combinar las listas de resultados de ambos métodos en un ranking único.
- **RRF — Reciprocal Rank Fusion:** el método de fusión más popular. Combina rankings sin necesitar que los scores sean comparables entre sí.
  ```
  RRF_score(doc) = Σ 1 / (k + rank_en_cada_lista)
  ```
  donde `k` suele ser 60. El documento que aparece alto en ambas listas gana.
- **Alpha (α):** alternativa a RRF — promedio ponderado de scores normalizados. `α=1` es puro dense, `α=0` es puro sparse.

**Cuándo usar cada uno:**
| Caso | Mejor opción |
|---|---|
| Nombres propios, SKUs, códigos | Keyword (sparse) |
| Preguntas en lenguaje natural | Semántica (dense) |
| Uso general en producción | Hybrid |

---

### Módulo 7 — LLMs (Modelos de Lenguaje Grande)

Los modelos que generan la respuesta final en un sistema RAG. Entienden contexto, siguen instrucciones y sintetizan información.

**Conceptos clave:**

- **Transformer:** arquitectura base de todos los LLMs modernos (BERT, GPT, Claude…). Usa mecanismos de atención para relacionar palabras entre sí en el texto.
- **Atención (Attention):** mecanismo que permite al modelo ponderar qué partes del texto son más relevantes para generar cada siguiente token.
- **Token:** unidad mínima de texto para el modelo. No es exactamente una palabra — puede ser una sílaba, una palabra o un signo de puntuación. Aproximación: 1 token ≈ 0.75 palabras en inglés.
- **Ventana de contexto (context window):** número máximo de tokens que el modelo puede procesar a la vez (entrada + salida). Define cuánto texto puedes pasarle.
- **Prompt:** instrucción o texto de entrada que guía al modelo. En RAG, el prompt incluye el contexto recuperado + la pregunta del usuario.
- **System prompt:** instrucciones de comportamiento que se dan al modelo antes de la conversación. Define su rol, tono y restricciones.
- **Temperature:** controla la aleatoriedad de las respuestas. `0` = determinista y conservador, `1+` = más creativo y variado.
- **Parámetros (pesos):** valores numéricos aprendidos durante el entrenamiento. Un modelo de 7B tiene 7.000 millones de parámetros.
- **Memoria paramétrica:** el conocimiento que el modelo tiene "de serie" por su entrenamiento — a diferencia del contexto que le pasamos nosotros (memoria no paramétrica).
- **Alucinación:** cuando el modelo genera información falsa con total confianza. RAG reduce este problema al darle contexto con la información correcta.

---

### Módulo 8 — Reranking

Los primeros resultados recuperados (por keyword o semántica) no siempre son los más relevantes. El reranker los reordena con un modelo más potente.

**Conceptos clave:**

- **Bi-encoder:** arquitectura usada en el retrieval inicial. Codifica la query y cada documento por separado → comparación rápida pero menos precisa.
  ```
  score = cosine_similarity(embed(query), embed(doc))
  ```
- **Cross-encoder:** arquitectura usada para reranking. Recibe query + documento juntos → atención cruzada entre ambos → mucho más preciso pero más lento.
  ```
  score = model(query + doc)  # los procesa juntos
  ```
- **Pipeline típico:**
  ```
  Retrieval (top-100 docs, bi-encoder rápido)
      → Reranker (top-5 docs, cross-encoder preciso)
          → LLM (genera respuesta)
  ```
- **Por qué no usar el cross-encoder para todo:** es demasiado lento para comparar una query contra millones de documentos. Se usa solo en la fase final sobre un conjunto reducido.
- **Modelos populares:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, Cohere Rerank, `bge-reranker`.

---

### Módulo 9 — Evaluación

Sin métricas no puedes saber si tu sistema mejora o empeora. Evaluar un sistema RAG es más complejo que evaluar un clasificador simple.

**Conceptos clave:**

**Métricas de recuperación (retrieval):**
- **Precision@K:** de los K documentos recuperados, ¿qué fracción es relevante?
- **Recall@K:** de todos los documentos relevantes que existen, ¿qué fracción está entre los K recuperados?
- **MRR (Mean Reciprocal Rank):** promedio del recíproco del rango del primer resultado relevante. Penaliza que el resultado correcto aparezca tarde.
- **NDCG (Normalized Discounted Cumulative Gain):** tiene en cuenta el orden de los resultados y su grado de relevancia, no solo si son relevantes o no.

**Métricas de generación (RAG-específicas):**
- **Faithfulness (fidelidad):** ¿la respuesta está respaldada por el contexto recuperado? Mide alucinaciones.
- **Answer relevance:** ¿la respuesta responde realmente a la pregunta?
- **Context relevance:** ¿los chunks recuperados son realmente útiles para responder?

**Frameworks de evaluación:**
- **RAGAS:** framework open-source para evaluar sistemas RAG automáticamente usando un LLM como juez.
- **LLM-as-a-judge:** usar un modelo como Claude o GPT-4 para puntuar la calidad de las respuestas. Escalable pero puede tener sesgos.
- **Golden dataset:** conjunto de preguntas con respuestas de referencia correctas, creadas manualmente. Necesario para una evaluación fiable.

---

### Módulo 10 — Augmented Generation (Generación Aumentada)

La pieza final que une la recuperación con la generación: cómo construir el prompt y cómo el LLM usa el contexto para responder.

**Conceptos clave:**

- **RAG pipeline completo:**
  ```
  Pregunta del usuario
      → Retrieval (recuperar chunks relevantes)
      → Augmentation (construir prompt con contexto)
      → Generation (LLM genera respuesta)
  ```
- **Prompt de RAG típico:**
  ```
  Eres un asistente. Responde SOLO usando el contexto proporcionado.
  Si no encuentras la respuesta en el contexto, di que no lo sabes.

  Contexto:
  [chunk 1]
  [chunk 2]
  [chunk 3]

  Pregunta: {pregunta del usuario}
  ```
- **Grounding:** el proceso de anclar la respuesta del LLM a fuentes concretas. RAG es una técnica de grounding.
- **Citation (citas):** hacer que el LLM indique de qué chunk proviene cada parte de su respuesta. Requiere añadir metadata al contexto.
- **Lost in the middle:** fenómeno documentado donde los LLMs tienden a ignorar información que está en el centro del contexto. Los chunks más relevantes deben ir al principio o al final del prompt.
- **Query rewriting:** antes de buscar, reescribir la pregunta del usuario para mejorar el retrieval. Ej: expandir acrónimos, añadir contexto de conversaciones previas.
- **Naive RAG vs Advanced RAG:** el pipeline básico funciona, pero técnicas avanzadas (HyDE, step-back prompting, multi-query) mejoran notablemente la calidad.

---

### Módulo 11 — Agentic (Sistemas Agénticos)

En lugar de un pipeline fijo, un agente decide dinámicamente qué herramientas usar y en qué orden para resolver una tarea.

**Conceptos clave:**

- **Agente:** LLM que puede usar herramientas (tools) y tomar decisiones sobre qué hacer a continuación en función de los resultados.
- **Tool / herramienta:** función que el agente puede llamar. Ejemplos: búsqueda web, calculadora, base de datos vectorial, API externa.
- **ReAct (Reasoning + Acting):** patrón de agente que alterna entre razonar sobre el problema y ejecutar una acción:
  ```
  Thought: necesito buscar X
  Action: search("X")
  Observation: [resultado]
  Thought: con esto puedo responder
  Answer: ...
  ```
- **Loop de razonamiento:** el agente puede iterar múltiples pasos antes de dar una respuesta final, en lugar de responder en un solo pase.
- **Agentic RAG:** sistema donde el agente decide cuándo buscar, qué buscar, y si los resultados son suficientes o hay que buscar más.
- **Multi-agent:** varios agentes especializados que colaboran. Uno recupera, otro verifica, otro genera.
- **Tool calling / function calling:** mecanismo estándar de la API para que el LLM indique qué herramienta quiere usar y con qué argumentos.
- **Riesgo principal:** los agentes pueden entrar en loops, usar herramientas incorrectamente o acumular errores a lo largo de los pasos. La evaluación y los límites de iteración son importantes.

---

### Módulo 12 — Multimodal

Extender el sistema RAG más allá del texto para incluir imágenes, audio, vídeo u otros tipos de datos.

**Conceptos clave:**

- **Multimodal:** capacidad de un modelo para procesar y generar múltiples tipos de datos (texto + imagen, texto + audio…).
- **Multimodal embeddings:** vectores que representan texto e imágenes en el mismo espacio. Permiten buscar imágenes con texto o texto con imágenes.
- **CLIP (Contrastive Language-Image Pretraining):** modelo de OpenAI que entrena embeddings de texto e imagen juntos. Punto de partida del multimodal moderno.
- **Vision-Language Models (VLM):** modelos que entienden imágenes como entrada junto al texto. Ejemplos: GPT-4o, Claude 3.5, LLaVA, Gemini.
- **Multimodal RAG:** sistema que puede recuperar no solo chunks de texto sino también imágenes, tablas, gráficos o fragmentos de audio relevantes para la pregunta.
- **OCR (Optical Character Recognition):** extracción de texto de imágenes o PDFs escaneados. Necesario para indexar documentos con imágenes que contienen texto.
- **Procesamiento de PDFs complejos:** documentos con tablas, fórmulas o gráficos requieren estrategias especiales — no basta con extraer el texto plano.
