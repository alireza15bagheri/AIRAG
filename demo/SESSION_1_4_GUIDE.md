# Session 1.4 Guide: Vector Databases and ANN Algorithms

**Demo script:** `demo/session_1_4_vector_databases.py`  
**Estimated walkthrough time:** 5–10 minutes  
**Phase:** Phase 1 – Understanding RAG Concepts (Session 1.4 of 5)

---

## What This Demo Teaches

You will learn how a vector database stores and searches millions of text embeddings efficiently. The demo covers:

1. **Brute-force search** — checking every vector (exact but slow)
2. **HNSW graphs** — a fast approximate search method used by real vector databases
3. **Recall vs speed trade-offs** — how search quality changes with settings
4. **ChromaDB** — a real, persistent vector database with metadata filtering

This is the missing link between "embeddings exist" (Session 1.2) and "search works at scale" (Session 1.3). Without a vector database, you cannot search millions of document chunks in a reasonable time.

---

## Prerequisites

Before running, make sure your Python environment has the project dependencies installed:

```bash
# From the project root
.venv\Scripts\activate        # Windows PowerShell
# or
source .venv/Scripts/activate # Git Bash

pip install -r requirements.txt
```

Key packages used by this demo:

| Package | Purpose | Required? |
|---------|---------|-----------|
| `numpy` | Vector math and arrays | Yes |
| `sentence-transformers` | Generate text embeddings | Yes (for full demo) |
| `chromadb` | Real vector database integration | Optional (skipped if missing) |

If `chromadb` is not installed, the demo still runs all pure-Python sections and skips the ChromaDB section gracefully.

---

## How to Run the Demo

From the project root directory (`E:\projects\airag`):

### 1. Run the full demo (recommended first run)

```bash
python demo/session_1_4_vector_databases.py
```

This runs all sections: brute-force search, HNSW concept, recall evaluation, ChromaDB integration, and prints key takeaways.

### 2. Run only the self-tests (fast, no model download)

```bash
python demo/session_1_4_vector_databases.py --self-test
```

Verifies the math and HNSW logic without loading any ML models. Use this to confirm the script works before running the full demo.

### 3. Run a one-shot query

```bash
python demo/session_1_4_vector_databases.py --query "A kitten is sleeping."
```

Searches the demo corpus with your query text using both brute-force and HNSW.

### 4. Run the performance benchmark

```bash
python demo/session_1_4_vector_databases.py --benchmark
```

Compares brute-force vs HNSW search speed at 100, 500, and 1,000 vector scales.

### 5. Enter interactive mode

```bash
python demo/session_1_4_vector_databases.py --interactive
```

Type your own sentences and compare brute-force vs HNSW results side by side. Type `/quit` to exit.

### 6. Combine options

```bash
python demo/session_1_4_vector_databases.py --query "Deep learning" --benchmark --top-k 10
```

Available flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Sentence-Transformers model name | `all-MiniLM-L6-v2` |
| `--top-k` | Number of results to show | `5` |
| `--query` | One-shot query (can repeat) | none |
| `--interactive` | Enter interactive query mode | off |
| `--benchmark` | Run performance benchmark | off |
| `--self-test` | Run fast math checks only, then exit | off |

---

## Section-by-Section Walkthrough

When you run the full demo, it executes five sections in order:

### Section 1: Brute-Force Exact Search

**What it does:** Computes cosine similarity between a query vector and every stored vector, then returns the top-k matches.

**What you'll see:**
- The formula: `cosine(A, B) = (A · B) / (||A|| × ||B||)`
- A ranked list of the 5 most similar documents
- Time taken for the search

**Key idea:** Brute-force is **exact** (always finds the true nearest neighbors) but **O(n)** — every new document adds linear search cost. With 1 million vectors, this is impractical.

### Section 2: HNSW Conceptual Implementation

**What it does:** Builds a simplified Hierarchical Navigable Small World (HNSW) graph index from scratch and searches it.

**What you'll see:**
- HNSW build time and statistics (number of levels, level distribution)
- Sample node connections showing how vectors link to neighbors
- HNSW search results compared to brute-force

**Key idea:** HNSW builds layered graphs:
- **Top layers:** sparse, long-range edges for fast coarse navigation
- **Bottom layer (L0):** dense graph for fine-grained refinement
- Search starts at the top, greedily descends to L0

**Three parameters control the trade-off:**
- `M` — max connections per node per layer
- `ef_construction` — candidate list size during index build (higher = better quality, slower build)
- `ef_search` — candidate list size during search (higher = better recall, slower query)

### Section 3: Recall Evaluation

**What it does:** Compares HNSW results against brute-force ground truth to measure recall.

**What you'll see:**
- For 4 different queries, the brute-force results vs HNSW results
- Recall percentage (how many true neighbors HNSW found)
- A table showing how recall changes as `ef_search` increases (10, 25, 50, 100, 200)

**Key idea:** Higher `ef_search` → better recall but slower queries. You choose the balance based on your application's needs.

### Section 4: ChromaDB Integration

**What it does:** Uses the real ChromaDB vector database to store, query, and filter embeddings.

**What you'll see:**
- Documents added to a persistent ChromaDB collection
- Semantic search results (no filter)
- Semantic search with metadata filter (`topic == "astronomy"`)

**Key idea:** ChromaDB handles indexing, persistence, and metadata filtering automatically. You don't need to implement HNSW yourself — production vector databases do it for you.

### Section 5: Performance Benchmark

**What it does:** (Only with `--benchmark`) Compares brute-force vs HNSW at 100, 500, and 1,000 vectors.

**What you'll see:**
- A table showing brute-force time, HNSW time, and speedup factor at each scale

**Key idea:** HNSW's advantage grows with dataset size. At small scales the difference is negligible; at large scales it's orders of magnitude faster.

---

## Example Output

Here is what a typical full run looks like:

```
================================================================================
  SESSION 1.4: VECTOR DATABASES AND ANN ALGORITHMS
  From brute-force search to HNSW graphs to ChromaDB
================================================================================

DOCUMENT CORPUS
--------------------------------------------------------------------------------
20 documents across categories:

  [ 0] Animals    The cat sat on the warm windowsill.
  [ 1] Animals    A dog barked loudly at the mail carrier.
  ...

1. BRUTE-FORCE EXACT SEARCH (O(n) baseline)
--------------------------------------------------------------------------------
...
Rank   Score      Cat           Document
--------------------------------------------------------------------------------
   1    1.0000   Animals       The cat sat on the warm windowsill.
   2    0.8234   Animals       Kittens chase laser pointers across the floor.
...

2. HNSW (Hierarchical Navigable Small World) — Conceptual
--------------------------------------------------------------------------------
...
Build time: 45.23 ms
Nodes: 20, Max level: 3
Level distribution: {0: 20, 1: 8, 2: 2, 3: 1}

3. RECALL EVALUATION: HNSW vs Brute Force
--------------------------------------------------------------------------------
...
Query [0] 'The cat sat on the warm windowsill.'
  Brute force:   [0, 2, 1, 3, 4]
  HNSW (ef=50):  [0, 2, 1, 3, 4]
  Recall: 100%   Overlap: [0, 2, 1, 3, 4]

...
Effect of ef_search on recall (query [0], top_k=5):
   ef_search      Recall
-------------------------
          10    80%
          25    90%
          50   100%
         100   100%
         200   100%

4. CHROMADB — Production Vector Database
--------------------------------------------------------------------------------
...
Query (metadata filter: topic == 'Space'):
Rank   Distance     Cat           Document
--------------------------------------------------------------------------------
   1    0.1234     astronomy     The rocket launched into orbit.

================================================================================
  SESSION 1.4 COMPLETE
================================================================================

Key Takeaways:
  • Brute-force search is exact but O(n) — doesn't scale to millions of vectors
  • HNSW builds layered graphs for fast approximate nearest-neighbor search
  • ef_search controls recall vs latency trade-off at query time
  • ef_construction controls build quality vs build time
  • ChromaDB wraps HNSW with persistence, metadata filtering, and CRUD
  • Metadata filtering (WHERE) narrows search before vector similarity
  • Production RAG systems combine vector DB + metadata filters + hybrid search
```

---

## Key Concepts in Simple Terms

| Concept | Simple Explanation |
|---------|-------------------|
| **Embedding** | A list of numbers that represents the meaning of a sentence |
| **Vector database** | A specialized database that stores embeddings and finds similar ones quickly |
| **Cosine similarity** | A measure of how similar two embeddings are (1 = identical, 0 = unrelated) |
| **Brute-force search** | Compare the query to every vector one by one (exact but slow) |
| **HNSW** | A graph-based shortcut that finds approximate nearest neighbors quickly |
| **Recall** | What percentage of the true best matches did we find? |
| **Metadata filter** | Narrow the search to only documents matching certain criteria (e.g., topic = "Space") |

---

## Troubleshooting

**`sentence-transformers` not installed**

```
sentence-transformers required. Install: pip install sentence-transformers
```

Fix:
```bash
pip install sentence-transformers
```

**`chromadb` not installed**

The ChromaDB section is skipped automatically. The rest of the demo still works. To enable it:
```bash
pip install chromadb
```

**Script takes a long time on first run**

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). Subsequent runs use the local cache and are much faster.

**No output / script appears to hang**

Make sure you are running from the project root directory (`E:\projects\airag`) and that the virtual environment is activated.

---

## Next Steps

After completing this demo:

1. **Read Session 1.5** — Complete RAG Pipeline Walkthrough (planned next session in Phase 1)
2. **Proceed to Phase 2** — Document Ingestion (multi-format loaders: PDF, DOCX, MD, TXT)
3. **Experiment:** Try different `--ef-search` values in the interactive mode to see the recall/speed trade-off yourself
4. **Explore ChromaDB:** Look at the `demo/output/chroma_demo_db/` folder created by Section 4 to see persisted data on disk

---

## Related Files

- `demo/session_1_1_tokenization.py` — Session 1.1: LLMs, Transformers, and Tokens
- `demo/session_1_2_embeddings.py` — Session 1.2: Text Embeddings and Vector Space
- `demo/session_1_3_semantic_vs_keyword.py` — Session 1.3: Semantic Search vs Keyword Search
- `demo/session_1_4_vector_databases.py` — This script: Vector Databases and ANN Algorithms
