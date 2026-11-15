# Session 1.5 Guide: Complete RAG Pipeline Walkthrough
======================================================

**Demo script:** `demo/session_1_5_rag_pipeline.py`  
**Estimated walkthrough time:** 10–15 minutes  
**Phase:** Phase 1 – Understanding RAG Concepts (Session 1.5 of 5)

---

## What This Demo Teaches

This is the capstone of Phase 1. It wires together all previous sessions into a working **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Document Ingestion** → Chunking → Embedding → Storage (ChromaDB)
2. **Query Processing** → Embedding → Semantic Retrieval (Top-K)
3. **Prompt Construction** → Context Assembly + Citation Markers
4. **LLM Generation** (Ollama) → Answer with Inline Citations

You'll see how each Phase 1 component (tokens, embeddings, search, vector DB) connects to form a complete QA system.

---

## Prerequisites

```bash
# From project root
.venv\Scripts\activate        # Windows PowerShell
# or
source .venv/Scripts/activate # Git Bash

pip install -r requirements.txt
```

Required packages (from `requirements.txt`):
| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Embeddings (all-MiniLM-L6-v2) |
| `chromadb` | Vector database with HNSW |
| `requests` | Ollama HTTP client |

Optional:
| Package | Purpose |
|---------|---------|
| `ollama` (service) | Local LLM for generation. If unavailable, mock LLM is used. |

---

## How to Run

From the project root (`E:\projects\airag`):

### 1. Run the full demo walkthrough
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py
```
Runs ingestion → retrieval → generation → comparison → citations.

### 2. Run a one-shot query
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py --query "What is RAG?"
```

### 3. Run interactively
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py --interactive
```

### 4. Retrieval-only (no LLM)
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py --no-llm
```

### 5. Run self-tests
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py --self-test
```

### 6. Customize parameters
```bash
E:/projects/airag/.venv/Scripts/python.exe demo/session_1_5_rag_pipeline.py \
  --model all-MiniLM-L6-v2 \
  --chunk-size 300 \
  --overlap 50 \
  --top-k 5 \
  --ollama-model llama3.2:3b \
  --persist-dir ./my_chroma_db
```

---

## Section-by-Section Walkthrough

### Section 1: Document Ingestion Pipeline

**What it does:**
1. Takes raw documents (title + content)
2. Splits into overlapping chunks (configurable size/overlap)
3. Generates embeddings for each chunk
4. Stores chunks + embeddings in ChromaDB

**What you'll see:**
```
Created 18 chunks from 5 documents
Generated 18 embeddings in 45.2 ms
Stored 18 chunks in ChromaDB
```

**Key concepts:**
- **Chunking**: Fixed-size with overlap preserves context at boundaries
- **Embeddings**: 384-dim normalized vectors (cosine similarity ready)
- **Storage**: ChromaDB persists to disk, uses HNSW index

---

### Section 2: Semantic Retrieval

**What it does:**
1. Embeds the query using the same model
2. Queries ChromaDB for top-K similar chunks
3. Returns chunks with similarity scores

**What you'll see:**
```
Query: 'What is HNSW?'
Top 3 results:
  1. [0.8234] doc_3 (chunk 0) - Vector Databases and HNSW...
  2. [0.4121] doc_3 (chunk 1) - HNSW builds layered graphs...
  3. [0.3876] doc_4 (chunk 2) - Semantic chunking uses...
```

**Key concepts:**
- **Same embedding model** for queries and documents (critical for alignment)
- **Cosine similarity**: 1.0 = identical, 0.0 = unrelated
- **Top-K**: Trade-off between recall and context window size

---

### Section 3: Answer Generation with LLM

**What it does:**
1. Builds prompt: `System + Context + Query`
2. Sends to Ollama (or mock LLM)
3. Returns answer with inline citations `[1], [2]`

**What you'll see:**
```
QUERY: What is RAG?

ANSWER:
RAG (Retrieval-Augmented Generation) combines information retrieval with 
text generation [1]. It retrieves relevant documents from a knowledge base 
and uses them as context for generating answers, reducing hallucinations [2].

CITATIONS:
  [1] doc_1 (chunk 0) - score: 0.9234
      Retrieval-Augmented Generation (RAG) is an AI technique that combines...
  [2] doc_1 (chunk 1) - score: 0.8912
      Instead of relying solely on the model's training data...
```

**Key concepts:**
- **Citations**: `[1]`, `[2]` map to retrieved chunks for verification
- **Context window**: Only top-K chunks fit in prompt
- **Temperature**: Low (0.1) for factual accuracy

---

### Section 4: RAG vs No-RAG Comparison

**What it does:**
Runs the same query with and without retrieval context.

**What you'll see:**
```
Query: 'How does HNSW work in vector databases?'

WITH RAG (context from ChromaDB):
HNSW builds layered graphs for fast approximate nearest neighbor search. 
The top layers have sparse connections for coarse navigation [1]. 
The bottom layer has dense connections for fine-grained search [2].

WITHOUT RAG (model knowledge only):
HNSW is a graph-based algorithm for approximate nearest neighbor search. 
It builds a hierarchical graph structure... (generic, no specifics)
```

**Key insight:** RAG provides specific, cited, up-to-date answers. Pure LLM gives generic, possibly outdated answers.

---

### Section 5: Citation Tracking

**What it does:**
Shows detailed mapping between answer claims and source chunks.

**What you'll see:**
```
Citation mapping:
  Source [1]: doc_3 (similarity: 0.8234)
  Text: HNSW builds layered graphs for fast approximate nearest neighbor search...
  
  Source [2]: doc_3 (chunk 1) (similarity: 0.4121)
  Text: The top layers have sparse connections for coarse navigation...
```

**Key concept:** Citations enable **verification** — users can check sources.

---

### Interactive Mode

Commands available:
| Command | Action |
|---------|--------|
| `<sentence>` | Run full RAG pipeline |
| `/norag <query>` | Generate without retrieval |
| `/topk <n>` | Set retrieval count |
| `/help` | Show commands |
| `/quit` | Exit |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INGESTION (offline)              QUERY TIME (online)              │
│  ─────────────────────             ─────────────────────           │
│                                                                     │
│  Documents                         User Query                       │
│       │                                │                            │
│       ▼                                ▼                            │
│  ┌─────────┐                     ┌─────────────┐                   │
│  │ Chunking │                     │  Embedding  │                   │
│  │ (size,   │                     │  (same model)                │
│  │  overlap)│                     └──────┬──────┘                   │
│  └────┬─────┘                           │                            │
│       │                                 ▼                            │
│       ▼                        ┌──────────────────┐                 │
│  ┌─────────┐                    │  ChromaDB Query  │                 │
│  │ Embedding│                    │  (HNSW search)   │                 │
│  │(MiniLM)  │                    └────────┬─────────┘                 │
│  └────┬─────┘                             │                           │
│       │                                   ▼                           │
│       ▼                          ┌──────────────────┐                │
│  ┌─────────┐                       │  Top-K Chunks    │               │
│  │ ChromaDB │                       │  + Scores        │               │
│  │(HNSW)    │                       └────────┬─────────┘               │
│  └─────────┘                                │                          │
│                                             ▼                          │
│                                  ┌──────────────────┐                 │
│                                  │  Prompt Builder  │                 │
│                                  │  (Context + Q)   │                 │
│                                  └────────┬─────────┘                 │
│                                           │                            │
│                                           ▼                            │
│                                  ┌──────────────────┐                 │
│                                  │     Ollama LLM   │                 │
│                                  │  (or Mock)       │                 │
│                                  └────────┬─────────┘                 │
│                                           │                            │
│                                           ▼                            │
│                                  ┌──────────────────┐                 │
│                                  │  Answer + Cites  │                 │
│                                  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Classes & Functions

| Class/Function | Purpose |
|----------------|---------|
| `Chunk` | Document chunk with text, metadata, embedding |
| `RetrievedChunk` | Chunk + similarity score |
| `RAGResponse` | Answer + citations + latency |
| `chunk_text()` | Fixed-size overlapping chunking |
| `create_chunks_from_docs()` | Pipeline: docs → chunks with metadata |
| `ChromaVectorStore` | Wrapper: add, query, count |
| `OllamaClient` | HTTP client with mock fallback |
| `build_prompt()` | System + context + query → prompt |
| `run_rag_pipeline()` | End-to-end: query → answer + citations |
| `run_rag_pipeline()` | Core orchestration function |

---

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `all-MiniLM-L6-v2` | Embedding model |
| `--chunk-size` | `200` | Chunk size (chars) |
| `--overlap` | `50` | Chunk overlap (chars) |
| `--top-k` | `3` | Retrieved chunks |
| `--ollama-url` | `http://localhost:11434` | Ollama endpoint |
| `--ollama-model` | `llama3.2:3b` | LLM model |
| `--persist-dir` | `demo/output/chroma_rag_demo` | ChromaDB path |
| `--query` | — | One-shot query |
| `--interactive` | — | Interactive mode |
| `--self-test` | — | Run self-tests |
| `--no-llm` | — | Skip LLM (retrieval only) |

---

## Troubleshooting

**`sentence-transformers` not installed**
```bash
pip install sentence-transformers
```

**`chromadb` not installed**
```bash
pip install chromadb
```

**Ollama not running**
```
Warning: Ollama not available - using mock LLM for demonstration
```
Start Ollama: `ollama serve` and pull model: `ollama pull llama3.2:3b`

**Script hangs on first run**
First run downloads embedding model (~90 MB). Subsequent runs use cache.

**ChromaDB persistence**
Use `--persist-dir` to keep data between runs. Default: `demo/output/chroma_rag_demo/`

---

## Next Steps

After completing Session 1.5:

1. **Phase 2: Document Ingestion** — Production loaders (PDF, DOCX, MD, TXT), metadata preservation, error handling
2. **Phase 3: Text Chunking** — Configurable strategies (fixed, sentence-aware, semantic), visualization
3. **Phase 4: Embeddings** — Model comparison, batch processing, dimension analysis
4. **Phase 8: FastAPI REST API** — Document management, QA endpoints, OpenAPI docs
5. **Phase 9: Frontend** — React chat interface with citations

---

## Related Files

- `demo/session_1_1_tokenization.py` — Tokens & tokenizers
- `demo/session_1_2_embeddings.py` — Embeddings & vector space
- `demo/session_1_3_semantic_vs_keyword.py` — BM25 vs Dense vs Hybrid
- `demo/session_1_4_vector_databases.py` — HNSW & ChromaDB
- `demo/session_1_5_rag_pipeline.py` — **This script: Complete RAG pipeline**
- `README.md` — Project roadmap (Phase 1.5 = this session)

---

## Summary

This demo shows that **RAG is not magic** — it's a pipeline of well-understood components:

1. **Chunk** → respect boundaries, preserve context
2. **Embed** → same model for queries & docs
3. **Store** → ChromaDB with HNSW for scale
4. **Retrieve** → semantic similarity, top-K
5. **Prompt** → context + query + citation markers
6. **Generate** → LLM + citations → verifiable answers

Each component is swappable and testable independently. That's the power of modular RAG architecture.