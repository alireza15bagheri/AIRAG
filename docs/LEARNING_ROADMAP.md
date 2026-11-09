# Learning Roadmap

## Overview

This document outlines the structured learning path for the AIRAG project. Each phase builds upon the previous one, combining theoretical understanding with hands-on implementation.

**Current Phase:** Phase 1 – Understanding RAG Concepts

---

## Phase 0: Project Foundation ✅ Complete

| Objective | Status |
|-----------|--------|
| Repository initialized with Git remote | ✅ |
| Python 3.15 virtual environment | ✅ |
| Dependency management (pyproject.toml + requirements.txt) | ✅ |
| Configuration system (Pydantic Settings) | ✅ |
| Source package scaffold (src/airag/) | ✅ |
| All 70+ dependencies installed & verified | ✅ |
| Comprehensive .gitignore | ✅ |
| README.md with project documentation | ✅ |

---

## Phase 1: Understanding RAG Concepts 📋 In Progress

**Goal:** Build deep conceptual understanding of RAG foundations through interactive explanations and code demonstrations.

**Format:** 5 sessions, each with concept explanation + tiny code demo + Q&A

**Prerequisites:** Python environment ready, packages installed

### Session 1.1: LLMs, Transformers, and Tokens

**Learning Objectives:**
- [ ] Explain what an LLM is and how it differs from traditional NLP models
- [ ] Describe Transformer architecture (attention, feed-forward, layer stacking)
- [ ] Understand tokenization: BPE, WordPiece, Unigram algorithms
- [ ] Explain context windows and their impact on RAG design
- [ ] Calculate token counts and estimate costs

**Hands-on Exercises:**
- [ ] Exercise 1.1.1: Tokenize 5 diverse texts using `tiktoken` (cl100k_base)
- [ ] Exercise 1.1.2: Compare tokenizers (GPT-4 vs LLaMA-3) on same input
- [ ] Exercise 1.1.3: Count tokens in sample documents; estimate context usage
- [ ] Exercise 1.1.4: Visualize token boundaries for a mixed-language paragraph

**Verification Questions:**
- [ ] What is the relationship between tokens and words? (roughly 1 token ≈ 0.75 words)
- [ ] Why does context window size determine chunking strategy?
- [ ] How do special tokens (BOS, EOS, PAD) affect model input?

---

### Session 1.2: Embeddings and Vector Space

**Learning Objectives:**
- [ ] Define embeddings: dense vector representations of semantic meaning
- [ ] Compare embedding models: MiniLM (384-dim), MPNet (768-dim), E5 (768-dim), BGE (768-dim)
- [ ] Explain cosine similarity, Euclidean distance, dot product
- [ ] Understand dimensionality trade-offs (quality vs speed vs storage)
- [ ] Explain Matryoshka Representation Learning (MRL) for adaptive dimensions

**Hands-on Exercises:**
- [ ] Exercise 1.2.1: Embed 20 sentences using `all-MiniLM-L6-v2`; compute similarity matrix
- [ ] Exercise 1.2.2: Visualize embeddings in 2D using PCA/t-SNE; interpret clusters
- [ ] Exercise 1.2.3: Compare 3 embedding models on same semantic similarity task
- [ ] Exercise 1.2.4: Batch embed 1000 chunks; measure throughput (docs/sec)

**Verification Questions:**
- [ ] Why does cosine similarity work better than Euclidean for normalized embeddings?
- [ ] What happens to retrieval quality when reducing dimensions from 768 to 384?
- [ ] How does batch size affect embedding throughput and memory usage?

---

### Session 1.3: Semantic Search vs Keyword Search

**Learning Objectives:**
- [ ] Explain BM25 algorithm: TF, IDF, document length normalization
- [ ] Contrast keyword matching (exact terms) vs semantic matching (conceptual meaning)
- [ ] Identify failure cases for each approach
- [ ] Understand hybrid search: score fusion, Reciprocal Rank Fusion (RRF)
- [ ] Know when to use each strategy

**Hands-on Exercises:**
- [ ] Exercise 1.3.1: Implement BM25 search using `rank-bm25` on sample corpus
- [ ] Exercise 1.3.2: Same queries → compare BM25 vs vector search results
- [ ] Exercise 1.3.3: Implement hybrid search with α-weighted fusion
- [ ] Exercise 1.3.4: Find queries where semantic wins, and where keyword wins

**Verification Questions:**
- [ ] Why does BM25 fail on "car" vs "automobile" but semantic succeeds?
- [ ] What is the role of IDF in BM25?
- [ ] How does RRF combine ranked lists without score normalization?

---

### Session 1.4: Vector Databases and ANN Algorithms

**Learning Objectives:**
- [ ] Explain HNSW (Hierarchical Navigable Small World) algorithm
- [ ] Understand HNSW parameters: M, ef_construction, ef_search
- [ ] Describe ChromaDB architecture: collections, metadata filtering, persistence
- [ ] Compare pre-filter vs post-filter metadata strategies
- [ ] Know when to use exact search vs ANN

**Hands-on Exercises:**
- [ ] Exercise 1.4.1: Create ChromaDB collection with custom `SentenceTransformerEmbeddingFunction`
- [ ] Exercise 1.4.2: Add 1000 documents with metadata; filter by metadata field
- [ ] Exercise 1.4.3: Benchmark HNSW params: vary M (16, 32, 64) and ef_search (50, 100, 200)
- [ ] Exercise 1.4.4: Test persistence: save, restart, load collection

**Verification Questions:**
- [ ] How does HNSW achieve O(log N) search instead of O(N)?
- [ ] What is the trade-off of increasing `ef_search`?
- [ ] When does metadata pre-filtering outperform post-filtering?

---

### Session 1.5: Complete RAG Pipeline Walkthrough

**Learning Objectives:**
- [ ] Trace end-to-end flow: ingestion → embedding → storage → retrieval → generation
- [ ] Design prompt templates for grounded generation
- [ ] Implement "I don't know" handling when context insufficient
- [ ] Understand basic evaluation: faithfulness, relevance, context precision
- [ ] Know how to swap components (embedding model, LLM, vector DB)

**Hands-on Exercises:**
- [ ] Exercise 1.5.1: Build minimal RAG pipeline in 50 lines (ingest → query → answer)
- [ ] Exercise 1.5.2: Test with 3 different embedding models; compare answer quality
- [ ] Exercise 1.5.3: Swap LLM: test with Ollama local vs simulated API response
- [ ] Exercise 1.5.4: Evaluate faithfulness: does answer cite retrieved context correctly?

**Verification Questions:**
- [ ] What prompt structure ensures the LLM only uses provided context?
- [ ] How do you detect when retrieved context is insufficient?
- [ ] What metrics would you track to evaluate RAG quality over time?

---

## Phase 1 Deliverables

| File | Description | Status |
|------|-------------|--------|
| `docs/LEARNING_ROADMAP.md` | This document | ✅ |
| `docs/RAG_CONCEPTS.md` | Deep-dive reference for all concepts | 📋 |
| `docs/ARCHITECTURE.md` | Architectural decisions from Phase 1 | 📋 |
| `docs/DECISIONS.md` | ADRs for Phase 1 choices | 📋 |

---

## Phase 2 Preview (Not Started)

| Session | Topic |
|---------|-------|
| 2.1 | Document loaders: PDF (PyMuPDF), DOCX (python-docx), Markdown, TXT |
| 2.2 | Text extraction with metadata preservation (page numbers, headers) |
| 2.3 | Error handling for corrupted/unsupported files |
| 2.4 | Unit tests for each loader |

---

## Learning Progress Tracker

```
Phase 0: ████████████████████ 100% ✅
Phase 1: ░░░░░░░░░░░░░░░░░░░░   0%  (5 sessions)
Phase 2: ░░░░░░░░░░░░░░░░░░░░   0%
Phase 3: ░░░░░░░░░░░░░░░░░░░░   0%
...
```

---

## Resources

| Resource | Type | Session |
|----------|------|---------|
| "Attention Is All You Need" (Vaswani et al., 2017) | Paper | 1.1 |
| "Retrieval-Augmented Generation" (Lewis et al., 2020) | Paper | 1.1, 1.5 |
| Sentence Transformers Documentation | Docs | 1.2 |
| ChromaDB Documentation | Docs | 1.4 |
| "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW" | Paper | 1.4 |
| BM25: "The Probabilistic Relevance Framework" (Robertson & Zaragoza) | Paper | 1.3 |

---

*Last Updated: 2026-11-09 | Phase: 1 – Understanding RAG Concepts*