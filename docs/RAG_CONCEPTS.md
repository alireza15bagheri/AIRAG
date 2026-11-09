# RAG Concepts Reference

## Overview

This document serves as a comprehensive reference for all theoretical and technical concepts covered in Phase 1. It consolidates the deep-dive material from each session for future reference.

---

## 1. LLMs and Transformers

### 1.1 What Is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained on vast text corpora to predict the next token in a sequence. Key characteristics:

| Property | Description |
|----------|-------------|
| **Architecture** | Transformer (decoder-only for GPT-style, encoder-decoder for T5, encoder-only for BERT) |
| **Training Objective** | Next-token prediction (causal LM) or masked token prediction (MLM) |
| **Scale** | Parameters: 100M (BERT-base) to 400B+ (GPT-4); Training tokens: billions to trillions |
| **Capabilities** | In-context learning, reasoning, code generation, translation, summarization |

### 1.2 Transformer Architecture

```
Input Tokens → Embedding → Positional Encoding → [Transformer Blocks] → Output Head → Logits
                                                      ↓
                    ┌───────────────────────────────┘
                    ▼
        ┌─────────────────────┐
        │ Transformer Block   │
        ├─────────────────────┤
        │ Multi-Head Attention│  ← "Attention is All You Need"
        │ Add & LayerNorm     │
        │ Feed-Forward Net    │
        │ Add & LayerNorm     │
        └─────────────────────┘
```

**Key Components:**

| Component | Purpose | Formula |
|-----------|---------|---------|
| **Multi-Head Attention** | Capture relationships between all token pairs | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| **Feed-Forward Network** | Process each token independently | `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2` |
| **Layer Normalization** | Stabilize training | `LN(x) = (x - μ)/σ * γ + β` |
| **Residual Connections** | Enable deep networks | `x + Sublayer(LN(x))` |

**Attention Mechanism:**
- **Q (Query):** What am I looking for?
- **K (Key):** What do I contain?
- **V (Value):** What do I actually pass?
- **Multi-Head:** Multiple attention heads in parallel → concatenate → project

### 1.3 Tokenization

Tokenization converts text into integer IDs that the model can process.

| Algorithm | Used By | Characteristics |
|-----------|---------|-----------------|
| **BPE (Byte Pair Encoding)** | GPT-2/3/4, LLaMA, Mistral | Merges frequent byte pairs; handles OOV via bytes |
| **WordPiece** | BERT, DistilBERT | Similar to BPE; merges based on likelihood |
| **Unigram** | T5, mT5, XLNet | Probabilistic; optimizes token inventory |
| **Character-level** | ByT5, CANINE | No vocabulary; each char = token |

**Special Tokens:**
| Token | Purpose | Typical ID |
|-------|---------|------------|
| `<bos>` / `<s>` | Beginning of sequence | 0 or 1 |
| `<eos>` / `</s>` | End of sequence | 2 |
| `<pad>` | Padding for batching | 0 or 1 |
| `<unk>` | Unknown/out-of-vocabulary | 3 |

**Token Count Estimation:**
- English: ~1 token ≈ 0.75 words ≈ 4 characters
- Code: ~1 token ≈ 0.5 words (more symbols)
- Multilingual: Varies by language

### 1.4 Context Window

The **context window** is the maximum number of tokens the model can process at once.

| Model | Context Window | Implications for RAG |
|-------|----------------|---------------------|
| GPT-3.5 | 4,096 | Small chunks, few retrieved docs |
| GPT-4 | 8,192 / 32,768 | Larger chunks, more context |
| GPT-4 Turbo | 128,000 | Entire documents possible |
| LLaMA-3.1 | 128,000 | Large local context |
| Gemini 1.5 | 1,000,000+ | Near-infinite context |

**Impact on RAG Design:**
```
Chunk Size + (Top-K × Chunk Size) + Query + Prompt Template + Answer ≤ Context Window
```

---

## 2. Embeddings and Vector Space

### 2.1 What Are Embeddings?

**Embeddings** are dense vector representations where semantic similarity maps to geometric proximity.

```
Text → [Embedding Model] → Vector ∈ ℝ^d
```

**Properties:**
- **Dense:** Every dimension carries information (unlike one-hot)
- **Fixed-size:** Same dimensionality regardless of input length
- **Semantic:** Similar meaning → small angle (high cosine similarity)

### 2.2 Embedding Models Comparison

| Model | Dimensions | Parameters | Speed | Quality (MTEB) | Best For |
|-------|------------|------------|-------|----------------|----------|
| `all-MiniLM-L6-v2` | 384 | 22M | ⚡⚡⚡ Fast | Good | Baseline, CPU, latency-sensitive |
| `all-mpnet-base-v2` | 768 | 110M | ⚡⚡ Medium | Better | Quality-critical |
| `e5-base-v2` | 768 | 110M | ⚡⚡ Medium | Strong retrieval | Asymmetric tasks (query ≠ doc) |
| `bge-base-en-v1.5` | 768 | 110M | ⚡⚡ Medium | SOTA open | General-purpose retrieval |
| `bge-large-en-v1.5` | 1024 | 335M | ⚡ Slow | Best open | Maximum quality |
| `text-embedding-3-small` | 1536 | - | API | Strong | OpenAI API users |
| `text-embedding-3-large` | 3072 | - | API | Best | OpenAI API, max quality |

**Asymmetric Embeddings (E5, BGE):**
- Query and document embeddings use different prefixes
- `"query: "` prefix for questions
- `"passage: "` prefix for documents
- Improves retrieval for question-answering

### 2.3 Similarity Metrics

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Cosine Similarity** | `A·B / (||A|| ||B||)` | [-1, 1] | Normalized embeddings (standard) |
| **Dot Product** | `A·B` | (-∞, ∞) | Unnormalized; faster if pre-normalized |
| **Euclidean (L2)** | `||A - B||` | [0, ∞) | When magnitude matters |
| **Manhattan (L1)** | `Σ|A_i - B_i|` | [0, ∞) | Sparse vectors |

**For normalized embeddings (standard practice):**
```
Cosine Similarity = Dot Product  (since ||A|| = ||B|| = 1)
```

### 2.4 Dimensionality Trade-offs

| Dimension | Pros | Cons |
|-----------|------|------|
| **Low (256-384)** | Fast search, less storage, lower memory | Some semantic info lost |
| **Medium (512-768)** | Good balance | Standard choice |
| **High (1024+)** | Maximum semantic fidelity | Slow, expensive storage, diminishing returns |

**Matryoshka Representation Learning (MRL):**
- Train once at max dimension
- Truncate to smaller dims at inference
- Maintains quality at reduced dimensions

### 2.5 Embedding Generation Best Practices

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Batch processing for throughput
embeddings = model.encode(
    sentences,           # List[str]
    batch_size=32,       # Tune for GPU memory
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity
)

# Shape: (n_sentences, 384)
```

---

## 3. Retrieval Strategies

### 3.1 Keyword Search: BM25

**BM25 (Best Matching 25)** is a probabilistic ranking function.

**Formula:**
```
score(D, Q) = Σ_{t∈Q} IDF(t) × (f(t,D) × (k1 + 1)) / (f(t,D) + k1 × (1 - b + b × |D|/avgdl))
```

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| `k1` | 1.2 - 2.0 | Term frequency saturation |
| `b` | 0.75 | Document length normalization |
| `IDF(t)` | `log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)` | Term rarity |

**Strengths:** Exact matches, rare terms, structured queries
**Weaknesses:** Synonyms, polysemy, conceptual matching

### 3.2 Semantic Search: Dense Retrieval

```
Query → [Embedding Model] → q_vector
Search: Top-K by similarity(q_vector, doc_vectors)
```

**Approximate Nearest Neighbor (ANN):**
- Exact search: O(N) — too slow for >100K vectors
- ANN: O(log N) or sublinear — small accuracy trade-off

### 3.3 Hybrid Search

Combines sparse (BM25) and dense (vector) retrieval.

**Fusion Methods:**

| Method | Formula | Notes |
|--------|---------|-------|
| **Weighted Sum** | `α × dense_score + (1-α) × sparse_score` | Requires score normalization |
| **Reciprocal Rank Fusion (RRF)** | `Σ 1 / (k + rank_i)` | k=60 typical; rank-agnostic |
| **Linear Combination** | `w1×dense + w2×sparse` | Learn weights on validation set |

**RRF Advantage:** Works with any ranking scores; no calibration needed.

---

## 4. Vector Databases and ANN

### 4.1 HNSW (Hierarchical Navigable Small World)

ChromaDB's default ANN index.

**Structure:**
```
Layer 2:  ●────●────●         (Sparse, long-range connections)
           │    │    │
Layer 1:  ●────●────●────●    (Medium density)
           │    │    │    │
Layer 0:  ●────●────●────●────● (Full graph, all vectors)
```

**Search Process:**
1. Start at entry point in top layer
2. Greedy search: move to neighbor with highest similarity
3. Drop to next layer; repeat
4. Layer 0: exhaustive local search

**Parameters:**

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `M` | 16 - 64 | Max connections per node; higher = better recall, more memory |
| `ef_construction` | 100 - 500 | Build-time search width; higher = better index quality, slower build |
| `ef_search` | 50 - 500 | Query-time search width; higher = better recall, slower query |

### 4.2 ChromaDB Architecture

```
Client → Collection → [HNSW Index + Metadata Store + Document Store]
                          ↓
                   Persistence (SQLite + binary index files)
```

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **Collection** | Isolated vector space with its own index |
| **Embedding Function** | Callable: List[str] → List[List[float]] |
| **Metadata** | JSON-serializable dict per document; filterable |
| **Distance Function** | `cosine` (default), `l2`, `ip` (inner product) |

**Metadata Filtering Strategies:**

| Strategy | When to Use |
|----------|-------------|
| **Pre-filter** | Few results match filter; apply before ANN |
| **Post-filter** | Many match; apply after ANN retrieval |
| **ChromaDB** | Uses `where` clause; optimizer chooses strategy |

### 4.3 Persistence

```
chroma_db/
├── chroma.sqlite3          # Metadata, collections, embeddings
├── <collection_id>/
│   ├── index.bin           # HNSW graph
│   └── data.bin            # Vectors + documents
```

---

## 5. Generation and Grounding

### 5.1 Prompt Engineering for RAG

**System Prompt:**
```
You are a helpful assistant that answers questions based ONLY on the provided context.
If the context does not contain sufficient information, say "I don't have enough information to answer."
Always cite your sources using [Source X] format.
```

**User Prompt Template:**
```
Context:
[1] Document A, Page 3: "Relevant excerpt..."
[2] Document B, Page 1: "Another excerpt..."

Question: {user_question}

Answer:
```

### 5.2 Grounding Principles

| Principle | Implementation |
|-----------|----------------|
| **Faithfulness** | Answer must be entailed by retrieved context |
| **Attribution** | Every claim cites source document + chunk |
| **Abstention** | Explicit "I don't know" when context insufficient |
| **No Hallucination** | Never use parametric knowledge for factual claims |

### 5.3 Handling Insufficient Context

```python
def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't have any relevant documents to answer this question."
    
    # Check similarity threshold
    max_similarity = max(chunk.similarity for chunk in retrieved_chunks)
    if max_similarity < 0.5:  # Threshold
        return "I don't have enough relevant information to answer confidently."
    
    # Build prompt with citations
    context = format_context(retrieved_chunks)
    prompt = build_prompt(query, context)
    return llm.generate(prompt)
```

---

## 6. Evaluation Basics

### 6.1 Key Metrics

| Metric | Question | Measurement |
|--------|----------|-------------|
| **Faithfulness** | Is answer grounded in context? | LLM-eval or NLI model |
| **Answer Relevance** | Does answer address the question? | LLM-eval or semantic similarity |
| **Context Precision** | Are retrieved chunks relevant? | Human eval or LLM-eval |
| **Context Recall** | Did we retrieve all needed info? | Human eval (gold standard) |
| **Retrieval Hit Rate** | Is gold chunk in top-K? | Automated (with labeled data) |

### 6.2 Evaluation Frameworks (Future)

- **RAGAS:** Automated faithfulness, relevance, precision
- **TruLens:** Instrumentation + feedback functions
- **DeepEval:** Unit tests for LLM outputs

---

## 7. Phase 1 Architectural Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding Function | `SentenceTransformerEmbeddingFunction` | Avoids onnxruntime dependency; flexible model choice |
| Embedding Model | `all-MiniLM-L6-v2` (384-dim) | Fast, CPU-friendly, good baseline quality |
| Vector DB | ChromaDB 1.5.9 | Local, persistent, metadata filtering, Python-native |
| Distance Metric | Cosine | Standard for normalized embeddings |
| Chunking Strategy | Sentence-aware (planned) | Preserves semantic boundaries |
| LLM Integration | Ollama (local) + abstract interface | Privacy, cost, swapability |

---

*Last Updated: 2026-11-09 | Phase: 1 – Understanding RAG Concepts*