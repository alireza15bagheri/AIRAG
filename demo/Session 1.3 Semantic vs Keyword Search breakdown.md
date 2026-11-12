# Session 1.3 Demo — Comprehensive Technical Analysis

---

## 1. Conceptual Breakdown: Semantic vs Keyword Search

### Core Implementation Differences

| Aspect | BM25 (Keyword) | Dense (Semantic) |
|--------|----------------|------------------|
| **Matching Logic** | Exact token overlap | Vector similarity in embedding space |
| **Representation** | Sparse bag-of-words | Dense 384-dim vectors |
| **Query Processing** | Tokenize → lookup inverted index | Encode → cosine similarity vs all docs |
| **Strength** | Exact phrases, rare terms, speed | Synonyms, concepts, polysemy resolution |
| **Weakness** | Synonym gap, polysemy | Diluted rare terms, no exact phrase awareness |

---

### BM25 Implementation (Lines 176-200)

```python
def tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())  # Simple word tokenization

def build_bm25_index(documents):
    corpus = [tokenize(doc.text) for doc in documents]
    return BM25Okapi(corpus)  # rank_bm25 library
```

**BM25 Formula** (implemented in `rank_bm25`):
```
score(D, Q) = Σ IDF(t) × (f(t,D) × (k1+1)) / (f(t,D) + k1×(1-b+b×|D|/avgdl))
```

| Component | Purpose |
|-----------|---------|
| `IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5))` | Rare terms = higher weight |
| `f(t,D)` | Term frequency in document |
| `k1=1.5` | TF saturation (diminishing returns) |
| `b=0.75` | Document length normalization |

---

### Dense Embedding Search (Lines 202-230)

```python
def generate_dense_embeddings(model, texts):
    return model.encode(texts, normalize_embeddings=True)  # Unit vectors

def cosine_similarity_to_all(query, matrix):
    return (matrix @ query) / (||query|| × ||matrix||)  # Batch cosine
```

**Key Insight:** With `normalize_embeddings=True`, all vectors have `||v|| = 1`, so:
```
cosine(A, B) = A · B  (dot product = cosine similarity)
```

---

### Hybrid Fusion Strategies (Lines 260-320)

| Method | Formula | Parameters |
|--------|---------|------------|
| **Weighted** | `α × dense_norm + (1-α) × sparse_norm` | `α ∈ [0,1]` (tunable) |
| **RRF** | `Σ 1/(k + rank_i)` | `k=60` (parameter-free) |

```python
def search_hybrid(bm25, model, query, doc_embeddings, docs, top_k, alpha=0.5):
    # Normalize both scores to [0,1] for fair combination
    bm25_norm = (bm25_scores - min) / (max - min)
    dense_norm = (dense_scores - min) / (max - min)
    return alpha * dense_norm + (1 - alpha) * bm25_norm
```

---

## 2. Step-by-Step Testing Logic Walkthrough

### Phase 1: Setup & Index Building
```
1. Load 20-document corpus (5 per topic: Animals, Tech, Food, Space)
2. Build BM25 index: tokenize all docs → BM25Okapi
3. Load dense model: all-MiniLM-L6-v2 (384-dim)
4. Generate dense embeddings for all 20 docs (normalized)
```

### Phase 2: Side-by-Side Comparison (6 Queries)
For each query in `DEMO_QUERIES`:
```
┌─────────────────────────────────────────────────────────┐
│ Query: "A small feline is sleeping in the sun."        │
├──────────────┬──────────────┬──────────┬───────────────┤
│ BM25         │ Dense        │ Hybrid   │ RRF           │
├──────────────┼──────────────┼──────────┼───────────────┤
│ 1. Mars/sun  │ 1. Cat/window│ 1. Mars  │ 1. Mars       │
│ (keyword)    │ (semantic)   │ (α=0.5)  │ (rank fusion) │
└──────────────┴──────────────┴──────────┴───────────────┘
```

### Phase 3: Failure Case Analysis
| Case | Query | BM25 | Dense |
|------|-------|------|-------|
| Synonym Gap | "automobile engine repair" | 0 results | Low (wrong domain) |
| Polysemy | "Apple stock price" | 0 results | Low (wrong sense) |
| Exact Phrase | "transformer architecture..." | Perfect match | High (0.96) |
| Rare Terms | "sourdough fermentation..." | Exact match | High + noise |

### Phase 4: Hybrid Fusion Analysis
```
Alpha sweep for "feline sleeping":
α=0.0 (pure BM25) → Mars/sun
α=0.5 (hybrid)   → Mars/sun  
α=0.75 (dense)   → Cat/window
α=1.0 (pure dense) → Cat/window
```

---

## 3. Execution Instructions

### Prerequisites
```bash
pip install rank_bm25 sentence-transformers scikit-learn numpy
```

### Run Modes

```bash
# Full demo (all sections)
python demo/session_1_3_semantic_vs_keyword.py

# Self-test only (no model loading)
python demo/session_1_3_semantic_vs_keyword.py --self-test

# Single method
python demo/session_1_3_semantic_vs_keyword.py --method bm25
python demo/session_1_3_semantic_vs_keyword.py --method dense
python demo/session_1_3_semantic_vs_keyword.py --method hybrid
python demo/session_1_3_semantic_vs_keyword.py --method rrf

# One-shot query
python demo/session_1_3_semantic_vs_keyword.py --query "feline sleeping"

# Interactive mode
python demo/session_1_3_semantic_vs_keyword.py --interactive

# Custom alpha for hybrid
python demo/session_1_3_semantic_vs_keyword.py --method hybrid --alpha 0.7
```

---

## 4. Output Interpretation Guide

### BM25 Score Interpretation
```
BM25 Score: 14.0182  → Exact phrase match (max possible)
BM25 Score: 5.8131   → Single keyword match ("sun")
BM25 Score: 0.0000   → No token overlap
```
**BM25 scores are unbounded** — higher = more relevant. Typical range: 0-20+.

### Dense (Cosine) Score Interpretation
```
Cosine: 0.96  → Nearly identical meaning
Cosine: 0.41  → Semantically related
Cosine: 0.17  → Weak/unrelated
Cosine: 0.00  → Orthogonal (unrelated)
```
**Range: [-1, 1]** — With normalized embeddings, `cosine = dot product`.

### Hybrid Score Interpretation
```
Hybrid: 0.7277  → Normalized weighted combination
```
**Range: [0, 1]** — Both inputs normalized to [0,1] before combination.

### RRF Score Interpretation
```
RRF: 0.0320  → 1/(60+rank_bm25) + 1/(60+rank_dense)
```
**Range: [0, ~0.033]** — Lower rank = higher score. Max when rank=1 in both.

---

### Comparison Table: "feline sleeping" Query

| Method | Top Result | Score | Why |
|--------|------------|-------|-----|
| **BM25** | "Mars...sun" | 5.81 | Matches "sun" keyword |
| **Dense** | "Cat...windowsill" | 0.41 | "feline" ≈ "cat" in vector space |
| **Hybrid** | "Mars...sun" | 0.73 | BM25 dominates at α=0.5 |
| **RRF** | "Mars...sun" | 0.032 | BM25 rank=1 dominates fusion |

---

### Key Diagnostic Patterns

| Pattern | Indicates |
|---------|-----------|
| BM25 high, Dense low | Exact phrase / rare terms |
| Dense high, BM25 low | Synonyms, concepts, polysemy |
| Both high | Strong exact + semantic match |
| Both low | Irrelevant query |
| Hybrid shifts with α | Tunable trade-off |
| RRF stable | Robust parameter-free fusion |

---

## Key Takeaways for RAG Design

| Scenario | Recommended Approach |
|----------|---------------------|
| Exact phrase lookup (legal, code) | BM25 or Hybrid (α≈0.3) |
| General QA, synonyms | Dense or Hybrid (α≈0.7) |
| Production RAG (no tuning) | RRF (parameter-free) |
| Speed-critical | BM25 (inverted index) |
| Cross-lingual | Dense (multilingual embeddings) |

---

The demo is ready to run and experiment with. Each section builds intuition for when to use each search paradigm in your RAG pipeline.