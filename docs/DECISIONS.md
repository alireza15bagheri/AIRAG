# Architectural Decision Records (ADRs)

## Overview

This document records significant architectural decisions made during the AIRAG project. Each ADR follows a lightweight format capturing context, decision, alternatives, and consequences.

**Format:** Based on [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)

---

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| 001 | Use SentenceTransformerEmbeddingFunction for ChromaDB | Accepted | 2026-11-09 |
| 002 | ChromaDB as Vector Database | Accepted | 2026-11-09 |
| 003 | Python 3.15 with onnxruntime Workaround | Accepted | 2026-11-09 |

---

## ADR-001: Use SentenceTransformerEmbeddingFunction for ChromaDB

### Status
**Accepted**

### Date
2026-11-09

### Context
ChromaDB provides a default embedding function (`ONNXMiniLM_L6_V2`) that uses ONNX Runtime to run a quantized version of `all-MiniLM-L6-v2`. However, `onnxruntime` does not publish wheels for Python 3.15 (released October 2026), making it impossible to install via pip on our target Python version.

We need an embedding solution that:
1. Works on Python 3.15+
2. Allows model flexibility (not locked to one model)
3. Uses our existing `sentence-transformers` dependency
4. Integrates cleanly with ChromaDB's embedding function interface

### Decision
Use ChromaDB's `SentenceTransformerEmbeddingFunction` with `all-MiniLM-L6-v2` as our primary embedding function instead of the default ONNX-based one.

**Implementation:**
```python
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",  # configurable via settings
    normalize_embeddings=True
)

collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function
)
```

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Wait for onnxruntime 3.15 wheel** | Use ChromaDB default | Unknown timeline; blocks development | ❌ Rejected |
| **Build onnxruntime from source** | Use default embedder | Requires C++ toolchain; complex CI | ❌ Rejected |
| **Downgrade to Python 3.12** | onnxruntime works | Lose Python 3.15 features/performance | ❌ Rejected |
| **Use OpenAI Embeddings API** | No local dependency | Cost, latency, privacy, offline unavailable | ❌ Rejected |
| **SentenceTransformerEmbeddingFunction** | Pure Python/PyTorch; model-flexible; works on 3.15 | Slightly slower than ONNX | ✅ Accepted |

### Consequences

**Positive:**
- ✅ Works on Python 3.15+ immediately
- ✅ Model-agnostic: can swap to `bge-base`, `e5-base`, `mpnet`, etc.
- ✅ Uses existing `sentence-transformers` dependency (already installed for other purposes)
- ✅ Pure PyTorch backend; supports CPU/MPS/CUDA
- ✅ Aligns with our learning objective: understand the embedding pipeline

**Negative:**
- ⚠️ ~2-3x slower than ONNX Runtime for batch embedding
- ⚠️ Higher memory usage (PyTorch vs quantized ONNX)
- ⚠️ Requires `sentence-transformers` as hard dependency

**Mitigation:**
- Batch embedding (configurable `embedding_batch_size=32`) for throughput
- CPU inference sufficient for our scale (thousands, not millions of docs)
- Can optimize later with ONNX when 3.15 wheel available

### Related Decisions
- ADR-002: ChromaDB as Vector Database
- ADR-003: Python 3.15 with onnxruntime Workaround

---

## ADR-002: ChromaDB as Vector Database

### Status
**Accepted**

### Date
2026-11-09

### Context
We need a vector database for storing document embeddings and performing similarity search. Requirements:
- Local-first (no external service dependency for development)
- Python-native API
- Persistent storage
- Metadata filtering support
- Free/open-source
- Good documentation and community

### Decision
Use **ChromaDB 1.5.9** as the initial vector database.

### Alternatives Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **ChromaDB** | Local, Python-native, persistent, metadata filtering, HNSW, simple API | Single-node; scaling limits | ✅ Accepted |
| **Qdrant** | High performance, Rust-based, clustering, rich filtering | Requires separate server/container; more complex | 🔄 Future consideration |
| **FAISS** | Very fast, Facebook Research, GPU support | No built-in persistence/metadata; lower-level API | 🔄 Future consideration |
| **Milvus** | Enterprise-grade, distributed, multi-tenancy | Overkill for local dev; complex setup | ❌ Rejected |
| **pgvector** | PostgreSQL extension, ACID, SQL queries | Requires Postgres; less optimized for ANN | 🔄 Future consideration |
| **Weaviate** | GraphQL API, modules, hybrid search | Separate service; Go-based; resource heavy | ❌ Rejected |

### Consequences

**Positive:**
- ✅ Zero-infrastructure local development
- ✅ Simple Python API: `client.create_collection()`, `collection.add()`, `collection.query()`
- ✅ Built-in HNSW index with configurable parameters
- ✅ Metadata filtering with `where` clauses
- ✅ Persistence to local filesystem (`./chroma_db/`)
- ✅ Supports custom embedding functions (critical for ADR-001)
- ✅ Active development and community

**Negative:**
- ⚠️ Single-node architecture; horizontal scaling requires migration
- ⚠️ HNSW index stored in memory + disk; memory grows with dataset
- ⚠️ No built-in replication or high availability
- ⚠️ Concurrent write performance limited

**Mitigation:**
- Design `VectorStore` interface to allow swapping backends
- Monitor dataset size; plan migration to Qdrant/Milvus at ~1M vectors
- Use metadata pre-filtering to reduce search space

### Configuration (Phase 1)

```python
# ChromaDB settings from config
chroma_persist_dir = "./chroma_db"
chroma_collection_name = "documents"
chroma_distance_function = "cosine"  # cosine, l2, ip

# HNSW parameters (defaults; tunable later)
hnsw_space = "cosine"
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 100
```

### Related Decisions
- ADR-001: SentenceTransformerEmbeddingFunction (requires ChromaDB custom embedding function support)
- ADR-003: Python 3.15 (ChromaDB 1.5.9 compatible)

---

## ADR-003: Python 3.15 with onnxruntime Workaround

### Status
**Accepted**

### Date
2026-11-09

### Context
The project uses Python 3.15.2 (latest stable as of October 2026). Several ML ecosystem packages lag in publishing wheels for new Python versions. Specifically:

- `onnxruntime`: No wheel for Python 3.15 (latest: 3.13/3.14)
- This blocks ChromaDB's default ONNX embedding function
- Other dependencies (PyTorch, sentence-transformers, transformers) have 3.15 wheels

### Decision
Proceed with Python 3.15 and use the `SentenceTransformerEmbeddingFunction` workaround (ADR-001) instead of waiting for `onnxruntime` 3.15 support.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Downgrade to Python 3.12** | Full ecosystem compatibility | Miss 3.15 performance/features; future migration needed | ❌ Rejected |
| **Wait for onnxruntime 3.15** | Use ChromaDB default | Indefinite timeline; blocks all progress | ❌ Rejected |
| **Build onnxruntime from source** | Use default embedder | Complex; requires MSVC; CI/CD burden | ❌ Rejected |
| **Use Python 3.15 + workaround** | Latest Python; unblocked development | Slightly slower embeddings | ✅ Accepted |

### Consequences

**Positive:**
- ✅ Access to Python 3.15 features (faster startup, better error messages, JIT improvements)
- ✅ Unblocked development; no dependency on external package maintainers
- ✅ Forces better architecture: abstract embedding interface from Day 1
- ✅ Learning opportunity: understand embedding pipeline internals

**Negative:**
- ⚠️ Embedding throughput ~2-3x slower than ONNX Runtime
- ⚠️ Higher memory usage for embedding model
- ⚠️ Technical debt: will want to switch back when onnxruntime 3.15 available

**Mitigation:**
- Document workaround clearly (this ADR + README note)
- Monitor `onnxruntime` releases; plan migration when 3.15 wheel appears
- Keep embedding interface abstract for zero-code swap

### Monitoring
- Watch: https://github.com/microsoft/onnxruntime/releases
- Target: Switch back when `onnxruntime>=1.18.0` publishes 3.15 wheels

### Related Decisions
- ADR-001: SentenceTransformerEmbeddingFunction (the workaround)
- ADR-002: ChromaDB (must support custom embedding functions)

---

## Future ADRs (Placeholder)

| ID | Title | Expected Phase |
|----|-------|----------------|
| 004 | Chunking Strategy: Sentence-Aware vs Semantic | Phase 3 |
| 005 | LLM Provider Abstraction Interface | Phase 7 |
| 006 | Hybrid Search Implementation (BM25 + Vector) | Phase 10 |
| 007 | Reranking with Cross-Encoders | Phase 10 |
| 008 | Production Vector DB Migration (Qdrant) | Phase 11 |
| 009 | Docker Multi-Stage Build Strategy | Phase 11 |
| 010 | Observability Stack (Logging, Metrics, Tracing) | Phase 11 |

---

## ADR Process

1. **Propose:** Create draft ADR when facing significant architectural choice
2. **Discuss:** Review alternatives, consequences with team (or self-reflection)
3. **Decide:** Mark status as Accepted/Rejected/Superseded
4. **Implement:** Code follows the decision
5. **Review:** Revisit when context changes (new requirements, new versions)

**Template for New ADRs:**

```markdown
## ADR-XXX: [Title]

### Status
[Proposed | Accepted | Rejected | Superseded]

### Date
YYYY-MM-DD

### Context
[What is the problem? What are the constraints?]

### Decision
[What did we decide? Include code/config snippets.]

### Alternatives Considered
| Alternative | Pros | Cons | Verdict |

### Consequences
**Positive:** ...
**Negative:** ...
**Mitigation:** ...

### Related Decisions
- ADR-XXX: ...

### Monitoring (if applicable)
[What to watch for future changes?]
```

---

*Last Updated: 2026-11-09 | Phase: 1 – Understanding RAG Concepts*