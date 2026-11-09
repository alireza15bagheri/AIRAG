# Architecture Documentation

## System Overview

AIRAG follows a modular, layered architecture designed for extensibility and learning transparency. Each layer has a single responsibility and communicates via well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AIRAG APPLICATION                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    API LAYER (FastAPI)                      │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │
│  │  │ Documents│ │  Query   │ │  Health  │ │  Admin   │       │ │
│  │  │  Routes  │ │  Routes  │ │  Routes  │ │  Routes  │       │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    SERVICE LAYER                            │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │ │
│  │  │ Ingestion    │ │ Retrieval    │ │ Generation   │        │ │
│  │  │ Service      │ │ Service      │ │ Service      │        │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    CORE DOMAIN LAYER                        │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │
│  │  │ Document │ │  Chunk   │ │  Query   │ │  Answer  │       │ │
│  │  │  Models  │ │  Models  │ │  Models  │ │  Models  │       │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    INFRASTRUCTURE LAYER                     │ │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │ │
│  │ │ Document │ │ Embedding│ │  Vector  │ │   LLM    │        │ │
│  │ │ Loaders  │ │ Providers│ │  Store   │ │ Providers│        │ │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Ingestion Pipeline (Offline/Batch)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Upload     │────▶│  Extract    │────▶│  Normalize  │────▶│   Chunk     │────▶│  Embed &    │
│  Document   │     │  Text       │     │  (Clean)    │     │  (Split)    │     │  Store      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
                                                        ┌─────────────────────┐
                                                        │ Vector Database     │
                                                        │ (ChromaDB)          │
                                                        │ - Vectors           │
                                                        │ - Chunk Text        │
                                                        │ - Metadata          │
                                                        └─────────────────────┘
```

### Query Pipeline (Online/Real-time)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  User Query │────▶│  Embed      │────▶│  Search     │────▶│  Retrieve   │
│  (Text)     │     │  Query      │     │  (ANN)      │     │  Top-K      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                        │
                                                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Answer +   │◀───▶│  Generate   │◀───▶│  Build      │◀───▶│  Assemble   │
│  Citations  │     │  (LLM)      │     │  Prompt     │     │  Context    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Component Specifications

### 1. Configuration (`src/airag/config.py`)

**Technology:** Pydantic Settings (v2)

**Responsibilities:**
- Load from `.env` file with validation
- Type-safe access to all settings
- Singleton pattern via module-level `settings` instance

**Key Settings Groups:**

| Group | Parameters |
|-------|------------|
| App | `app_name`, `debug`, `log_level` |
| Server | `host`, `port` |
| Document | `allowed_extensions`, `max_file_size_mb`, `upload_dir` |
| Chunking | `chunk_size`, `chunk_overlap`, `chunk_strategy` |
| Embeddings | `embedding_model`, `embedding_device`, `embedding_batch_size` |
| Vector DB | `chroma_persist_dir`, `chroma_collection_name`, `chroma_distance_function` |
| Retrieval | `top_k`, `similarity_threshold` |
| LLM (Ollama) | `ollama_base_url`, `ollama_model`, `ollama_temperature` |
| Generation | `max_context_tokens`, `system_prompt`, `enable_streaming` |

---

### 2. Document Loaders (Phase 2+)

**Interface:**
```python
class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> List[Document]: ...
    
    @abstractmethod
    def supported_extensions(self) -> List[str]: ...
```

**Implementations (Planned):**
| Loader | Library | Formats | Metadata |
|--------|---------|---------|----------|
| `PDFLoader` | PyMuPDF | `.pdf` | Page numbers, headers |
| `DOCXLoader` | python-docx | `.docx` | Paragraphs, styles |
| `MarkdownLoader` | markdown | `.md`, `.markdown` | Headings, code blocks |
| `TextLoader` | stdlib | `.txt` | Line numbers |

---

### 3. Chunking Strategies (Phase 3+)

**Interface:**
```python
class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]: ...
```

**Strategies (Planned):**
| Strategy | Description | Parameters |
|----------|-------------|------------|
| `FixedSizeChunker` | Fixed token count with overlap | `chunk_size`, `overlap` |
| `SentenceAwareChunker` | Split at sentence boundaries | `chunk_size`, `overlap` |
| `SemanticChunker` | Split by embedding similarity | `threshold`, `window_size` |

**Chunk Metadata Propagation:**
```python
@dataclass
class Chunk:
    text: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
```

---

### 4. Embedding Providers (Phase 1 Decision)

**Interface:**
```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]: ...
    
    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

**Phase 1 Decision:** `SentenceTransformerEmbeddingProvider`
- Wraps `sentence-transformers` library
- Model: `all-MiniLM-L6-v2` (384-dim)
- Device: CPU (configurable)
- Batch size: 32 (configurable)

**Why Not ChromaDB Default ONNX?**
- Requires `onnxruntime` (no Python 3.15 wheel)
- Locked to single model (`all-MiniLM-L6-v2` ONNX)
- Our approach: model-agnostic, pure PyTorch

---

### 5. Vector Store (ChromaDB)

**Wrapper Interface:**
```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None: ...
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int, 
               filter: Optional[Dict] = None) -> List[SearchResult]: ...
    
    @abstractmethod
    def delete(self, document_id: str) -> None: ...
    
    @abstractmethod
    def get_collection_stats(self) -> Dict: ...
```

**ChromaDB Configuration (Phase 1):**
| Setting | Value | Rationale |
|---------|-------|-----------|
| Persistence | `./chroma_db` | Local filesystem |
| Distance | `cosine` | Normalized embeddings |
| HNSW `M` | 16 (default) | Balance memory/recall |
| HNSW `ef_construction` | 200 (default) | Good index quality |
| HNSW `ef_search` | 100 (default) | Query-time recall |

**Metadata Schema:**
```json
{
  "document_id": "uuid",
  "chunk_index": 0,
  "page_number": 3,
  "section_title": "Introduction",
  "source_path": "/uploads/doc.pdf",
  "created_at": "2026-11-09T10:00:00Z"
}
```

---

### 6. LLM Providers (Phase 7+)

**Interface:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
```

**Planned Implementations:**
| Provider | Model | Use Case |
|----------|-------|----------|
| `OllamaProvider` | `llama3.2:3b`, `phi3:mini` | Local, private, free |
| `OpenAIProvider` | `gpt-4o`, `gpt-4o-mini` | Cloud, highest quality |
| `AnthropicProvider` | `claude-3.5-sonnet` | Cloud, large context |

---

### 7. Prompt Templates (Phase 7+)

**System Prompt (Configurable):**
```
You are a helpful assistant that answers questions based ONLY on the provided context.
If the context does not contain sufficient information to answer the question,
say "I don't have enough information to answer."
Always cite sources using [Source X] format where X is the chunk number.
```

**User Prompt Template:**
```
Context:
{context}

Question: {query}

Answer:
```

**Context Formatting:**
```
[1] Document: "filename.pdf" (Page 3)
    "Relevant excerpt from the document..."

[2] Document: "report.docx" (Section: "Methodology")
    "Another relevant excerpt..."
```

---

## Configuration Architecture

### Environment Variables (`.env`)

All settings flow through Pydantic Settings:

```
.env → Pydantic Settings → config.settings (singleton) → All components
```

**Validation Rules:**
- `chunk_size > chunk_overlap > 0`
- `top_k > 0`
- `similarity_threshold ∈ [0, 1]`
- `embedding_device ∈ {cpu, cuda, mps}`

---

## Error Handling Strategy

| Layer | Approach |
|-------|----------|
| API | FastAPI exception handlers → structured JSON errors |
| Service | Custom exceptions (`IngestionError`, `RetrievalError`, `GenerationError`) |
| Infrastructure | Retry logic (tenacity), circuit breakers, graceful degradation |

**Custom Exceptions (Phase 2+):**
```python
class AIRAGError(Exception): pass
class IngestionError(AIRAGError): pass
class RetrievalError(AIRAGError): pass
class GenerationError(AIRAGError): pass
class ConfigurationError(AIRAGError): pass
```

---

## Testing Architecture

| Layer | Test Type | Tools |
|-------|-----------|-------|
| Core Models | Unit | pytest, pytest-mock |
| Loaders/Chunkers | Unit | pytest, fixtures |
| Embedding/Vector | Integration | pytest, testcontainers (optional) |
| API Routes | Integration | httpx, pytest-asyncio |
| E2E Pipeline | E2E | pytest, test fixtures |

---

## Deployment Architecture (Future)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  FastAPI    │────▶│  ChromaDB   │
│  (Reverse   │     │  (Uvicorn)  │     │  (Local)    │
│   Proxy)    │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │  Ollama │   │  Redis  │
              │ (Local) │   │ (Cache) │
              └─────────┘   └─────────┘
```

---

## Phase 1 Architecture Decisions

| Component | Decision | ADR Reference |
|-----------|----------|---------------|
| Embedding Function | `SentenceTransformerEmbeddingFunction` | ADR-001 |
| Embedding Model | `all-MiniLM-L6-v2` (384-dim) | ADR-001 |
| Vector DB | ChromaDB 1.5.9 | ADR-002 |
| Distance Metric | Cosine | ADR-002 |
| Python Version | 3.15.2 | ADR-003 |
| onnxruntime Workaround | Custom embedding function | ADR-003 |

---

*Last Updated: 2026-11-09 | Phase: 1 – Understanding RAG Concepts*