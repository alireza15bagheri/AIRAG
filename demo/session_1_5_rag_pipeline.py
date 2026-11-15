#!/usr/bin/env python
"""
Session 1.5 Demo: Complete RAG Pipeline Walkthrough
====================================================
End-to-end demonstration of a Retrieval-Augmented Generation pipeline:
  1. Document ingestion → chunking → embedding → storage (ChromaDB)
  2. Query processing → embedding → retrieval
  3. Prompt construction with retrieved context
  4. LLM generation (Ollama) → answer with citations

Run the full demo:
    python demo/session_1_5_rag_pipeline.py

Run a one-shot query:
    python demo/session_1_5_rag_pipeline.py --query "What is RAG?"

Run interactively:
    python demo/session_1_5_rag_pipeline.py --interactive

Run only the fast, model-free self-tests:
    python demo/session_1_5_rag_pipeline.py --self-test

If Ollama is not running, the demo uses a mock LLM for demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Try to import required packages
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_DIM = 384

# Sample documents for the demo
SAMPLE_DOCS: list[dict[str, str]] = [
    {
        "id": "doc_1",
        "title": "What is RAG?",
        "content": """Retrieval-Augmented Generation (RAG) is an AI technique that combines 
        information retrieval with text generation. Instead of relying solely on the 
        model's training data, RAG retrieves relevant documents from a knowledge base 
        and uses them as context for generating answers. This approach reduces 
        hallucinations and allows the model to access up-to-date information."""
    },
    {
        "id": "doc_2",
        "title": "How Embeddings Work",
        "content": """Text embeddings convert sentences into dense vectors of numbers. 
        Similar meanings produce vectors that are close together in high-dimensional 
        space. The all-MiniLM-L6-v2 model produces 384-dimensional vectors. 
        Cosine similarity measures the angle between vectors - smaller angles mean 
        more similar meanings. This enables semantic search beyond keyword matching."""
    },
    {
        "id": "doc_3",
        "title": "Vector Databases and HNSW",
        "content": """Vector databases like ChromaDB store embeddings and enable fast 
        similarity search. They use HNSW (Hierarchical Navigable Small World) graphs 
        to achieve sub-millisecond search times even with millions of vectors. HNSW 
        builds layered graphs where upper layers have long-range connections for 
        fast navigation, and the bottom layer has dense connections for accuracy."""
    },
    {
        "id": "doc_4",
        "title": "Chunking Strategies",
        "content": """Large documents must be split into chunks before embedding. 
        Fixed-size chunking splits by character count with overlap. 
        Sentence-aware chunking respects sentence boundaries. 
        Semantic chunking uses embedding similarity to find natural boundaries. 
        Overlap between chunks preserves context at boundaries. 
        Typical chunk sizes range from 100-500 tokens with 10-20% overlap."""
    },
    {
        "id": "doc_5",
        "title": "RAG vs Fine-tuning",
        "content": """RAG and fine-tuning serve different purposes. RAG adds external 
        knowledge without changing model weights - it's faster to update and 
        provides citations. Fine-tuning modifies model weights for specific tasks 
        but requires retraining for new knowledge. RAG is better for dynamic 
        knowledge bases; fine-tuning is better for style and behavior adaptation."""
    },
]


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: np.ndarray | None = None


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score."""
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    citations: list[RetrievedChunk]
    query: str
    latency_ms: float


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def create_chunks_from_docs(
    docs: list[dict[str, str]], 
    chunk_size: int, 
    overlap: int
) -> list[Chunk]:
    """Create chunks from documents with metadata."""
    chunks = []
    for doc in docs:
        doc_id = doc["id"]
        full_text = f"{doc['title']}\n\n{doc['content']}"
        text_chunks = chunk_text(full_text, chunk_size, overlap)
        
        for i, chunk_text_item in enumerate(text_chunks):
            start_char = full_text.find(chunk_text_item)
            chunks.append(Chunk(
                text=chunk_text_item,
                doc_id=doc_id,
                chunk_index=i,
                start_char=start_char,
                end_char=start_char + len(chunk_text_item)
            ))
    return chunks


def load_embedding_model(model_name: str) -> Any:
    """Load SentenceTransformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise SystemExit("sentence-transformers required. Install: pip install sentence-transformers")
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    dim = getattr(model, "get_embedding_dimension", lambda: model.get_sentence_embedding_dimension())()
    print(f"Model: {model.__class__.__name__}, Dimension: {dim}, Device: {model.device}")
    return model


def generate_embeddings(model: Any, texts: Sequence[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


class ChromaVectorStore:
    """Wrapper around ChromaDB for vector storage and retrieval."""
    
    def __init__(
        self, 
        collection_name: str = "rag_demo", 
        persist_dir: str | None = None
    ):
        if not CHROMADB_AVAILABLE:
            raise SystemExit("chromadb required. Install: pip install chromadb")
        
        if persist_dir:
            client = chromadb.PersistentClient(path=persist_dir)
        else:
            client = chromadb.Client()
        
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks with embeddings to the collection."""
        if not chunks:
            return
        
        ids = [f"{c.doc_id}_chunk_{c.chunk_index}" for c in chunks]
        embeddings = [c.embedding.tolist() for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    
    def query(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Query the vector store for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                # Convert distance to similarity (cosine distance -> similarity)
                score = 1.0 - distance
                chunk = Chunk(
                    text=results["documents"][0][i],
                    doc_id=meta["doc_id"],
                    chunk_index=meta["chunk_index"],
                    start_char=meta["start_char"],
                    end_char=meta["end_char"],
                )
                retrieved.append(RetrievedChunk(chunk=chunk, score=score))
        return retrieved
    
    def count(self) -> int:
        return self.collection.count()
    
    def clear(self) -> None:
        """Delete all documents in the collection."""
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
        except Exception:
            pass


class OllamaClient:
    """Client for Ollama LLM API."""
    
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Generate text using Ollama."""
        if not self.available:
            return self._mock_generate(prompt)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 512),
            }
        }
        if system:
            payload["system"] = system
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama error: {e}, falling back to mock")
        
        return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Generate a mock response when Ollama is unavailable."""
        # Extract context from prompt for more realistic mock
        if "RAG" in prompt or "retrieval" in prompt.lower():
            return ("RAG (Retrieval-Augmented Generation) combines information retrieval "
                    "with text generation. It retrieves relevant documents from a knowledge "
                    "base and uses them as context for generating answers, reducing "
                    "hallucinations and providing citations.")
        elif "embedding" in prompt.lower():
            return ("Text embeddings convert sentences into dense vectors where similar "
                    "meanings are close in vector space. The all-MiniLM-L6-v2 model "
                    "produces 384-dimensional vectors.")
        elif "hnsw" in prompt.lower() or "vector database" in prompt.lower():
            return ("HNSW (Hierarchical Navigable Small World) builds layered graphs "
                    "for fast approximate nearest neighbor search. ChromaDB uses HNSW "
                    "for sub-millisecond search over millions of vectors.")
        else:
            return ("Based on the provided context, I can answer your question. "
                    "The context contains information about RAG, embeddings, and vector "
                    "databases. Please ask a specific question for a more targeted answer.")


def build_prompt(query: str, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Build the prompt with retrieved context and citations."""
    # System prompt
    system = """You are a helpful assistant that answers questions using ONLY the provided context. 
    Cite your sources using [1], [2], etc. corresponding to the numbered context chunks.
    If the context doesn't contain enough information, say so explicitly.
    Be concise and accurate."""
    
    # Build context
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = f"{chunk.chunk.doc_id} (chunk {chunk.chunk.chunk_index})"
        context_parts.append(f"[{i}] {source}: {chunk.chunk.text}")
    
    context = "\n\n".join(context_parts)
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer using only the context above. Cite sources like [1], [2]. If insufficient context, say so."""
    
    return system, user_prompt


def run_rag_pipeline(
    query: str,
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    llm: OllamaClient,
    top_k: int = DEFAULT_TOP_K,
) -> RAGResponse:
    """Execute the complete RAG pipeline for a query."""
    start_time = time.perf_counter()
    
    # 1. Embed the query
    query_embedding = generate_embeddings(embedding_model, [query])[0]
    
    # 2. Retrieve relevant chunks
    retrieved = vector_store.query(query_embedding, top_k=top_k)
    
    # 3. Build prompt with context
    system, prompt = build_prompt(query, retrieved)
    
    # 4. Generate answer with LLM
    answer = llm.generate(prompt, system=system)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return RAGResponse(
        answer=answer,
        citations=retrieved,
        query=query,
        latency_ms=latency_ms,
    )


def format_response(response: RAGResponse) -> str:
    """Format RAG response for display."""
    lines = [
        f"\nQuery: {response.query!r}",
        f"Latency: {response.latency_ms:.1f} ms",
        f"Retrieved {len(response.citations)} chunks",
        "\n" + "─" * 60,
        "ANSWER:",
        "─" * 60,
        response.answer,
        "\n" + "─" * 60,
        "CITATIONS:",
        "─" * 60,
    ]
    
    for i, cite in enumerate(response.citations, 1):
        preview = textwrap.shorten(cite.chunk.text, width=120, placeholder="...")
        lines.append(f"  [{i}] {cite.chunk.doc_id} (chunk {cite.chunk.chunk_index}) - score: {cite.score:.4f}")
        lines.append(f"      {preview}")
    
    return "\n".join(lines)


def demo_ingestion(
    docs: list[dict[str, str]],
    chunk_size: int,
    overlap: int,
    embedding_model: Any,
    vector_store: ChromaVectorStore,
) -> list[Chunk]:
    """Demonstrate document ingestion pipeline."""
    print_section("1. DOCUMENT INGESTION PIPELINE")
    
    print("Splitting documents into chunks...")
    chunks = create_chunks_from_docs(docs, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    
    # Show chunking example
    print("\nChunking example (first document):")
    first_doc = docs[0]
    full_text = f"{first_doc['title']}\n\n{first_doc['content']}"
    sample_chunks = chunk_text(full_text, chunk_size, overlap)
    for i, chunk in enumerate(sample_chunks[:3]):
        preview = textwrap.shorten(chunk, width=80, placeholder="...")
        print(f"  Chunk {i}: {preview}")
    if len(sample_chunks) > 3:
        print(f"  ... and {len(sample_chunks) - 3} more chunks")
    
    print("\nGenerating embeddings...")
    start = time.perf_counter()
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(embedding_model, texts)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Generated {len(embeddings)} embeddings in {elapsed:.1f} ms")
    
    # Attach embeddings to chunks
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    
    print("\nStoring in ChromaDB...")
    vector_store.clear()  # Clean slate
    vector_store.add_chunks(chunks)
    print(f"Stored {vector_store.count()} chunks in ChromaDB")
    
    return chunks


def demo_retrieval(
    queries: list[str],
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    top_k: int,
) -> None:
    """Demonstrate retrieval."""
    print_section("2. SEMANTIC RETRIEVAL")
    
    for query in queries:
        print(f"\nQuery: {query!r}")
        query_embedding = generate_embeddings(embedding_model, [query])[0]
        retrieved = vector_store.query(query_embedding, top_k=top_k)
        
        print(f"  Top {len(retrieved)} results:")
        for i, item in enumerate(retrieved, 1):
            preview = textwrap.shorten(item.chunk.text, width=100, placeholder="...")
            print(f"    {i}. [{item.score:.4f}] {item.chunk.doc_id} (chunk {item.chunk.chunk_index})")
            print(f"       {preview}")


def demo_generation(
    queries: list[str],
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    llm: OllamaClient,
    top_k: int,
) -> None:
    """Demonstrate answer generation."""
    print_section("3. ANSWER GENERATION WITH LLM")
    
    if llm.available:
        print(f"Using Ollama model: {llm.model}")
    else:
        print("Ollama not available - using mock LLM for demonstration")
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"QUERY: {query}")
        print("─" * 60)
        
        response = run_rag_pipeline(query, vector_store, embedding_model, llm, top_k=3)
        print(format_response(response))


def demo_pipeline_comparison(
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    llm: OllamaClient,
) -> None:
    """Compare RAG vs no-RAG responses."""
    print_section("4. RAG vs NO-RAG COMPARISON")
    
    query = "How does HNSW work in vector databases?"
    
    # With RAG
    rag_response = run_rag_pipeline(query, vector_store, embedding_model, llm, top_k=3)
    
    # Without RAG (no context)
    no_rag_prompt = f"Question: {query}\nAnswer:"
    no_rag_answer = llm.generate(no_rag_prompt, system="You are a helpful assistant.")
    
    print(f"Query: {query!r}\n")
    print("WITH RAG (context from ChromaDB):")
    print("-" * 50)
    print(rag_response.answer)
    print(f"\nCitations: {len(rag_response.citations)} chunks")
    
    print("\nWITHOUT RAG (model knowledge only):")
    print("-" * 50)
    print(no_rag_answer)


def demo_citation_tracking(
    query: str,
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    llm: OllamaClient,
) -> None:
    """Show how citations work in the response."""
    print_section("5. CITATION TRACKING")
    
    response = run_rag_pipeline(query, vector_store, embedding_model, llm, top_k=3)
    
    print("Answer with inline citations:")
    print(response.answer)
    print("\nDetailed citation mapping:")
    
    # Show which parts of answer might come from which source
    for i, cite in enumerate(response.citations, 1):
        print(f"\n  Source [{i}]: {cite.chunk.doc_id} (similarity: {cite.score:.4f})")
        print(f"  Text: {textwrap.shorten(cite.chunk.text, 200, placeholder='...')}")


def interactive_mode(
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    llm: OllamaClient,
    top_k: int,
) -> None:
    """Interactive query mode."""
    print_section("INTERACTIVE MODE")
    print("Enter queries to test the RAG pipeline.")
    print("Commands: /help, /quit, /topk <n>, /norag <query>")
    print()
    
    while True:
        try:
            raw = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not raw:
            continue
        
        if raw.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if raw.lower() in {"/help", "help", "?"}:
            print("  <query>        Run RAG pipeline")
            print("  /norag <query> Generate without retrieval")
            print("  /topk <n>      Set number of retrieved chunks")
            print("  /help          Show this help")
            print("  /quit          Exit")
            continue
        
        if raw.lower().startswith("/topk "):
            try:
                top_k = int(raw.split()[1])
                print(f"top_k set to {top_k}")
            except (IndexError, ValueError):
                print("Usage: /topk <number>")
            continue
        
        if raw.lower().startswith("/norag "):
            query = raw[7:].strip()
            answer = llm.generate(f"Question: {query}\nAnswer:", 
                                  system="You are a helpful assistant.")
            print(f"\nNO-RAG Answer:\n{answer}\n")
            continue
        
        # Run RAG pipeline
        response = run_rag_pipeline(raw, vector_store, embedding_model, llm, top_k=top_k)
        print(format_response(response))


def self_test() -> None:
    """Fast self-tests without model loading."""
    print_section("SELF-TEST: RAG PIPELINE COMPONENTS")
    
    # Test chunking
    text = "This is a test sentence. This is another sentence. And a third one."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1, "Should create multiple chunks"
    print("  PASS  chunk_text creates overlapping chunks")
    
    # Test overlap
    full = "A" * 50
    chunks = chunk_text(full, chunk_size=20, overlap=10)
    # Verify overlap
    for i in range(len(chunks) - 1):
        # The last 10 chars of chunk i should match first 10 of chunk i+1
        assert chunks[i][-10:] == chunks[i+1][:10], "Overlap not preserved"
    print("  PASS  chunk overlap preserved correctly")
    
    # Test Chunk dataclass
    chunk = Chunk(text="test", doc_id="doc1", chunk_index=0, start_char=0, end_char=4)
    assert chunk.text == "test"
    print("  PASS  Chunk dataclass works")
    
    # Test cosine similarity
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    print("  PASS  cosine_similarity returns 1.0 for identical vectors")
    
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-6
    print("  PASS  cosine_similarity returns 0.0 for orthogonal vectors")
    
    # Test prompt building
    chunks = [
        RetrievedChunk(
            chunk=Chunk(text="RAG combines retrieval and generation.", 
                       doc_id="doc1", chunk_index=0, start_char=0, end_char=40),
            score=0.9
        )
    ]
    system, prompt = build_prompt("What is RAG?", chunks)
    assert "[1]" in prompt
    assert "RAG combines retrieval and generation" in prompt
    print("  PASS  build_prompt includes citations and context")
    
    # Test RAGResponse formatting
    response = RAGResponse(
        answer="RAG combines retrieval and generation [1].",
        citations=[],
        query="What is RAG?",
        latency_ms=100.0
    )
    formatted = format_response(response)
    assert "RAG combines" in formatted
    assert "100.0" in formatted
    print("  PASS  format_response includes answer and latency")
    
    print("\nAll self-tests passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end RAG Pipeline Walkthrough"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retrieved chunks per query")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--persist-dir", help="ChromaDB persistence directory")
    parser.add_argument("--query", action="append", default=[], help="One-shot query (repeatable)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests only")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation (retrieval only)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    
    if args.self_test:
        self_test()
        return 0
    
    print("\n" + "=" * 80)
    print("  SESSION 1.5: COMPLETE RAG PIPELINE WALKTHROUGH")
    print("  From documents to answers with citations")
    print("=" * 80)
    
    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("sentence-transformers not available. Install: pip install sentence-transformers")
        return 1
    if not CHROMADB_AVAILABLE:
        print("chromadb not available. Install: pip install chromadb")
        return 1
    
    # Initialize components
    embedding_model = load_embedding_model(args.model)
    
    persist_dir = args.persist_dir or str(Path(__file__).parent / "output" / "chroma_rag_demo")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vector_store = ChromaVectorStore(persist_dir=persist_dir)
    
    llm = None
    if not args.no_llm:
        llm = OllamaClient(base_url=args.ollama_url, model=args.ollama_model)
    
    # One-shot queries
    if args.query:
        if not llm:
            print("LLM required for queries. Use --no-llm for retrieval only.")
            return 1
        
        # Need to ingest first
        chunks = demo_ingestion(SAMPLE_DOCS, args.chunk_size, args.overlap, embedding_model, vector_store)
        
        for query in args.query:
            print(f"\n{'=' * 80}")
            print(f"QUERY: {query}")
            print("=" * 80)
            response = run_rag_pipeline(query, vector_store, embedding_model, llm, top_k=args.top_k)
            print(format_response(response))
        return 0
    
    # Full demo walkthrough
    print("\nLoading sample documents...")
    
    # 1. Ingestion
    chunks = demo_ingestion(SAMPLE_DOCS, args.chunk_size, args.overlap, embedding_model, vector_store)
    
    # 2. Retrieval
    demo_queries = [
        "What is RAG?",
        "How do embeddings work?",
        "What is HNSW?",
        "Explain chunking strategies",
    ]
    demo_retrieval(demo_queries, vector_store, embedding_model, args.top_k)
    
    # 3. Generation (if LLM available)
    if llm:
        demo_generation(demo_queries[:2], vector_store, embedding_model, llm, args.top_k)
        demo_pipeline_comparison(vector_store, embedding_model, llm)
        demo_citation_tracking("How does HNSW work?", vector_store, embedding_model, llm)
    
    # 4. Interactive mode
    should_interact = args.interactive or (not args.query and sys.stdin.isatty())
    if should_interact and llm:
        interactive_mode(vector_store, embedding_model, llm, args.top_k)
    
    print("\n" + "=" * 80)
    print("  SESSION 1.5 COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • RAG = Retrieval + Generation: retrieve context, then generate answer")
    print("  • Ingestion: chunk → embed → store (ChromaDB with HNSW)")
    print("  • Retrieval: embed query → vector search → top-k chunks")
    print("  • Prompting: context + query → LLM → answer with citations")
    print("  • Citations enable verification and trust")
    print("  • RAG beats pure LLM for factual, up-to-date, domain-specific QA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())