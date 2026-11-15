#!/usr/bin/env python
"""
Session 1.4 Demo: Vector Databases and ANN Algorithms
======================================================
Educational demonstration of how vector databases store and search
high-dimensional embeddings efficiently using Approximate Nearest
Neighbor (ANN) algorithms, with ChromaDB integration.

Run the full demo:
    python demo/session_1_4_vector_databases.py

Run only the fast, model-free self-tests:
    python demo/session_1_4_vector_databases.py --self-test

Run a one-shot query:
    python demo/session_1_4_vector_databases.py --query "A kitten is sleeping."

Run interactively:
    python demo/session_1_4_vector_databases.py --interactive

Compare exact vs approximate search at different scales:
    python demo/session_1_4_vector_databases.py --benchmark

If ChromaDB is not installed, the demo still runs the pure-Python ANN
and brute-force sections; the ChromaDB section is skipped gracefully.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_DIMENSION = 384
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 50
DEFAULT_M = 16

# Try to import chromadb
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except (ImportError, KeyboardInterrupt):
    CHROMADB_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, KeyboardInterrupt):
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass(frozen=True)
class Document:
    """A document with text, id, and metadata."""
    text: str
    doc_id: int
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HNSWNode:
    """A node in the HNSW graph."""
    vector_id: int
    vector: np.ndarray
    layers: dict[int, list[int]] = field(default_factory=dict)  # level -> neighbor ids


DOCUMENTS: list[Document] = [
    Document("The cat sat on the warm windowsill.", 0, "Animals", {"topic": "pets"}),
    Document("A dog barked loudly at the mail carrier.", 1, "Animals", {"topic": "pets"}),
    Document("Kittens chase laser pointers across the floor.", 2, "Animals", {"topic": "pets"}),
    Document("The puppy wagged its tail happily.", 3, "Animals", {"topic": "pets"}),
    Document("A horse galloped across the open field.", 4, "Animals", {"topic": "pets"}),
    Document("Machine learning models learn patterns from data.", 5, "Technology", {"topic": "ai"}),
    Document("Neural networks are inspired by the human brain.", 6, "Technology", {"topic": "ai"}),
    Document("The transformer architecture uses self-attention.", 7, "Technology", {"topic": "ai"}),
    Document("Large language models generate text one token at a time.", 8, "Technology", {"topic": "ai"}),
    Document("Software engineers debug code and write tests.", 9, "Technology", {"topic": "software"}),
    Document("I baked a fresh loaf of sourdough bread.", 10, "Food", {"topic": "cooking"}),
    Document("The chef added garlic and basil to the pasta sauce.", 11, "Food", {"topic": "cooking"}),
    Document("She poured a cup of hot coffee.", 12, "Food", {"topic": "cooking"}),
    Document("Fresh strawberries taste sweet in summer.", 13, "Food", {"topic": "cooking"}),
    Document("The restaurant served grilled salmon with lemon.", 14, "Food", {"topic": "cooking"}),
    Document("The rocket launched into orbit.", 15, "Space", {"topic": "astronomy"}),
    Document("Mars is the fourth planet from the sun.", 16, "Space", {"topic": "astronomy"}),
    Document("Astronauts float in microgravity aboard the space station.", 17, "Space", {"topic": "astronomy"}),
    Document("A telescope captured an image of a distant galaxy.", 18, "Space", {"topic": "astronomy"}),
    Document("The moon orbits Earth once every month.", 19, "Space", {"topic": "astronomy"}),
]

DEMO_QUERIES = [
    "A small feline is sleeping in the sun.",
    "Deep learning and attention mechanisms.",
    "A spaceship travels to another planet.",
    "A warm meal with fresh ingredients.",
]


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


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of top k scores, best first."""
    return np.argsort(-np.asarray(scores))[:k]


def brute_force_search(query_vector: np.ndarray, embeddings: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Exact nearest neighbor search via brute-force O(n) scan."""
    scores = np.array([cosine_similarity(query_vector, emb) for emb in embeddings])
    top_indices = top_k_indices(scores, min(top_k, len(embeddings)))
    return [(int(idx), float(scores[idx])) for idx in top_indices]


class SimpleHNSW:
    """
    A minimal, educational HNSW (Hierarchical Navigable Small World) index.

    This is a simplified implementation for demonstration purposes only.
    Production vector databases use highly optimized C++ implementations.
    """

    def __init__(
        self,
        dimension: int,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        M: int = DEFAULT_M,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self.rng = random.Random(seed)
        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int | None = None
        self.max_level = 0

    def _random_level(self) -> int:
        """Generate a random layer for a new node (exponential decay)."""
        level = 0
        while self.rng.random() < 0.5 and level < 16:
            level += 1
        return level

    def _search_layer(
        self,
        query: np.ndarray,
        entry_id: int,
        level: int,
        ef: int,
    ) -> list[tuple[float, int]]:
        """Greedy nearest-neighbor search within a single HNSW layer."""
        visited: set[int] = {entry_id}
        entry_vector = self.nodes[entry_id].vector
        entry_score = cosine_similarity(query, entry_vector)

        # Max-heap of candidate nodes (negative score for max-heap behavior)
        candidates: list[tuple[float, int]] = [(-entry_score, entry_id)]
        # Min-heap of nearest neighbors found so far (keep best ef)
        nearest: list[tuple[float, int]] = [(entry_score, entry_id)]
        heapq.heapify(candidates)
        heapq.heapify(nearest)

        while candidates:
            neg_score, cand_id = heapq.heappop(candidates)
            cand_score = -neg_score

            # If we have enough candidates and this one is worse than our worst, stop
            if len(nearest) >= ef and cand_score < nearest[0][0]:
                break

            node = self.nodes[cand_id]
            neighbor_ids = node.layers.get(level, [])

            for neighbor_id in neighbor_ids:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor_vector = self.nodes[neighbor_id].vector
                score = cosine_similarity(query, neighbor_vector)

                if len(nearest) < ef or score > nearest[0][0]:
                    heapq.heappush(candidates, (-score, neighbor_id))
                    heapq.heappush(nearest, (score, neighbor_id))

                    if len(nearest) > ef:
                        heapq.heappop(nearest)

        # Return sorted by score descending (best first)
        nearest.sort(key=lambda x: -x[0])
        return nearest

    def add(self, vector_id: int, vector: np.ndarray) -> None:
        """Add a vector to the HNSW index."""
        level = self._random_level()
        node = HNSWNode(vector_id=vector_id, vector=vector)

        if self.entry_point is None:
            self.nodes[vector_id] = node
            self.entry_point = vector_id
            self.max_level = level
            for l in range(level + 1):
                node.layers[l] = []
            return

        # Search from top level down to level+1 to find entry point
        current_id = self.entry_point
        for l in range(self.max_level, level, -1):
            neighbors = self._search_layer(vector, current_id, l, ef=self.ef_construction)
            current_id = neighbors[0][1]

        # Search and connect at levels 0..level
        for l in range(level, -1, -1):
            neighbors = self._search_layer(vector, current_id, l, self.ef_construction)
            neighbor_ids = [nid for _, nid in neighbors[: self.M]]
            node.layers[l] = neighbor_ids

            # Update neighbors' layer lists (bidirectional links)
            for nid in neighbor_ids:
                neighbor_node = self.nodes[nid]
                if l not in neighbor_node.layers:
                    neighbor_node.layers[l] = []
                if vector_id not in neighbor_node.layers[l]:
                    neighbor_node.layers[l].append(vector_id)
                    # Prune if exceeding M*2 (simplified)
                    if len(neighbor_node.layers[l]) > self.M * 2:
                        neighbor_vectors = [
                            (cosine_similarity(self.nodes[nid].vector, self.nodes[oid].vector), oid)
                            for oid in neighbor_node.layers[l]
                        ]
                        neighbor_vectors.sort(reverse=True)
                        neighbor_node.layers[l] = [oid for _, oid in neighbor_vectors[: self.M]]

            if neighbors:
                current_id = neighbors[0][1]

        self.nodes[vector_id] = node

        if level > self.max_level:
            self.entry_point = vector_id
            self.max_level = level

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Approximate nearest neighbor search."""
        if self.entry_point is None:
            return []

        current_id = self.entry_point
        for l in range(self.max_level, 0, -1):
            neighbors = self._search_layer(query, current_id, l, ef=1)
            current_id = neighbors[0][1]

        neighbors = self._search_layer(query, current_id, 0, self.ef_search)
        return [(nid, score) for score, nid in neighbors[:top_k]]

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        level_counts: dict[int, int] = {}
        for node in self.nodes.values():
            for l in node.layers:
                level_counts[l] = level_counts.get(l, 0) + 1
        return {
            "num_nodes": len(self.nodes),
            "max_level": self.max_level,
            "level_distribution": dict(sorted(level_counts.items())),
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "M": self.M,
        }


def generate_embeddings(model: Any, texts: Sequence[str]) -> np.ndarray:
    """Encode texts into embeddings using SentenceTransformers."""
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_model(model_name: str) -> Any:
    """Load a SentenceTransformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise SystemExit("sentence-transformers required. Install: pip install sentence-transformers")
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = getattr(model, "get_embedding_dimension", lambda: model.get_sentence_embedding_dimension())()
    print(f"Dimension: {dim}, Device: {model.device}")
    return model


def demo_brute_force(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Section 1: Exact brute-force nearest neighbor search."""
    print_section("1. BRUTE-FORCE EXACT SEARCH (O(n) baseline)")

    print("How it works:")
    print("  • Compute cosine similarity between the query vector and every stored vector")
    print("  • Return the top-k highest-scoring vectors")
    print("  • Guaranteed to find the true nearest neighbors")
    print("  • Cost: O(n * d) per query — too slow for millions of vectors\n")

    query_text = "A small feline is sleeping in the sun."
    # Use a known document vector as the query for deterministic results
    query_vector = embeddings[0]  # "The cat sat on the warm windowsill."

    start = time.perf_counter()
    results = brute_force_search(query_vector, embeddings, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Query: {query_text!r}")
    print(f"Documents: {len(documents)}")
    print(f"Time: {elapsed_ms:.2f} ms\n")
    print(f"{'Rank':>4}  {'Score':>8s}  {'Cat':<11s}  Document")
    print("-" * 96)
    for rank, (idx, score) in enumerate(results, 1):
        doc = documents[idx]
        print(f"{rank:>4d}  {score:>8.4f}  {doc.category:<11s}  {doc.text}")

    print("\n✅ Exact — but O(n) per query doesn't scale.")


def demo_hnsw_concept(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Section 2: Conceptual HNSW graph navigation."""
    print_section("2. HNSW (Hierarchical Navigable Small World) — Conceptual")

    print("How HNSW works:")
    print("  • Builds multiple layers of a graph, like skiplists")
    print("  • Top layers: sparse, long-range edges for fast coarse navigation")
    print("  • Bottom layer (L0): dense, full graph for fine-grained refinement")
    print("  • Search starts at the top layer, greedily descends to L0")
    print("  • Parameters:")
    print("    - M: max connections per node per layer")
    print("    - ef_construction: candidate list size during build (accuracy vs build time)")
    print("    - ef_search: candidate list size during search (accuracy vs query time)\n")

    hnsw = SimpleHNSW(
        dimension=embeddings.shape[1],
        ef_construction=DEFAULT_EF_CONSTRUCTION,
        ef_search=DEFAULT_EF_SEARCH,
        M=DEFAULT_M,
    )

    print(f"Building HNSW index with {len(documents)} vectors...")
    build_start = time.perf_counter()
    for i, doc in enumerate(documents):
        hnsw.add(i, embeddings[i])
    build_ms = (time.perf_counter() - build_start) * 1000

    stats = hnsw.stats()
    print(f"Build time: {build_ms:.2f} ms")
    print(f"Nodes: {stats['num_nodes']}, Max level: {stats['max_level']}")
    print(f"Level distribution: {stats['level_distribution']}")

    # Visualize a few nodes' connections
    print("\nSample node connections (layer 0):")
    for i in (0, 5, 10, 15):
        node = hnsw.nodes[i]
        neighbors = node.layers.get(0, [])
        neighbor_texts = [documents[n].text[:30] for n in neighbors]
        print(f"  [{i}] {documents[i].text[:35]!r} -> {neighbor_texts}")

    # Search
    query_vector = embeddings[0]
    search_start = time.perf_counter()
    hnsw_results = hnsw.search(query_vector, top_k)
    search_ms = (time.perf_counter() - search_start) * 1000

    print(f"\nQuery: {documents[0].text!r}")
    print(f"HNSW search time: {search_ms:.2f} ms")
    print(f"{'Rank':>4}  {'Score':>8s}  {'Cat':<11s}  Document")
    print("-" * 96)
    for rank, (idx, score) in enumerate(hnsw_results, 1):
        doc = documents[idx]
        print(f"{rank:>4d}  {score:>8.4f}  {doc.category:<11s}  {doc.text}")

    print("\n⚠️  Approximate — fast, but may miss true nearest neighbors.")


def demo_hnsw_recall(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Section 3: HNSW recall evaluation vs brute force."""
    print_section("3. RECALL EVALUATION: HNSW vs Brute Force")

    print("Recall = (true neighbors found by HNSW) / (true neighbors from brute force)")
    print("Higher ef_search → higher recall, slower queries.\n")

    brute_results_map: dict[int, set[int]] = {}
    hnsw_results_map: dict[int, set[int]] = {}

    for query_idx in (0, 5, 10, 15):
        query_vector = embeddings[query_idx]
        brute = set(idx for idx, _ in brute_force_search(query_vector, embeddings, top_k))

        hnsw = SimpleHNSW(
            dimension=embeddings.shape[1],
            ef_construction=DEFAULT_EF_CONSTRUCTION,
            ef_search=DEFAULT_EF_SEARCH,
            M=DEFAULT_M,
        )
        for i, doc in enumerate(documents):
            hnsw.add(i, embeddings[i])
        hnsw_res = set(idx for idx, _ in hnsw.search(query_vector, top_k))

        brute_results_map[query_idx] = brute
        hnsw_results_map[query_idx] = hnsw_res

        overlap = brute & hnsw_res
        recall = len(overlap) / len(brute) if brute else 0.0
        print(f"Query [{query_idx}] {documents[query_idx].text[:40]!r}")
        print(f"  Brute force:  {sorted(brute)}")
        print(f"  HNSW (ef={DEFAULT_EF_SEARCH}): {sorted(hnsw_res)}")
        print(f"  Recall: {recall:.0%}  Overlap: {sorted(overlap)}")
        print()

    # Show effect of ef_search
    print("Effect of ef_search on recall (query [0], top_k=5):")
    print(f"{'ef_search':>10s}  {'Recall':>8s}")
    print("-" * 25)
    for ef in (10, 25, 50, 100, 200):
        query_vector = embeddings[0]
        brute = set(idx for idx, _ in brute_force_search(query_vector, embeddings, top_k))

        hnsw = SimpleHNSW(dimension=embeddings.shape[1], ef_search=ef, M=DEFAULT_M)
        for i, doc in enumerate(documents):
            hnsw.add(i, embeddings[i])
        hnsw_res = set(idx for idx, _ in hnsw.search(query_vector, top_k))

        recall = len(brute & hnsw_res) / len(brute) if brute else 0.0
        print(f"  {ef:>10d}  {recall:>7.0%}")

    print("\nTrade-off: higher ef_search = better recall, slower queries.")


def demo_chromadb(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Section 4: ChromaDB integration (real vector database)."""
    print_section("4. CHROMADB — Production Vector Database")

    if not CHROMADB_AVAILABLE:
        print("chromadb not installed. Install: pip install chromadb")
        print("Skipping ChromaDB section.\n")
        return

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("sentence-transformers not available. Skipping ChromaDB section.\n")
        return

    print("ChromaDB is a local, persistent vector database with:")
    print("  • HNSW indexing (default)")
    print("  • Metadata filtering (WHERE clauses)")
    print("  • Persistence to disk")
    print("  • CRUD operations for documents/chunks\n")

    db_path = Path(__file__).resolve().parent / "output" / "chroma_demo_db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    # Clean slate for demo
    try:
        client.delete_collection("demo_collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="demo_collection",
        metadata={"hnsw:construction_ef": DEFAULT_EF_CONSTRUCTION, "hnsw:search_ef": DEFAULT_EF_SEARCH},
    )

    texts = [doc.text for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [str(doc.doc_id) for doc in documents]

    print(f"Adding {len(documents)} documents to ChromaDB...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"Collection count: {collection.count()}\n")

    # Query 1: semantic search without filter
    query_text = "A small feline is sleeping in the sun."
    model = SentenceTransformer(DEFAULT_MODEL)
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]

    print(f"Query (no filter): {query_text!r}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    print(f"{'Rank':>4}  {'Distance':>9s}  {'Cat':<11s}  Document")
    print("-" * 96)
    for rank in range(len(results["ids"][0])):
        doc_id = int(results["ids"][0][rank])
        distance = results["distances"][0][rank]
        doc_text = results["documents"][0][rank]
        doc_meta = results["metadatas"][0][rank]
        category = doc_meta.get("topic", "unknown")
        print(f"{rank:>4d}  {distance:>9.4f}  {category:<11s}  {doc_text}")

    # Query 2: with metadata filter
    print(f"\nQuery (metadata filter: topic == 'Space'): {query_text!r}")
    filtered = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"topic": "astronomy"},
    )
    print(f"{'Rank':>4}  {'Distance':>9s}  {'Cat':<11s}  Document")
    print("-" * 96)
    for rank in range(len(filtered["ids"][0])):
        doc_id = int(filtered["ids"][0][rank])
        distance = filtered["distances"][0][rank]
        doc_text = filtered["documents"][0][rank]
        doc_meta = filtered["metadatas"][0][rank]
        category = doc_meta.get("topic", "unknown")
        print(f"{rank:>4d}  {distance:>9.4f}  {category:<11s}  {doc_text}")

    print("\n✅ ChromaDB handles indexing, persistence, and filtering automatically.")


def demo_benchmark(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Section 5: Performance benchmark at different scales."""
    print_section("5. PERFORMANCE BENCHMARK: Exact vs HNSW at Scale")

    sizes = [100, 500, 1000]
    print(f"{'Scale':>8s}  {'Brute Force':>14s}  {'HNSW':>12s}  {'Speedup':>10s}")
    print("-" * 52)

    rng = np.random.default_rng(42)

    for n in sizes:
        # Generate random vectors of the target dimension
        dim = embeddings.shape[1]
        test_embeddings = rng.normal(size=(n, dim)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / np.clip(norms, 1e-12, None)

        query = rng.normal(size=dim).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Brute force
        start = time.perf_counter()
        brute_force_search(query, test_embeddings, top_k)
        brute_ms = (time.perf_counter() - start) * 1000

        # HNSW
        hnsw = SimpleHNSW(dimension=dim, ef_search=DEFAULT_EF_SEARCH, M=DEFAULT_M)
        for i in range(n):
            hnsw.add(i, test_embeddings[i])
        start = time.perf_counter()
        hnsw.search(query, top_k)
        hnsw_ms = (time.perf_counter() - start) * 1000

        speedup = brute_ms / hnsw_ms if hnsw_ms > 0 else 0.0
        print(f"  {n:>7d}  {brute_ms:>12.2f} ms  {hnsw_ms:>10.2f} ms  {speedup:>8.1f}x")

    print("\n✅ HNSW advantage grows with dataset size.")


def demo_corpus(documents: list[Document]) -> None:
    """Print the document corpus."""
    print_section("DOCUMENT CORPUS")
    print(f"{len(documents)} documents across categories:\n")
    for doc in documents:
        print(f"  [{doc.doc_id:2d}] {doc.category:<11s} {doc.text}")


def interactive_loop(embeddings: np.ndarray, documents: list[Document], top_k: int) -> None:
    """Interactive query mode comparing brute force vs HNSW."""
    print_section("INTERACTIVE MODE")
    print("Type a sentence to search. Compare brute force vs HNSW results.")
    print("Commands: /help, /quit\n")

    hnsw = SimpleHNSW(
        dimension=embeddings.shape[1],
        ef_search=DEFAULT_EF_SEARCH,
        M=DEFAULT_M,
    )
    for i, doc in enumerate(documents):
        hnsw.add(i, embeddings[i])

    while True:
        try:
            raw = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLeaving interactive mode.")
            break

        if not raw:
            continue
        if raw.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if raw.lower() in {"/help", "help", "?"}:
            print("  <sentence>   search with brute force + HNSW")
            print("  /help        show this help")
            print("  /quit        exit")
            continue

        model = load_model(DEFAULT_MODEL)
        query_vector = generate_embeddings(model, [raw])[0]

        brute = brute_force_search(query_vector, embeddings, top_k)
        hnsw_res = hnsw.search(query_vector, top_k)

        print(f"\nQuery: {raw!r}")
        print(f"{'Rank':>4}  {'Brute':>8s}  {'HNSW':>8s}  Document")
        print("-" * 80)
        brute_dict = dict(brute)
        hnsw_dict = dict(hnsw_res)
        
        # Show top-k from brute force and HNSW side by side
        for rank in range(1, top_k + 1):
            # Get rank-th result from brute force
            if rank <= len(brute):
                b_idx, b_score = brute[rank - 1]
                b_str = f"{b_score:>8.4f}"
            else:
                b_str = "   —   "
            
            # Get rank-th result from HNSW
            if rank <= len(hnsw_res):
                h_idx, h_score = hnsw_res[rank - 1]
                h_str = f"{h_score:>8.4f}"
            else:
                h_str = "   —   "
            
            # Show document from brute force if available, else from HNSW
            doc_idx = brute[rank - 1][0] if rank <= len(brute) else (hnsw_res[rank - 1][0] if rank <= len(hnsw_res) else None)
            doc_text = documents[doc_idx].text if doc_idx is not None else ""
            
            print(f"{rank:>4d}  {b_str}  {h_str}  {doc_text}")
        print()


def self_test() -> None:
    """Fast self-tests without loading a model."""
    print_section("SELF-TEST: SEARCH & HNSW MATH")

    # Cosine similarity sanity checks
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])

    checks = [
        ("identical vectors", cosine_similarity(a, b), 1.0),
        ("orthogonal vectors", cosine_similarity(a, c), 0.0),
        ("opposite vectors", cosine_similarity(a, d), -1.0),
    ]
    for label, actual, expected in checks:
        assert np.isclose(actual, expected), f"{label}: {actual} != {expected}"
        print(f"  PASS  {label:<20s} cosine = {actual:+.4f}")

    # Brute force returns correct top-k
    matrix = np.vstack([a, b, c, d])
    results = brute_force_search(a, matrix, 2)
    assert len(results) == 2
    assert results[0][0] in (0, 1)  # a or b (identical to query)
    print("  PASS  brute force top-2 includes identical vectors")

    # HNSW adds and searches
    hnsw = SimpleHNSW(dimension=3, ef_search=10, M=4)
    for i in range(4):
        hnsw.add(i, matrix[i])
    res = hnsw.search(a, 2)
    assert len(res) == 2
    print("  PASS  HNSW add + search returns results")

    # HNSW stats
    stats = hnsw.stats()
    assert stats["num_nodes"] == 4
    assert "max_level" in stats
    print("  PASS  HNSW stats reported correctly")

    print("\nAll self-tests passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate vector databases, HNSW ANN, and ChromaDB integration."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Results to show")
    parser.add_argument("--query", action="append", default=[], help="One-shot query (may repeat)")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark at scale")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests only, then exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    print("\n" + "=" * 80)
    print("  SESSION 1.4: VECTOR DATABASES AND ANN ALGORITHMS")
    print("  From brute-force search to HNSW graphs to ChromaDB")
    print("=" * 80)

    demo_corpus(DOCUMENTS)

    model = None
    embeddings = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        model = load_model(args.model)
        embeddings = generate_embeddings(model, [doc.text for doc in DOCUMENTS])
    else:
        print("sentence-transformers not available; using random vectors for demo sections.")
        rng = np.random.default_rng(42)
        dim = DEFAULT_DIMENSION
        embeddings = rng.normal(size=(len(DOCUMENTS), dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

    demo_brute_force(embeddings, DOCUMENTS, args.top_k)
    demo_hnsw_concept(embeddings, DOCUMENTS, args.top_k)
    demo_hnsw_recall(embeddings, DOCUMENTS, args.top_k)
    demo_chromadb(embeddings, DOCUMENTS, args.top_k)

    if args.benchmark:
        demo_benchmark(embeddings, DOCUMENTS, args.top_k)

    # One-shot queries
    if args.query and model is not None:
        for query in args.query:
            print(f"\n{'=' * 80}")
            print(f"QUERY: {query!r}")
            print(f"{'=' * 80}")
            query_vector = generate_embeddings(model, [query])[0]
            brute = brute_force_search(query_vector, embeddings, args.top_k)
            print(f"\nBrute force results:")
            print(f"{'Rank':>4}  {'Score':>8s}  {'Cat':<11s}  Document")
            print("-" * 96)
            for rank, (idx, score) in enumerate(brute, 1):
                doc = DOCUMENTS[idx]
                print(f"{rank:>4d}  {score:>8.4f}  {doc.category:<11s}  {doc.text}")

    # Interactive mode
    should_interact = args.interactive or (not args.query and sys.stdin.isatty())
    if should_interact and embeddings is not None:
        interactive_loop(embeddings, DOCUMENTS, args.top_k)

    print("\n" + "=" * 80)
    print("  SESSION 1.4 COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Brute-force search is exact but O(n) — doesn't scale to millions of vectors")
    print("  • HNSW builds layered graphs for fast approximate nearest-neighbor search")
    print("  • ef_search controls recall vs latency trade-off at query time")
    print("  • ef_construction controls build quality vs build time")
    print("  • ChromaDB wraps HNSW with persistence, metadata filtering, and CRUD")
    print("  • Metadata filtering (WHERE) narrows search before vector similarity")
    print("  • Production RAG systems combine vector DB + metadata filters + hybrid search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
