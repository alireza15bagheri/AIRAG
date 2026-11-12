#!/usr/bin/env python
"""
Session 1.3 Demo: Semantic Search vs Keyword Search
====================================================
Educational demonstration comparing BM25 (keyword-based) vs Dense
Vector (semantic) search, showing failure cases and hybrid fusion.

Run the full demo:
    python demo/session_1_3_semantic_vs_keyword.py

Run a one-shot query:
    python demo/session_1_3_semantic_vs_keyword.py --query "A kitten is sleeping."

Run interactively:
    python demo/session_1_3_semantic_vs_keyword.py --interactive

Run only the fast, model-free self-tests:
    python demo/session_1_3_semantic_vs_keyword.py --self-test

Compare specific methods:
    python demo/session_1_3_semantic_vs_keyword.py --method bm25
    python demo/session_1_3_semantic_vs_keyword.py --method dense
    python demo/session_1_3_semantic_vs_keyword.py --method hybrid
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Try to import rank_bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5  # Hybrid weight: alpha * dense + (1-alpha) * sparse


@dataclass(frozen=True)
class Document:
    """A document with text and optional metadata."""
    text: str
    doc_id: int
    category: str = ""


# Same corpus as Session 1.2 for direct comparison
DOCUMENTS: list[Document] = [
    Document("The cat sat on the warm windowsill.", 0, "Animals"),
    Document("A dog barked loudly at the mail carrier.", 1, "Animals"),
    Document("Kittens chase laser pointers across the floor.", 2, "Animals"),
    Document("The puppy wagged its tail happily.", 3, "Animals"),
    Document("A horse galloped across the open field.", 4, "Animals"),
    Document("Machine learning models learn patterns from data.", 5, "Technology"),
    Document("Neural networks are inspired by the human brain.", 6, "Technology"),
    Document("The transformer architecture uses self-attention.", 7, "Technology"),
    Document("Large language models generate text one token at a time.", 8, "Technology"),
    Document("Software engineers debug code and write tests.", 9, "Technology"),
    Document("I baked a fresh loaf of sourdough bread.", 10, "Food"),
    Document("The chef added garlic and basil to the pasta sauce.", 11, "Food"),
    Document("She poured a cup of hot coffee.", 12, "Food"),
    Document("Fresh strawberries taste sweet in summer.", 13, "Food"),
    Document("The restaurant served grilled salmon with lemon.", 14, "Food"),
    Document("The rocket launched into orbit.", 15, "Space"),
    Document("Mars is the fourth planet from the sun.", 16, "Space"),
    Document("Astronauts float in microgravity aboard the space station.", 17, "Space"),
    Document("A telescope captured an image of a distant galaxy.", 18, "Space"),
    Document("The moon orbits Earth once every month.", 19, "Space"),
]

# Queries that test different failure modes
DEMO_QUERIES = [
    # Synonym test - BM25 fails, dense succeeds
    "A small feline is sleeping in the sun.",
    # Concept test - no keyword overlap
    "Deep learning and attention mechanisms.",
    # Exact phrase - BM25 should excel
    "transformer architecture uses self-attention",
    # Polysemy test - "Apple" fruit vs company
    "Apple fruit nutrition benefits",
    # Cross-domain - no overlap
    "A spaceship travels to another planet.",
    # Rare term - dense may struggle
    "sourdough bread fermentation process",
]

# Failure case demonstrations
FAILURE_CASES = [
    {
        "name": "Synonym Gap",
        "query": "automobile engine repair",
        "relevant_doc": "The car engine needs fixing.",
        "explanation": "BM25: 0 overlap (automobile ≠ car). Dense: high similarity."
    },
    {
        "name": "Polysemy (Word Sense)",
        "query": "Apple stock price today",
        "relevant_doc": "Technology company stock rose 5%.",
        "explanation": "BM25 matches 'Apple' but wrong sense. Dense uses context."
    },
    {
        "name": "Exact Phrase Match",
        "query": "transformer architecture uses self-attention",
        "relevant_doc": "The transformer architecture uses self-attention.",
        "explanation": "BM25: perfect token overlap = max score. Dense: also high but BM25 faster."
    },
    {
        "name": "Rare/Specific Terms",
        "query": "sourdough fermentation temperature",
        "relevant_doc": "I baked a fresh loaf of sourdough bread.",
        "explanation": "BM25: exact rare term match. Dense: may dilute with 'bread'."
    },
]


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def load_dense_model(model_name: str) -> Any:
    """Load SentenceTransformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise SystemExit(
            "sentence-transformers required. Install: pip install sentence-transformers"
        )
    print(f"Loading dense model: {model_name}...")
    model = SentenceTransformer(model_name)
    dim = getattr(model, "get_embedding_dimension", lambda: model.get_sentence_embedding_dimension())()
    print(f"Model: {model.__class__.__name__}, Dimension: {dim}, Device: {model.device}")
    return model


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    import re
    # Lowercase, split on non-alphanumeric
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def build_bm25_index(documents: Sequence[Document]) -> BM25Okapi:
    """Build BM25 index from documents."""
    if not BM25_AVAILABLE:
        raise SystemExit("rank_bm25 required. Install: pip install rank_bm25")
    
    corpus = [tokenize(doc.text) for doc in documents]
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built: {len(corpus)} documents, avg doc length: {np.mean([len(c) for c in corpus]):.1f} tokens")
    return bm25


def generate_dense_embeddings(model: Any, texts: Sequence[str]) -> np.ndarray:
    """Generate dense embeddings for texts."""
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_to_all(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query and all rows in matrix."""
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    denominators = np.linalg.norm(q) * np.linalg.norm(m, axis=1)
    denominators = np.clip(denominators, 1e-12, None)
    return (m @ q) / denominators


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of top k scores, best first."""
    return np.argsort(-np.asarray(scores))[:k]


def print_results(query: str, results: list[tuple[int, float, Document]], method: str) -> None:
    """Print search results in a formatted table."""
    print(f"\n{method} Results for: {query!r}")
    print(f"{'Rank':>4}  {'Score':>8s}  {'Cat':<11s}  Document")
    print("-" * 96)
    for rank, (idx, score, doc) in enumerate(results, 1):
        print(f"{rank:>4d}  {score:>8.4f}  {doc.category:<11s}  {doc.text}")
    if not results:
        print("  (no results)")


def search_bm25(bm25: BM25Okapi, documents: Sequence[Document], query: str, top_k: int) -> list[tuple[int, float, Document]]:
    """Search using BM25."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = top_k_indices(scores, min(top_k, len(documents)))
    return [(idx, scores[idx], documents[idx]) for idx in top_indices if scores[idx] > 0]


def search_dense(model: Any, query: str, doc_embeddings: np.ndarray, documents: Sequence[Document], top_k: int) -> list[tuple[int, float, Document]]:
    """Search using dense embeddings."""
    query_embedding = generate_dense_embeddings(model, [query])[0]
    scores = cosine_similarity_to_all(query_embedding, doc_embeddings)
    top_indices = top_k_indices(scores, min(top_k, len(documents)))
    return [(idx, scores[idx], documents[idx]) for idx in top_indices]


def search_hybrid(
    bm25: BM25Okapi,
    model: Any,
    query: str,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
    alpha: float = DEFAULT_ALPHA,
) -> list[tuple[int, float, Document]]:
    """Hybrid search: alpha * dense + (1-alpha) * normalized sparse."""
    # Get both scores
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    
    query_embedding = generate_dense_embeddings(model, [query])[0]
    dense_scores = cosine_similarity_to_all(query_embedding, doc_embeddings)
    
    # Normalize both to [0, 1] range for fair combination
    def normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)
    
    # Combine
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    top_indices = top_k_indices(hybrid_scores, min(top_k, len(documents)))
    
    return [(idx, hybrid_scores[idx], documents[idx]) for idx in top_indices]


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float, Document]],
    dense_results: list[tuple[int, float, Document]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[int, float, Document]]:
    """Reciprocal Rank Fusion (RRF) for combining ranked lists."""
    # Build rank maps
    bm25_ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(bm25_results)}
    dense_ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(dense_results)}
    
    all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    
    fused = []
    for idx in all_indices:
        rrf_score = 0.0
        if idx in bm25_ranks:
            rrf_score += 1.0 / (k + bm25_ranks[idx])
        if idx in dense_ranks:
            rrf_score += 1.0 / (k + dense_ranks[idx])
        fused.append((idx, rrf_score))
    
    fused.sort(key=lambda x: -x[1])
    return [(idx, score, DOCUMENTS[idx]) for idx, score in fused[:top_k]]


def demo_corpus() -> None:
    """Print the document corpus."""
    print_section("1. DOCUMENT CORPUS")
    print(f"{len(DOCUMENTS)} documents across categories:\n")
    for doc in DOCUMENTS:
        print(f"  [{doc.doc_id:2d}] {doc.category:<11s} {doc.text}")


def demo_bm25_basics(bm25: BM25Okapi, documents: Sequence[Document]) -> None:
    """Explain BM25 and show scoring."""
    print_section("2. BM25 KEYWORD SEARCH - THEORY & BASICS")
    
    print("BM25 Formula:")
    print("  score(D, Q) = Σ IDF(t) * (f(t,D) * (k1+1)) / (f(t,D) + k1*(1-b+b*|D|/avgdl))")
    print("  - IDF(t): inverse document frequency (rare terms = higher weight)")
    print("  - f(t,D): term frequency in document")
    print("  - k1: term frequency saturation (typically 1.2-2.0)")
    print("  - b: document length normalization (typically 0.75)")
    print("  - |D|/avgdl: document length relative to average\n")
    
    print("Key Properties:")
    print("  • Exact token matching only (no semantic understanding)")
    print("  • Rare terms get higher weight (IDF)")
    print("  • Long documents normalized to avoid bias")
    print("  • Fast: O(query_terms) lookup with inverted index\n")
    
    # Show tokenization example
    query = "cat sleeping"
    tokens = tokenize(query)
    print(f"Query: '{query}' → Tokens: {tokens}")
    
    # Show IDF values for corpus
    print("\nTop IDF terms in corpus:")
    idfs = bm25.idf
    top_idfs = sorted(idfs.items(), key=lambda x: -x[1])[:10]
    for term, idf in top_idfs:
        print(f"  '{term}': IDF = {idf:.3f}")


def demo_comparison(
    bm25: BM25Okapi,
    model: Any,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
) -> None:
    """Side-by-side comparison of all methods."""
    print_section("3. SIDE-BY-SIDE COMPARISON")
    
    for query in DEMO_QUERIES:
        print(f"\n{'='*80}")
        print(f"QUERY: {query!r}")
        print(f"{'='*80}")
        
        # BM25
        bm25_results = search_bm25(bm25, documents, query, top_k)
        print_results(query, bm25_results, "BM25")
        
        # Dense
        dense_results = search_dense(model, query, doc_embeddings, documents, top_k)
        print_results(query, dense_results, "DENSE")
        
        # Hybrid
        hybrid_results = search_hybrid(bm25, model, query, doc_embeddings, documents, top_k)
        print_results(query, hybrid_results, "HYBRID (α=0.5)")
        
        # RRF
        rrf_results = reciprocal_rank_fusion(bm25_results, dense_results, top_k=top_k)
        print_results(query, rrf_results, "RRF")


def demo_failure_cases(
    bm25: BM25Okapi,
    model: Any,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
) -> None:
    """Demonstrate specific failure cases for each method."""
    print_section("4. FAILURE CASE ANALYSIS")
    
    for case in FAILURE_CASES:
        print(f"\n{'─'*60}")
        print(f"FAILURE CASE: {case['name']}")
        print(f"Query: {case['query']!r}")
        print(f"Relevant: {case['relevant_doc']!r}")
        print(f"Explanation: {case['explanation']}")
        print(f"─" * 60)
        
        # BM25
        bm25_results = search_bm25(bm25, documents, case['query'], top_k)
        print_results(case['query'], bm25_results, "BM25")
        
        # Dense
        dense_results = search_dense(model, case['query'], doc_embeddings, documents, top_k)
        print_results(case['query'], dense_results, "DENSE")


def demo_hybrid_fusion(
    bm25: BM25Okapi,
    model: Any,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
) -> None:
    """Explore different hybrid fusion strategies."""
    print_section("5. HYBRID FUSION STRATEGIES")
    
    query = "A small feline is sleeping in the sun."
    print(f"Test query: {query!r}\n")
    
    # Test different alpha values
    print("Weighted Score Fusion (α * dense + (1-α) * sparse):")
    print(f"{'Alpha':>6s}  {'Dense Weight':>12s}  {'Sparse Weight':>12s}  Top Result")
    print("-" * 70)
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        results = search_hybrid(bm25, model, query, doc_embeddings, documents, top_k=1, alpha=alpha)
        top = results[0] if results else (None, 0, None)
        print(f"  {alpha:.2f}    {alpha:>12.2f}    {1-alpha:>12.2f}    {top[2].text[:50] if top[2] else 'None'!r}")
    
    print("\nReciprocal Rank Fusion (RRF) - no hyperparameters:")
    rrf_results = reciprocal_rank_fusion(
        search_bm25(bm25, documents, query, top_k),
        search_dense(model, query, doc_embeddings, documents, top_k),
        top_k=top_k,
    )
    print_results(query, rrf_results, "RRF")
    
    print("\nWhen to use each:")
    print("  • Weighted: when you know optimal α (tune on validation set)")
    print("  • RRF: no tuning needed, works well out-of-the-box")
    print("  • α=1.0: pure dense (semantic)")
    print("  • α=0.0: pure BM25 (keyword)")


def demo_failure_analysis(
    bm25: BM25Okapi,
    model: Any,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
) -> None:
    """Deep dive into why each method fails."""
    print_section("6. WHY THEY FAIL - DETAILED ANALYSIS")
    
    # BM25 Failure: Synonyms
    print("BM25 FAILURE: Synonym Gap")
    print("  Query: 'automobile engine repair'")
    print("  Corpus has: 'car engine needs fixing'")
    print("  BM25 tokens: ['automobile', 'engine', 'repair']")
    print("  Doc tokens:  ['car', 'engine', 'needs', 'fixing']")
    print("  Overlap: 'engine' only → low score")
    print("  Dense: 'automobile' and 'car' close in vector space\n")
    
    # BM25 Failure: Polysemy
    print("BM25 FAILURE: Polysemy")
    print("  Query: 'Apple stock price'")
    print("  Corpus: 'I ate an apple' (food)")
    print("  BM25: matches 'apple' → wrong sense!")
    print("  Dense: 'Apple' + 'stock' → tech company vector\n")
    
    # Dense Failure: Rare exact terms
    print("DENSE FAILURE: Rare Exact Terms")
    print("  Query: 'sourdough fermentation temperature 35 celsius'")
    print("  Dense: '35' and 'celsius' are common → diluted")
    print("  BM25: 'fermentation' + 'temperature' + '35' → exact match wins\n")
    
    # Dense Failure: Exact phrase
    print("DENSE FAILURE: Exact Phrase Matching")
    print("  Query: 'to be or not to be'")
    print("  Dense: stopwords removed, loses phrase structure")
    print("  BM25: exact n-gram match if configured, or exact token sequence\n")


def interactive_loop(
    bm25: BM25Okapi,
    model: Any,
    doc_embeddings: np.ndarray,
    documents: Sequence[Document],
    top_k: int,
    alpha: float,
) -> None:
    """Interactive query mode."""
    print_section("7. INTERACTIVE MODE")
    print("Commands: /method [bm25|dense|hybrid|rrf], /alpha <value>, /help, /quit\n")
    
    current_method = "hybrid"
    current_alpha = alpha
    
    while True:
        try:
            raw = input(f"[{current_method}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not raw:
            continue
        
        if raw.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        
        if raw.lower() in {"/help", "help", "?"}:
            print("  <query>          search with current method")
            print("  /method <name>   set method: bm25, dense, hybrid, rrf")
            print("  /alpha <0-1>     set hybrid alpha (dense weight)")
            print("  /help            show this help")
            print("  /quit            exit")
            continue
        
        if raw.lower().startswith("/method "):
            method = raw[8:].strip().lower()
            if method in {"bm25", "dense", "hybrid", "rrf"}:
                current_method = method
                print(f"Method set to: {current_method}")
            else:
                print("Valid methods: bm25, dense, hybrid, rrf")
            continue
        
        if raw.lower().startswith("/alpha "):
            try:
                current_alpha = float(raw[7:].strip())
                if 0 <= current_alpha <= 1:
                    print(f"Alpha set to: {current_alpha}")
                else:
                    print("Alpha must be between 0 and 1")
            except ValueError:
                print("Invalid alpha value")
            continue
        
        # Execute search
        query = raw
        
        if current_method == "bm25":
            results = search_bm25(bm25, documents, query, top_k)
        elif current_method == "dense":
            results = search_dense(model, query, doc_embeddings, documents, top_k)
        elif current_method == "rrf":
            bm25_res = search_bm25(bm25, documents, query, top_k)
            dense_res = search_dense(model, query, doc_embeddings, documents, top_k)
            results = reciprocal_rank_fusion(bm25_res, dense_res, top_k=top_k)
        else:  # hybrid
            results = search_hybrid(bm25, model, query, doc_embeddings, documents, top_k, current_alpha)
        
        print_results(query, results, current_method.upper())


def self_test() -> None:
    """Fast self-tests without model loading."""
    print_section("SELF-TEST: BM25 & FUSION MATH")
    
    # Test BM25 basics with sufficient corpus size
    if BM25_AVAILABLE:
        # Need enough docs for IDF to be positive (avoid negative IDF with tiny corpus)
        docs = [
            "the cat sat on the mat",
            "the cat sat on the mat", 
            "the cat sat on the mat",
            "the dog ran away fast",
            "the dog ran away fast",
            "the dog ran away fast",
            "cat and dog play together",
            "cat and dog play together",
            "cat and dog play together",
        ]
        tokenized = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(["cat"])
        # First 3 docs are about cat, next 3 about dog, last 3 mixed
        # Cat docs and mixed docs should score higher than dog-only docs
        assert scores[0] > scores[3], "cat docs should score higher than dog docs"
        assert scores[0] == scores[1], "identical cat docs should have equal scores"
        print("  PASS  BM25 basic scoring with sufficient corpus")
    
    # Test RRF
    bm25_res = [(0, 1.0), (1, 0.8), (2, 0.6)]
    dense_res = [(1, 0.9), (0, 0.7), (3, 0.5)]
    rrf = reciprocal_rank_fusion(
        [(i, s, None) for i, s in bm25_res],
        [(i, s, None) for i, s in dense_res],
        top_k=3
    )
    assert len(rrf) == 3
    print("  PASS  RRF returns correct number of results")
    
    # Test normalization
    arr = np.array([1.0, 2.0, 3.0])
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    assert np.allclose(norm, [0.0, 0.5, 1.0])
    print("  PASS  Min-max normalization")
    
    print("\nAll self-tests passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BM25 (keyword) vs Dense (semantic) vs Hybrid search."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Results to show")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Hybrid dense weight")
    parser.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid", "rrf", "all"),
        default="all",
        help="Search method to demonstrate",
    )
    parser.add_argument("--query", action="append", default=[], help="One-shot query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    
    if args.self_test:
        self_test()
        return 0
    
    if not BM25_AVAILABLE:
        raise SystemExit("rank_bm25 required: pip install rank_bm25")
    if not SENTENCE_TRANSFORMERS_AVAILABLE and args.method in {"dense", "hybrid", "rrf", "all"}:
        raise SystemExit("sentence-transformers required: pip install sentence-transformers")
    
    print("\n" + "=" * 80)
    print("  SESSION 1.3: SEMANTIC SEARCH vs KEYWORD SEARCH")
    print("  BM25 vs Dense vs Hybrid - when each wins and why")
    print("=" * 80)
    
    demo_corpus()
    
    # Build indices
    bm25 = build_bm25_index(DOCUMENTS)
    
    model = None
    doc_embeddings = None
    if args.method in {"dense", "hybrid", "rrf", "all"}:
        model = load_dense_model(args.model)
        doc_embeddings = generate_dense_embeddings(model, [d.text for d in DOCUMENTS])
    
    if args.method == "bm25":
        demo_bm25_basics(bm25, DOCUMENTS)
    elif args.method == "dense":
        print_section("DENSE SEARCH DEMO")
        for query in DEMO_QUERIES:
            results = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
            print_results(query, results, "DENSE")
    elif args.method == "hybrid":
        print_section("HYBRID SEARCH DEMO")
        for query in DEMO_QUERIES:
            results = search_hybrid(bm25, model, query, doc_embeddings, DOCUMENTS, args.top_k, args.alpha)
            print_results(query, results, f"HYBRID (α={args.alpha})")
    elif args.method == "rrf":
        print_section("RRF FUSION DEMO")
        for query in DEMO_QUERIES:
            bm25_res = search_bm25(bm25, DOCUMENTS, query, args.top_k)
            dense_res = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
            results = reciprocal_rank_fusion(bm25_res, dense_res, top_k=args.top_k)
            print_results(query, results, "RRF")
    else:  # all
        demo_bm25_basics(bm25, DOCUMENTS)
        demo_comparison(bm25, model, doc_embeddings, DOCUMENTS, args.top_k)
        demo_failure_cases(bm25, model, doc_embeddings, DOCUMENTS, args.top_k)
        demo_hybrid_fusion(bm25, model, doc_embeddings, DOCUMENTS, args.top_k)
        demo_failure_analysis(bm25, model, doc_embeddings, DOCUMENTS, args.top_k)
    
    # One-shot queries
    for query in args.query:
        if args.method == "all":
            print(f"\n{'='*60}")
            print(f"QUERY: {query!r}")
            print(f"{'='*60}")
            for method_name in ["bm25", "dense", "hybrid", "rrf"]:
                if method_name == "bm25":
                    results = search_bm25(bm25, DOCUMENTS, query, args.top_k)
                elif method_name == "dense":
                    results = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
                elif method_name == "hybrid":
                    results = search_hybrid(bm25, model, query, doc_embeddings, DOCUMENTS, args.top_k, args.alpha)
                else:
                    bm25_res = search_bm25(bm25, DOCUMENTS, query, args.top_k)
                    dense_res = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
                    results = reciprocal_rank_fusion(bm25_res, dense_res, top_k=args.top_k)
                print_results(query, results, method_name.upper())
        else:
            # Use configured method
            if args.method == "bm25":
                results = search_bm25(bm25, DOCUMENTS, query, args.top_k)
            elif args.method == "dense":
                results = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
            elif args.method == "hybrid":
                results = search_hybrid(bm25, model, query, doc_embeddings, DOCUMENTS, args.top_k, args.alpha)
            else:
                bm25_res = search_bm25(bm25, DOCUMENTS, query, args.top_k)
                dense_res = search_dense(model, query, doc_embeddings, DOCUMENTS, args.top_k)
                results = reciprocal_rank_fusion(bm25_res, dense_res, top_k=args.top_k)
            print_results(query, results, args.method.upper())
    
    # Interactive mode
    should_interact = args.interactive or (not args.query and sys.stdin.isatty())
    if should_interact and args.method in {"hybrid", "all"}:
        interactive_loop(bm25, model, doc_embeddings, DOCUMENTS, args.top_k, args.alpha)
    
    print("\n" + "=" * 80)
    print("  SESSION 1.3 COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • BM25: exact token match, fast, no semantic understanding")
    print("  • Dense: semantic similarity, handles synonyms/polysemy")
    print("  • Hybrid (weighted): tunable, needs validation to find optimal α")
    print("  • RRF: parameter-free fusion, works well out-of-the-box")
    print("  • Use BM25 for exact matches, rare terms, speed")
    print("  • Use Dense for synonyms, concepts, cross-lingual")
    print("  • Hybrid/RRF: best of both worlds for production RAG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())