#!/usr/bin/env python
"""
Session 1.2 Demo: Text Embeddings and Vector Space
==================================================
Educational demonstration of how sentences become dense vectors, how
cosine similarity measures semantic closeness, and how PCA/t-SNE can
project high-dimensional embeddings into 2D or 3D for inspection.

Run the full demo:
    python demo/session_1_2_embeddings.py

Run a one-shot query:
    python demo/session_1_2_embeddings.py --query "A kitten is sleeping."

Run interactively after the demo:
    python demo/session_1_2_embeddings.py --interactive

Run only the fast, model-free math self-tests:
    python demo/session_1_2_embeddings.py --self-test

Optional graphical plot:
    pip install matplotlib
    python demo/session_1_2_embeddings.py --show

If Matplotlib is not installed, the script automatically prints an ASCII
projection instead, so the demo still works with minimal dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Sample:
    """A sentence plus the topic label used only for visualization."""

    text: str
    category: str


# A small, deliberately clustered corpus. The categories are not used by
# the embedding model; they are only a human-readable check that semantically
# related sentences end up near each other in vector space.
SAMPLES: list[Sample] = [
    Sample("The cat sat on the warm windowsill.", "Animals"),
    Sample("A dog barked loudly at the mail carrier.", "Animals"),
    Sample("Kittens chase laser pointers across the floor.", "Animals"),
    Sample("The puppy wagged its tail happily.", "Animals"),
    Sample("A horse galloped across the open field.", "Animals"),
    Sample("Machine learning models learn patterns from data.", "Technology"),
    Sample("Neural networks are inspired by the human brain.", "Technology"),
    Sample("The transformer architecture uses self-attention.", "Technology"),
    Sample("Large language models generate text one token at a time.", "Technology"),
    Sample("Software engineers debug code and write tests.", "Technology"),
    Sample("I baked a fresh loaf of sourdough bread.", "Food"),
    Sample("The chef added garlic and basil to the pasta sauce.", "Food"),
    Sample("She poured a cup of hot coffee.", "Food"),
    Sample("Fresh strawberries taste sweet in summer.", "Food"),
    Sample("The restaurant served grilled salmon with lemon.", "Food"),
    Sample("The rocket launched into orbit.", "Space"),
    Sample("Mars is the fourth planet from the sun.", "Space"),
    Sample("Astronauts float in microgravity aboard the space station.", "Space"),
    Sample("A telescope captured an image of a distant galaxy.", "Space"),
    Sample("The moon orbits Earth once every month.", "Space"),
]

# Queries that deliberately use different words from the target sentences.
# These show semantic matching rather than exact keyword matching.
DEMO_QUERIES = [
    "A small feline is sleeping in the sun.",
    "Deep learning and attention mechanisms.",
    "A spaceship travels to another planet.",
    "A warm meal with fresh ingredients.",
]

# Pairs used to make the cosine-similarity table concrete.
SIMILARITY_PAIRS = [
    (0, 0, "identical sentence"),
    (0, 2, "cat vs kittens (same topic)"),
    (0, 1, "cat vs dog (both animals)"),
    (0, 15, "cat vs rocket (different topics)"),
    (0, 13, "cat vs strawberries (unrelated)"),
    (5, 7, "ML vs transformers (same topic)"),
    (10, 11, "bread vs pasta (both food)"),
    (15, 16, "rocket vs Mars (both space)"),
    (5, 13, "ML vs strawberries (unrelated)"),
]


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def load_embedding_model(model_name: str) -> Any:
    """Load a Sentence-Transformers model and report its key properties."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for this demo.\n"
            "Install it with: pip install sentence-transformers"
        ) from exc

    print(f"Loading model: {model_name}")
    print("(The first run downloads the model; later runs use the local cache.)")
    model = SentenceTransformer(model_name)
    # Sentence-Transformers renamed this method in newer releases; support both.
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if callable(dimension_getter):
        dimension = dimension_getter()
    else:
        dimension = model.get_sentence_embedding_dimension()

    print(f"Model class:         {model.__class__.__name__}")
    print(f"Device:              {model.device}")
    print(f"Embedding dimension: {dimension}")
    print("Each sentence will become one vector of this size.")
    return model


def generate_embeddings(
    model: Any,
    texts: Sequence[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts into a 2D array of shape (number_of_texts, dimensions).

    ``normalize=True`` scales every vector to unit length. This is the
    standard setup for cosine similarity because the dot product of two
    unit-length vectors is exactly their cosine similarity.
    """
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors.

    Formula:
        cosine(A, B) = (A · B) / (||A|| * ||B||)

    Result range is [-1, 1]:
        1.0  -> same direction (very similar meaning)
        0.0  -> orthogonal (unrelated)
       -1.0  -> opposite direction
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def cosine_similarity_to_all(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and every row."""
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    denominators = np.linalg.norm(q) * np.linalg.norm(m, axis=1)
    denominators = np.clip(denominators, 1e-12, None)
    return (m @ q) / denominators


def similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute the full pairwise cosine-similarity matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    normalized = m / np.clip(norms, 1e-12, None)
    return normalized @ normalized.T


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k highest scores, best first."""
    return np.argsort(-np.asarray(scores))[:k]


def demo_sample_corpus(samples: Sequence[Sample]) -> None:
    """Print the corpus so indices used later are easy to follow."""
    print_section("1. SAMPLE CORPUS")
    print(f"{len(samples)} sentences across four topics:\n")
    for index, sample in enumerate(samples):
        print(f"  [{index:2d}] {sample.category:<11s} {sample.text}")


def demo_embedding_generation(
    model: Any,
    samples: Sequence[Sample],
    batch_size: int,
) -> np.ndarray:
    """Generate and inspect embeddings for the sample corpus."""
    print_section("2. GENERATING EMBEDDINGS")

    texts = [sample.text for sample in samples]
    embeddings = generate_embeddings(model, texts, batch_size=batch_size)

    print(f"Sentences:        {len(texts)}")
    print(f"Embedding matrix: {embeddings.shape}  (sentences x dimensions)")
    print(f"Data type:        {embeddings.dtype}")
    print()
    print("A matrix row is one sentence's dense semantic vector.")
    print("First 8 dimensions of sentence [0]:")
    print("  " + np.array2string(embeddings[0, :8], precision=4, suppress_small=True))

    norms = np.linalg.norm(embeddings, axis=1)
    print()
    print(
        "Vector norms: "
        f"min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
    )
    print("Because normalize_embeddings=True, every norm is approximately 1.0.")
    print("That makes dot product and cosine similarity equivalent.")
    return embeddings


def demo_cosine_similarity(
    samples: Sequence[Sample],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Show the cosine formula, concrete pairs, and the full matrix."""
    print_section("3. COSINE SIMILARITY: MEASURING SEMANTIC CLOSENESS")

    print("Formula:")
    print("  cosine(A, B) = (A · B) / (||A|| * ||B||)")
    print("  +1 = same direction, 0 = unrelated, -1 = opposite direction\n")

    # Show the arithmetic for one concrete pair instead of only the result.
    left, right = 0, 2
    a = embeddings[left]
    b = embeddings[right]
    dot_product = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    score = cosine_similarity(a, b)

    print(f"Worked example: [{left}] vs [{right}]")
    print(f"  A = {samples[left].text!r}")
    print(f"  B = {samples[right].text!r}")
    print(f"  A · B       = {dot_product:.6f}")
    print(f"  ||A||       = {norm_a:.6f}")
    print(f"  ||B||       = {norm_b:.6f}")
    print(
        f"  cos(A, B)   = {dot_product:.6f} / "
        f"({norm_a:.6f} * {norm_b:.6f}) = {score:.6f}"
    )

    print("\nSelected pairs:")
    print(f"{'Pair':<42s} {'Cosine':>8s}  Interpretation")
    print("-" * 90)
    for left, right, label in SIMILARITY_PAIRS:
        pair_score = cosine_similarity(embeddings[left], embeddings[right])
        print(f"{label:<42s} {pair_score:>8.4f}  {samples[left].text[:28]!r}")
        print(f"{'':<42s} {'':>8s}  {samples[right].text[:28]!r}")
    print("-" * 90)

    matrix = similarity_matrix(embeddings)
    print("\nFull 20 x 20 similarity matrix (rows/columns are sample indices):")
    np.set_printoptions(precision=2, suppress=True, linewidth=220)
    print(matrix)
    np.set_printoptions()

    # Compare within-topic and cross-topic similarity as a sanity check.
    categories = np.array([sample.category for sample in samples])
    off_diagonal = ~np.eye(len(samples), dtype=bool)
    same_topic = (categories[:, None] == categories[None, :]) & off_diagonal
    different_topic = categories[:, None] != categories[None, :]

    print("\nAverage similarity:")
    print(f"  Same topic:      {matrix[same_topic].mean():.4f}")
    print(f"  Different topic: {matrix[different_topic].mean():.4f}")
    print("Same-topic pairs should score higher on average.")

    print("\nNearest neighbors for a few sentences:")
    for index in (0, 5, 10, 15):
        scores = cosine_similarity_to_all(embeddings[index], embeddings)
        neighbors = top_k_indices(scores, 4)
        print(f"\n  Query [{index}] {samples[index].text!r}")
        for rank, neighbor in enumerate(neighbors[1:], start=1):
            if neighbor == index:
                continue
            print(
                f"    {rank}. {scores[neighbor]:.4f}  "
                f"[{neighbor:2d}] ({samples[neighbor].category}) "
                f"{samples[neighbor].text}"
            )

    return matrix


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    dimensions: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    """Project high-dimensional embeddings into 2D or 3D.

    PCA is fast, deterministic, and preserves global variance directions.
    t-SNE is slower and emphasizes local neighborhoods, which often makes
    clusters visually obvious. Both are visualization tools: distances in
    the projection are approximate and should not replace cosine similarity
    in the original high-dimensional space.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    if method == "pca":
        reducer = PCA(n_components=dimensions, random_state=random_state)
    elif method == "tsne":
        if len(embeddings) < 4:
            raise ValueError("t-SNE needs at least 4 samples")
        perplexity = float(min(30.0, max(2.0, (len(embeddings) - 1) / 3.0)))
        reducer = TSNE(
            n_components=dimensions,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    points = reducer.fit_transform(embeddings)
    return np.asarray(points), reducer


def matplotlib_available() -> bool:
    """Check for Matplotlib without importing it at module load time."""
    return importlib.util.find_spec("matplotlib") is not None


def ascii_scatter(
    points: np.ndarray,
    samples: Sequence[Sample],
    width: int = 72,
    height: int = 24,
) -> None:
    """Render a simple ASCII scatter plot as a dependency-free fallback."""
    x = points[:, 0]
    y = points[:, 1]
    x_span = float(x.max() - x.min()) or 1.0
    y_span = float(y.max() - y.min()) or 1.0

    columns = np.clip(((x - x.min()) / x_span * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(((y.max() - y) / y_span * (height - 1)).astype(int), 0, height - 1)

    categories = sorted({sample.category for sample in samples})
    markers = {category: chr(ord("A") + i) for i, category in enumerate(categories)}
    grid = [[" "] * width for _ in range(height)]

    for column, row, sample in zip(columns, rows, samples):
        marker = markers[sample.category]
        if grid[row][column] not in (" ", marker):
            marker = "*"
        grid[row][column] = marker

    print("ASCII projection (x increases right, y increases up):")
    for row in grid:
        print("  " + "".join(row).rstrip())
    print("  " + "-" * width)
    print("  Legend: " + ", ".join(f"{markers[c]}={c}" for c in categories))
    print("  * = overlapping points")


def plot_projection(
    points: np.ndarray,
    samples: Sequence[Sample],
    method: str,
    output_path: Path,
    dimensions: int,
    show: bool,
) -> Path | None:
    """Create a Matplotlib 2D/3D scatter plot, if Matplotlib is installed."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    categories = sorted({sample.category for sample in samples})
    color_map = plt.get_cmap("tab10")
    colors = {category: color_map(i % 10) for i, category in enumerate(categories)}

    fig = plt.figure(figsize=(12, 9))
    if dimensions == 3:
        axis = fig.add_subplot(111, projection="3d")
        for category in categories:
            indices = [
                i for i, sample in enumerate(samples) if sample.category == category
            ]
            axis.scatter(
                points[indices, 0],
                points[indices, 1],
                points[indices, 2],
                label=category,
                s=80,
                alpha=0.85,
                color=colors[category],
                edgecolors="white",
            )
        for index in range(len(samples)):
            axis.text(
                points[index, 0],
                points[index, 1],
                points[index, 2],
                f" {index}",
                fontsize=8,
            )
        axis.set_zlabel("Component 3")
    else:
        axis = fig.add_subplot(111)
        for category in categories:
            indices = [
                i for i, sample in enumerate(samples) if sample.category == category
            ]
            axis.scatter(
                points[indices, 0],
                points[indices, 1],
                label=category,
                s=80,
                alpha=0.85,
                color=colors[category],
                edgecolors="white",
            )
        for index in range(len(samples)):
            axis.annotate(
                str(index),
                (points[index, 0], points[index, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_title(f"Sentence embeddings projected with {method.upper()} ({dimensions}D)")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Topic", loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def demo_projection(
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    method: str,
    dimensions: int,
    random_state: int,
    output_dir: Path,
    show: bool,
    force_ascii: bool,
) -> tuple[np.ndarray, Any]:
    """Project embeddings and display the result graphically or as ASCII."""
    print_section(f"4. PROJECTING {embeddings.shape[1]}D EMBEDDINGS WITH {method.upper()}")

    points, reducer = project_embeddings(
        embeddings,
        method=method,
        dimensions=dimensions,
        random_state=random_state,
    )

    if method == "pca":
        explained = reducer.explained_variance_ratio_
        total = float(np.sum(explained))
        parts = " + ".join(f"{value:.1%}" for value in explained)
        print(f"Explained variance: {parts} = {total:.1%}")
        print("The projection keeps this fraction of the original variance.")
    else:
        print("t-SNE preserves local neighborhoods rather than global variance.")
        print("Axes have no direct physical meaning; only relative clusters matter.")

    header = f"{'Idx':>3}  {'Topic':<11s} " + " ".join(
        f"{'xyz'[i]:>8s}" for i in range(dimensions)
    )
    print("\nProjected coordinates:")
    print(header)
    print("-" * len(header))
    for index, sample in enumerate(samples):
        coordinates = " ".join(f"{points[index, i]:>8.3f}" for i in range(dimensions))
        print(f"{index:>3d}  {sample.category:<11s} {coordinates}")

    output_path = output_dir / f"embeddings_{method}_{dimensions}d.png"
    if not force_ascii and matplotlib_available():
        plot_projection(
            points,
            samples,
            method,
            output_path,
            dimensions,
            show,
        )
    else:
        if force_ascii:
            print("\nASCII mode requested.")
        else:
            print("\nMatplotlib is not installed; using the ASCII fallback.")
            print("For a graphical plot: pip install matplotlib")
        ascii_scatter(points[:, :2], samples)

    return points, reducer


def print_query_results(
    query_text: str,
    query_vector: np.ndarray,
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Print the most semantically similar sentences for one query."""
    scores = cosine_similarity_to_all(query_vector, embeddings)
    limit = min(top_k, len(samples))
    best = top_k_indices(scores, limit)

    print(f"\nQuery: {query_text!r}")
    print(f"Query vector norm: {np.linalg.norm(query_vector):.4f}")
    print(f"{'Rank':>4}  {'Cosine':>8s}  {'Topic':<11s} Sentence")
    print("-" * 96)
    for rank, index in enumerate(best, start=1):
        print(
            f"{rank:>4d}  {scores[index]:>8.4f}  "
            f"{samples[index].category:<11s} {samples[index].text}"
        )
    return scores


def demo_example_queries(
    model: Any,
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    top_k: int,
) -> None:
    """Show semantic matches for queries that avoid the exact target words."""
    print_section("5. SEMANTIC QUERY EXAMPLES")
    print("These queries share few or no exact words with the target sentences.")
    print("High scores therefore demonstrate meaning-based matching.\n")

    for query in DEMO_QUERIES:
        query_vector = generate_embeddings(model, [query])[0]
        print_query_results(query, query_vector, samples, embeddings, top_k)


def interactive_loop(
    model: Any,
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    top_k: int,
) -> None:
    """Let the user type sentences and inspect their nearest neighbors."""
    print_section("6. INTERACTIVE QUERY MODE")
    print("Type any sentence and press Enter to see its closest matches.")
    print("Commands: /add <sentence>, /help, /quit\n")

    current_samples = list(samples)
    current_embeddings = embeddings

    while True:
        try:
            raw = input("embedding> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLeaving interactive mode.")
            break

        if not raw:
            continue

        command = raw.lower()
        if command in {"/quit", "quit", "exit", "/exit"}:
            break

        if command in {"/help", "help", "?"}:
            print("  <sentence>       find the closest sentences")
            print("  /add <sentence>  add the sentence to the searchable dataset")
            print("  /quit            exit interactive mode")
            continue

        if command.startswith("/add "):
            new_text = raw[5:].strip()
            if not new_text:
                print("Usage: /add <sentence>")
                continue
            new_vector = generate_embeddings(model, [new_text])[0]
            current_embeddings = np.vstack([current_embeddings, new_vector])
            current_samples.append(Sample(new_text, "User"))
            print(f"Added sentence [{len(current_samples) - 1}]: {new_text!r}")
            continue

        query_vector = generate_embeddings(model, [raw])[0]
        print_query_results(
            raw,
            query_vector,
            current_samples,
            current_embeddings,
            top_k,
        )


def self_test() -> None:
    """Fast, model-free checks for the similarity math and projection API."""
    print_section("SELF-TEST: SIMILARITY MATH")

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

    matrix = np.vstack([a, b, c, d])
    scores = cosine_similarity_to_all(a, matrix)
    assert np.allclose(scores, [1.0, 1.0, 0.0, -1.0])
    print("  PASS  batch similarity matches pairwise results")

    full_matrix = similarity_matrix(matrix)
    assert np.allclose(np.diag(full_matrix), 1.0)
    assert np.isclose(full_matrix[0, 1], 1.0)
    assert np.isclose(full_matrix[0, 2], 0.0)
    print("  PASS  full similarity matrix is correct")

    best = top_k_indices(scores, 2)
    assert list(best) == [0, 1]
    print("  PASS  top-k ordering is correct")

    random_embeddings = np.random.default_rng(0).normal(size=(10, 8))
    points, _ = project_embeddings(random_embeddings, method="pca", dimensions=2)
    assert points.shape == (10, 2)
    print("  PASS  PCA projection returns the expected shape")

    print("\nAll self-tests passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Demonstrate text embeddings, cosine similarity, and 2D/3D projections."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Matches to show")
    parser.add_argument(
        "--method",
        choices=("pca", "tsne"),
        default="pca",
        help="Dimensionality-reduction method",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=(2, 3),
        default=2,
        help="Projection dimensions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PCA/t-SNE")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Run a one-shot query; may be supplied multiple times",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive query mode after the demo",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Never enter interactive mode",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window instead of only saving the plot",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Force the dependency-free ASCII projection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for saved plots",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run fast math checks without loading a model, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full demonstration."""
    args = parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    print("\n" + "=" * 72)
    print("  SESSION 1.2: TEXT EMBEDDINGS AND VECTOR SPACE")
    print("  From sentences to vectors, similarity, and visual clusters")
    print("=" * 72)

    samples = list(SAMPLES)
    demo_sample_corpus(samples)

    model = load_embedding_model(args.model)
    embeddings = demo_embedding_generation(model, samples, args.batch_size)

    demo_cosine_similarity(samples, embeddings)

    demo_projection(
        samples,
        embeddings,
        method=args.method,
        dimensions=args.dimensions,
        random_state=args.seed,
        output_dir=args.output_dir,
        show=args.show,
        force_ascii=args.ascii,
    )

    demo_example_queries(model, samples, embeddings, args.top_k)

    for query in args.query:
        query_vector = generate_embeddings(model, [query])[0]
        print_query_results(query, query_vector, samples, embeddings, args.top_k)

    should_interact = args.interactive or (
        not args.no_interactive and not args.query and sys.stdin.isatty()
    )
    if should_interact:
        interactive_loop(model, samples, embeddings, args.top_k)

    print("\n" + "=" * 72)
    print("  SESSION 1.2 COMPLETE")
    print("=" * 72)
    print("\nKey Takeaways:")
    print("  • An embedding maps text to a fixed-size dense vector.")
    print("  • Similar meaning produces a small angle between vectors.")
    print("  • Cosine similarity measures that angle and ignores vector length.")
    print("  • Normalized embeddings make cosine similarity equal to a dot product.")
    print("  • PCA/t-SNE projections are for inspection, not for retrieval decisions.")
    print("  • RAG uses the same embedding model for documents and queries.")
    print("  • 384 dimensions are enough for MiniLM to separate these topics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
