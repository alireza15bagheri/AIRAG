"""
Semantic chunking strategy (step 2.3.4).

Splits text at semantic boundaries: adjacent sentences are embedded and the
cosine similarity between them is compared with ``threshold``. A boundary is
placed where similarity drops below the threshold, or earlier if adding the
sentence would exceed ``ChunkConfig.chunk_size``.

The embedder is injectable (``Callable[[list[str]], Any]``) for deterministic
tests. The default lazily loads ``sentence-transformers`` with
``all-MiniLM-L6-v2`` on first use; it may download model weights if not cached.
``ChunkConfig.overlap`` is intentionally not applied here — re-including tail
sentences would blur the semantic boundary just detected.

Sentence spans, page boundaries and chunk metadata are shared with the other
strategies via ``chunking/sentence.py`` and ``chunking/base.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, build_chunk, page_segments
from src.airag.ingestion.chunking.sentence import SentenceSplitter, sentence_spans
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument

Embedder = Callable[[list[str]], Any]

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.75


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ChunkError(
            f"Embedding dimensions differ: {len(a)} vs {len(b)}"
        )
    dot = sum(
        left * right for left, right in zip(a, b, strict=True)
    )
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker:
    """Split documents at embedding-similarity boundaries."""

    def __init__(
        self,
        config: ChunkConfig,
        embedder: Embedder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        sentence_splitter: SentenceSplitter | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            config: Validated size/length-unit settings; ``overlap`` is ignored.
            embedder: Optional callable embedding a list of sentence strings.
            threshold: Similarity below which a boundary is placed (0..1).
            sentence_splitter: Optional sentence splitter callable.
            tokenizer: Optional tiktoken-compatible tokenizer for token mode.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ChunkError("threshold must be between 0 and 1")
        self.config = config
        self.threshold = threshold
        self._embedder = embedder
        self._sentence_splitter = sentence_splitter
        self._tokenizer = tokenizer
        self._model: Any = None

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        """Split ``document`` into ordered, metadata-carrying chunks."""
        chunks: list[Chunk] = []
        for page_number, page_start, page_text in page_segments(document):
            spans = sentence_spans(page_text, self._sentence_splitter)
            if not spans:
                continue
            texts = [page_text[start:end] for start, end in spans]
            embeddings = self._embeddings(texts)
            for start, end in self._group(spans, embeddings, page_text):
                chunks.append(
                    build_chunk(
                        document,
                        page_text[start:end],
                        page_number,
                        len(chunks),
                        page_start + start,
                        page_start + end,
                    )
                )
        return chunks

    def _embeddings(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed sentences and normalize the result to tuples of floats."""
        try:
            raw = (
                self._embedder(texts)
                if self._embedder is not None
                else self._default_embedder(texts)
            )
        except ChunkError:
            raise
        except Exception as e:
            raise ChunkError(f"Embedding failed: {e}") from e
        try:
            embeddings = [tuple(float(value) for value in vector) for vector in raw]
        except (TypeError, ValueError) as e:
            raise ChunkError(f"Invalid embeddings returned: {e}") from e
        if len(embeddings) != len(texts):
            raise ChunkError(
                f"Embedder returned {len(embeddings)} vectors for "
                f"{len(texts)} sentences"
            )
        return embeddings

    def _default_embedder(self, texts: list[str]) -> Any:
        """Lazily load sentence-transformers and encode ``texts``."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ChunkError(
                "sentence-transformers is required for the default embedder"
            ) from e
        try:
            if self._model is None:
                self._model = SentenceTransformer(DEFAULT_MODEL)
            return self._model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            raise ChunkError(f"Embedding failed: {e}") from e

    def _group(
        self,
        spans: list[tuple[int, int]],
        embeddings: list[tuple[float, ...]],
        page_text: str,
    ) -> list[tuple[int, int]]:
        """Group sentence spans by similarity and the configured size cap."""
        groups: list[tuple[int, int]] = []
        start = 0
        for i in range(1, len(spans)):
            similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
            prospective = page_text[spans[start][0] : spans[i][1]]
            if (
                similarity < self.threshold
                or self._length(prospective) > self.config.chunk_size
            ):
                groups.append((spans[start][0], spans[i - 1][1]))
                start = i
        groups.append((spans[start][0], spans[-1][1]))
        return groups

    def _length(self, text: str) -> int:
        """Measure text length in the configured unit."""
        if self.config.length_unit == "char":
            return len(text)
        tokenizer = self._tokenizer
        if tokenizer is None:
            try:
                import tiktoken

                tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                raise ChunkError(f"Tokenization unavailable: {e}") from e
            self._tokenizer = tokenizer
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            raise ChunkError(f"Tokenization failed: {e}") from e
