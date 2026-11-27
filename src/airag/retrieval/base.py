"""
Core retrieval contracts (step 6.1).

``RetrievalResult`` is the single result shape returned by every retriever. It
carries the chunk text, similarity score and the source/page/section metadata
needed for grounded answers. ``Retriever`` is the protocol implemented by
concrete retrievers in later steps.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.exceptions import ValidationError


def validate_top_k(top_k: int) -> int:
    """Validate a positive ``top_k`` value for retriever implementations."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k


def validate_score_threshold(score_threshold: float | None) -> float | None:
    """Validate an optional finite similarity threshold."""
    if score_threshold is None:
        return None
    if not isinstance(score_threshold, (int, float)) or isinstance(
        score_threshold, bool
    ):
        raise ValidationError(
            f"score_threshold must be a number or None, got {score_threshold!r}"
        )
    if not math.isfinite(score_threshold):
        raise ValidationError("score_threshold must be finite")
    return float(score_threshold)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One retrieved chunk with its provenance metadata.

    Attributes:
        chunk_id: Stable ID matching the vector store entry.
        text: Retrieved chunk text.
        score: Similarity score; higher means more relevant.
        source: Source document path.
        format: Document format (``pdf``, ``md``, ``txt``, ``docx``).
        page: One-based page number when the source was paginated.
        section: Section/heading when the chunker recorded one.
        metadata: Full stored metadata, including any extra keys.
    """

    chunk_id: str
    text: str
    score: float
    source: str = ""
    format: str = ""
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValidationError("chunk_id must not be empty")
        if not self.text:
            raise ValidationError("RetrievalResult text must not be empty")
        if not math.isfinite(self.score):
            raise ValidationError("score must be a finite number")

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, chunk_id: str, score: float
    ) -> RetrievalResult:
        """Build a result from a chunk and its stored ID/score.

        Source, format, page and section are read from the chunk metadata;
        the original metadata is deep-copied for extra filters.
        """
        metadata = copy.deepcopy(chunk.metadata)
        raw_page = metadata.get("page")
        try:
            page = int(raw_page) if raw_page not in (None, "") else None
        except (TypeError, ValueError):
            page = None
        raw_section = metadata.get("section")
        section = None if raw_section is None else str(raw_section)
        return cls(
            chunk_id=chunk_id,
            text=chunk.text,
            score=float(score),
            source=str(metadata.get("source") or ""),
            format=str(metadata.get("format") or ""),
            page=page,
            section=section,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the future API layer."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "format": self.format,
            "page": self.page,
            "section": self.section,
            "metadata": copy.deepcopy(self.metadata),
        }


@runtime_checkable
class Retriever(Protocol):
    """Contract for retrieving stored chunks relevant to a query."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        score_threshold: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return up to ``top_k`` results ordered by descending relevance.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of results.
            score_threshold: Optional minimum similarity score.
            where: Optional ChromaDB metadata filter (e.g. ``{"source": ...}``).

        Returns:
            A list of results; an empty list when nothing matches.
        """
        ...


__all__ = [
    "RetrievalResult",
    "Retriever",
    "validate_score_threshold",
    "validate_top_k",
]
