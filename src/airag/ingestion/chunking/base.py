"""
Chunking contract for Phase 2 step 2.3.1.

Defines the shared value objects every concrete chunker will return:

- ``ChunkConfig``: validated size/overlap settings. ``length_unit`` anticipates
  the char/token-aware fixed chunker (2.3.2) without adding a tokenizer
  dependency here. Invalid settings raise ``ChunkError``.
- ``Chunk``: one chunk of text plus its position and inherited document
  metadata. It lives in this module because 2.3.1 names ``chunking/base.py``
  as its only output and no ``Chunk`` model exists in ``models.py``.
- ``Chunker``: runtime-checkable protocol. Its single method consumes an
  ``ExtractedDocument`` (not a bare string) so ``source``, ``page_count``,
  ``content_hash`` and ``ingested_at`` can be propagated into every chunk,
  as required by the Phase 2 Definition of Done.

Concrete strategies: ``fixed.py`` (2.3.2), ``sentence.py`` (2.3.3) and
``semantic.py`` (2.3.4) are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument

VALID_LENGTH_UNITS = ("char", "token")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Validated configuration shared by all chunking strategies.

    Attributes:
        chunk_size: Maximum chunk length in ``length_unit`` units; must be > 0.
        overlap: Length shared between adjacent chunks; must be >= 0 and
            strictly smaller than ``chunk_size``.
        length_unit: ``"char"`` (default) or ``"token"``.
    """

    chunk_size: int
    overlap: int = 0
    length_unit: Literal["char", "token"] = "char"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ChunkError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ChunkError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ChunkError("overlap must be less than chunk_size")
        if self.length_unit not in VALID_LENGTH_UNITS:
            raise ChunkError(
                f"Unsupported length_unit {self.length_unit!r}; "
                "expected 'char' or 'token'"
            )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk of text with positional and inherited metadata.

    Attributes:
        text: The chunk text.
        metadata: Document metadata copied by the chunker, plus chunk-specific
            keys such as ``section`` when available.
        chunk_index: Zero-based position of this chunk in its document.
        start_char: Start offset in the source text.
        end_char: End offset in the source text (exclusive).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


def page_segments(document: ExtractedDocument) -> list[tuple[int, int, str]]:
    """Return ``(page_number, start_offset, page_text)`` for each page.

    Falls back to the full text when ``document.pages`` is empty. Offsets index
    into ``document.text`` so chunk spans stay valid for every format.
    """
    pages = document.pages or [document.text]
    segments: list[tuple[int, int, str]] = []
    cursor = 0
    for page_number, page_text in enumerate(pages, start=1):
        position = document.text.find(page_text, cursor)
        if position < 0:
            position = cursor
        segments.append((page_number, position, page_text))
        cursor = position + len(page_text)
    return segments


def build_chunk(
    document: ExtractedDocument,
    text: str,
    page_number: int,
    chunk_index: int,
    start_char: int,
    end_char: int,
) -> Chunk:
    """Build a chunk with inherited and chunk-specific metadata."""
    metadata = dict(document.metadata)
    metadata.update(
        {
            "page": page_number,
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "hash": document.content_hash,
            "ingested_at": document.ingested_at,
        }
    )
    return Chunk(
        text=text,
        metadata=metadata,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
    )


@runtime_checkable
class Chunker(Protocol):
    """Contract implemented by every concrete chunking strategy."""

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        """Split ``document`` into ordered, metadata-carrying chunks."""
        ...
