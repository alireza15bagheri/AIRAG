"""
Canonical ingestion data models (step 2.2.1, 2.4.4, 2.5.1).

``ExtractedDocument`` is the single extraction container carried inside
``LoadResult`` by every ``DocumentLoader.load()`` implementation. No parallel
shapes are allowed.

Contract:
    - ``text``: full extracted document text.
    - ``pages``: per-page text, never ``None``. Page-aware formats (PDF) emit
      one entry per page; non-paginated formats (DOCX/Markdown/Text) use the
      single-element convention ``pages=[full_text]`` observed in the loaders.
      ``[]`` unambiguously means "nothing extracted" (e.g. a scanned PDF with
      no text layer).
    - ``metadata``: document-level metadata, populated cooperatively by the
      per-format extractors in steps 2.2.2-2.2.5. Keys are format-specific;
      ``source``, ``page_count`` and ``format`` are shared across all loaders.
    - ``content_hash``: SHA-256 hash of the extracted text for idempotent writes (step 2.4.4).
    - ``ingested_at``: ISO-8601 timestamp when the document was ingested (step 2.5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """
    Result of document extraction (frozen, slotted value object).

    Attributes:
        text: The extracted text content.
        metadata: Structured metadata extracted from the document.
        pages: Per-page text; defaults to a fresh empty list, never None.
        content_hash: SHA-256 hash of the text content for deduplication.
        ingested_at: ISO-8601 timestamp when the document was ingested.
    """

    text: str
    metadata: dict[str, Any]
    pages: list[str] = field(default_factory=list)
    content_hash: str = field(default="")
    ingested_at: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", self._compute_hash(self.text))
        if not self.ingested_at:
            object.__setattr__(self, "ingested_at", datetime.now(timezone.utc).isoformat())

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadResult:
    """
    Result of loading a single document.

    Attributes:
        success: Whether the load succeeded.
        document: The extracted document (None if failed).
        error: Error message if failed.
        source_path: Path to the source file.
    """
    success: bool
    document: ExtractedDocument | None
    error: str | None
    source_path: Path


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """
    Result of ingesting a directory or collection of documents.

    Attributes:
        loaded: Number of successfully loaded documents.
        failed: Number of failed documents.
        skipped: Number of skipped (duplicate) documents.
        results: List of individual LoadResult objects.
        started_at: ISO-8601 timestamp when ingestion started.
        completed_at: ISO-8601 timestamp when ingestion completed.
    """

    loaded: int
    failed: int
    skipped: int
    results: list[LoadResult]
    started_at: str
    completed_at: str
