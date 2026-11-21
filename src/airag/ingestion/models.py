"""
Canonical ingestion data models (step 2.2.1).

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

Open items (deliberately not populated here):
    - content ``hash``: assigned by the idempotent-write step 2.4.4.
    - ingestion timestamps: no extraction step owns them yet (2.4/2.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """
    Result of document extraction (frozen, slotted value object).

    Attributes:
        text: The extracted text content.
        metadata: Structured metadata extracted from the document.
        pages: Per-page text; defaults to a fresh empty list, never None.
    """

    text: str
    metadata: dict[str, Any]
    pages: list[str] = field(default_factory=list)
