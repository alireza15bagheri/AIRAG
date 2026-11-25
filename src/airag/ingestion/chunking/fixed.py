"""
Fixed-size chunking strategy (step 2.3.2).

Sliding-window chunker over an ``ExtractedDocument``. Supports two length
units, selected by ``ChunkConfig.length_unit``:

- ``"char"``: windows of characters (default, dependency-free).
- ``"token"``: windows of BPE token IDs via tiktoken; character offsets are
  reconstructed from each token's decoded byte length, so ``Chunk.start_char``
  and ``Chunk.end_char`` always index into the source text.

Page boundaries from ``ExtractedDocument.pages`` are respected: each page is
chunked independently and every resulting ``Chunk`` carries a 1-based ``page``
metadata key. Document metadata (``source``, ``content_hash``, ``ingested_at``,
etc.) is copied into every chunk, alongside ``chunk_index`` and offsets.
"""

from __future__ import annotations

from typing import Any

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, build_chunk, page_segments
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument

DEFAULT_ENCODING = "cl100k_base"


class FixedSizeChunker:
    """Split documents into fixed-size overlapping windows."""

    def __init__(self, config: ChunkConfig, tokenizer: Any | None = None) -> None:
        """Initialize the chunker.

        Args:
            config: Validated chunk size, overlap and length unit.
            tokenizer: Optional tiktoken-compatible tokenizer (must provide
                ``encode(text)`` and ``decode_tokens_bytes(ids)``). Injecting one
                avoids loading the default encoding in tests.
        """
        self.config = config
        self._tokenizer = tokenizer

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        """Split ``document`` into ordered, metadata-carrying chunks."""
        chunks: list[Chunk] = []
        for page_number, page_start, page_text in page_segments(document):
            if self.config.length_unit == "char":
                windows = self._char_windows(page_text)
            else:
                windows = self._token_windows(page_text)

            for local_start, local_end, text in windows:
                chunks.append(
                    build_chunk(
                        document,
                        text,
                        page_number,
                        len(chunks),
                        page_start + local_start,
                        page_start + local_end,
                    )
                )
        return chunks

    def _char_windows(self, text: str) -> list[tuple[int, int, str]]:
        """Build overlapping character windows with absolute local offsets."""
        windows: list[tuple[int, int, str]] = []
        size, overlap = self.config.chunk_size, self.config.overlap
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            piece = text[start:end]
            if piece.strip():
                windows.append((start, end, piece))
            if end >= len(text):
                break
            start = end - overlap
        return windows

    def _token_windows(self, text: str) -> list[tuple[int, int, str]]:
        """Build overlapping token windows with character offsets preserved."""
        tokenizer = self._get_tokenizer()
        try:
            token_ids = tokenizer.encode(text)
        except Exception as e:
            raise ChunkError(f"Tokenization failed: {e}") from e
        if not token_ids:
            return []

        try:
            token_bytes = tokenizer.decode_tokens_bytes(token_ids)
        except Exception as e:
            raise ChunkError(f"Token offset mapping failed: {e}") from e

        char_starts = [0]
        total = 0
        for raw in token_bytes:
            total += len(raw.decode("utf-8", errors="replace"))
            char_starts.append(total)

        windows: list[tuple[int, int, str]] = []
        size, overlap = self.config.chunk_size, self.config.overlap
        start = 0
        while start < len(token_ids):
            end = min(start + size, len(token_ids))
            char_start, char_end = char_starts[start], char_starts[end]
            piece = text[char_start:char_end]
            if piece.strip():
                windows.append((char_start, char_end, piece))
            if end >= len(token_ids):
                break
            start = end - overlap
        return windows

    def _get_tokenizer(self) -> Any:
        """Return the injected tokenizer or the cached default tiktoken encoding."""
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import tiktoken
        except ImportError as e:
            raise ChunkError("tiktoken is required for length_unit='token'") from e
        try:
            self._tokenizer = tiktoken.get_encoding(DEFAULT_ENCODING)
        except Exception as e:
            raise ChunkError(
                f"Could not load tokenizer {DEFAULT_ENCODING!r}: {e}"
            ) from e
        return self._tokenizer
