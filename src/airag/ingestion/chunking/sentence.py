"""
Sentence-aware chunking strategy (step 2.3.3).

Packs whole sentences into chunks up to ``ChunkConfig.chunk_size``, with tail
sentence overlap, so chunks never cut a sentence in half.

Boundary detection uses a dependency-free default splitter that handles `.`,
`!`, `?`, ellipses and CJK/Arabic end punctuation, plus common abbreviations
(``Dr.``, ``e.g.``) and initials. The plan names spaCy/NLTK for this step;
neither is installed and both require model downloads, so an NLTK/spaCy-style
``Callable[[str], list[str]]`` can be injected via ``sentence_splitter``
instead. This is a documented deviation, not a silent one.

Page boundaries, offsets and chunk metadata use the shared helpers in
``chunking/base.py``, so output is consistent with ``FixedSizeChunker``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, build_chunk, page_segments
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument

SentenceSplitter = Callable[[str], list[str]]

_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？؟…])\s+")
_INITIAL_RE = re.compile(r"\b[A-Za-z]\.$")
_ABBREVIATIONS = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "a.m", "p.m", "inc", "ltd", "co", "no", "fig",
    "approx",
)


def _is_continuation(candidate: str, following: str) -> bool:
    """Whether a punctuation boundary is actually an abbreviation/initial."""
    tail = candidate.strip().lower()
    if any(tail.endswith(abbr + ".") for abbr in _ABBREVIATIONS):
        return True
    if _INITIAL_RE.search(candidate.strip()):
        return True
    return following[:1].islower()


class SentenceAwareChunker:
    """Split documents into sentence-aligned chunks."""

    def __init__(
        self,
        config: ChunkConfig,
        sentence_splitter: SentenceSplitter | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            config: Validated size/overlap settings. For sentence chunking,
                ``chunk_size`` and ``overlap`` are measured in characters by
                default, or in tokens when ``length_unit="token"``.
            sentence_splitter: Optional callable returning sentence strings,
                e.g. ``nltk.tokenize.sent_tokenize``.
            tokenizer: Optional tiktoken-compatible tokenizer for token mode.
        """
        self.config = config
        self._sentence_splitter = sentence_splitter
        self._tokenizer = tokenizer

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        """Split ``document`` into ordered, metadata-carrying chunks."""
        chunks: list[Chunk] = []
        for page_number, page_start, page_text in page_segments(document):
            spans = self._sentence_spans(page_text)
            for start, end in self._pack(page_text, spans):
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

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Return sentence spans, using the injected splitter when provided."""
        if self._sentence_splitter is not None:
            try:
                sentences = self._sentence_splitter(text)
            except Exception as e:
                raise ChunkError(f"Sentence splitting failed: {e}") from e
            return self._spans_from_sentences(text, sentences)
        return self._default_spans(text)

    @staticmethod
    def _spans_from_sentences(
        text: str, sentences: list[str]
    ) -> list[tuple[int, int]]:
        """Map splitter output back to offsets in the source text."""
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence in sentences:
            position = text.find(sentence, cursor)
            if position < 0:
                continue
            end = position + len(sentence)
            if text[position:end].strip():
                spans.append((position, end))
            cursor = end
        return spans

    @staticmethod
    def _default_spans(text: str) -> list[tuple[int, int]]:
        """Dependency-free sentence boundary detection with offsets."""
        spans: list[tuple[int, int]] = []
        start = 0
        for match in _BOUNDARY_RE.finditer(text):
            candidate = text[start : match.start()]
            following = text[match.end() :]
            if candidate.strip() and not _is_continuation(candidate, following):
                spans.append((start, match.start()))
                start = match.end()
        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))
        return spans

    def _pack(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Group sentence spans into overlapping chunks."""
        packed: list[tuple[int, int]] = []
        i = 0
        while i < len(spans):
            start = spans[i][0]
            j = i
            while (
                j + 1 < len(spans)
                and self._length(text[start : spans[j + 1][1]]) <= self.config.chunk_size
            ):
                j += 1
            end = spans[j][1]
            packed.append((start, end))

            next_i = j + 1
            if self.config.overlap > 0:
                k = j
                while (
                    k >= i
                    and self._length(text[spans[k][0] : end]) <= self.config.overlap
                ):
                    next_i = k
                    k -= 1
            i = max(i + 1, next_i)
        return packed

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
