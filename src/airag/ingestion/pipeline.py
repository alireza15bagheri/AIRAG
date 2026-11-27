"""
Chunk → embedding → document-ID pipeline (step 2.4.3).

This is the narrow 2.4.3 stage: it groups chunks by stable document ID, embeds
each document's chunks in one call through ``EmbeddingService``, and delegates
idempotent storage to ``VectorStoreWriter``. It does not load documents, walk
directories or own chunking — that broader orchestration belongs to step 2.5.1
and will call this pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.airag.ingestion.chunking.base import Chunk


@dataclass(slots=True)
class PipelineResult:
    """Aggregate outcome of an embedding pipeline run.

    Attributes:
        processed: Number of chunks seen.
        written: New chunk IDs upserted.
        updated: Existing chunk IDs upserted after a text change.
        skipped: Chunk IDs whose content was unchanged.
        document_ids: Stable document IDs, in first-seen order.
        chunk_ids: Stable IDs assigned to every processed chunk.
    """

    processed: int = 0
    written: int = 0
    updated: int = 0
    skipped: int = 0
    document_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)


class EmbeddingPipeline:
    """Embed chunks and hand them to the vector store with stable IDs."""

    def __init__(self, embedder: Any, writer: Any) -> None:
        """Initialize with an ``EmbeddingService`` and a ``VectorStoreWriter``.

        Args:
            embedder: Object exposing ``embed(texts) -> list[list[float]]``.
            writer: Object exposing ``write(chunks, embeddings)``, ``document_id``
                and ``chunk_id``.
        """
        self.embedder = embedder
        self.writer = writer

    def run(self, chunks: Sequence[Chunk]) -> PipelineResult:
        """Embed and store chunks, grouped by their document ID."""
        chunk_list = list(chunks)
        result = PipelineResult()
        if not chunk_list:
            return result

        groups: dict[str, list[Chunk]] = {}
        for chunk in chunk_list:
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"Expected Chunk instances, got {type(chunk).__name__}"
                )
            groups.setdefault(self.writer.document_id(chunk), []).append(chunk)

        for document_id, group in groups.items():
            vectors = self.embedder.embed([chunk.text for chunk in group])
            write_result = self.writer.write(group, vectors)

            result.processed += len(group)
            result.written += write_result.written
            result.updated += write_result.updated
            result.skipped += write_result.skipped
            result.document_ids.append(document_id)
            result.chunk_ids.extend(self.writer.chunk_id(chunk) for chunk in group)
        return result
