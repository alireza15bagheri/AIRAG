"""
Chunk → embedding → document-ID pipeline (step 2.4.3).

Step 2.4.3 provides ``EmbeddingPipeline``: it groups chunks by stable document
ID, embeds each document's chunks in one call through ``EmbeddingService``, and
delegates idempotent storage to ``VectorStoreWriter``.
Step 2.5.1 adds ``IngestionPipeline`` alongside it: a full orchestrator that
loads files/directories, chunks, embeds and stores, with per-file failure
isolation. Directory walking remains a delegate to ``DocumentIngestionService``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(slots=True)
class FileResult:
    """Outcome for one source file in an ingestion run."""

    source: str
    success: bool
    chunks: int = 0
    written: int = 0
    updated: int = 0
    skipped: int = 0
    error: str | None = None


@dataclass(slots=True)
class PipelineRunResult:
    """Aggregate outcome of an end-to-end ingestion run."""

    files: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks: int = 0
    written: int = 0
    updated: int = 0
    skipped: int = 0
    file_results: list[FileResult] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrate load → extract → chunk → embed → store for files or directories."""

    def __init__(
        self,
        chunker: Any,
        embedding_pipeline: Any,
        document_service: Any | None = None,
    ) -> None:
        """Compose the orchestrator from already-built stages.

        Args:
            chunker: Object exposing ``chunk(document) -> list[Chunk]``.
            embedding_pipeline: ``EmbeddingPipeline`` (or compatible object).
            document_service: ``DocumentIngestionService``; a default recursive
                service is created when omitted.
        """
        self.chunker = chunker
        self.embedding_pipeline = embedding_pipeline
        self.document_service = document_service or self._default_service()

    @staticmethod
    def _default_service() -> Any:
        from src.airag.ingestion.service import DocumentIngestionService

        return DocumentIngestionService()

    def ingest(self, path: str | Path) -> PipelineRunResult:
        """Ingest one file or a (recursively discovered) directory tree."""
        source = Path(path)
        ingestion = (
            self.document_service.ingest_directory(source)
            if source.is_dir()
            else self.document_service.ingest_files([source])
        )

        result = PipelineRunResult(files=len(ingestion.results))
        for load in ingestion.results:
            source_name = str(load.source_path)
            if not load.success or load.document is None:
                result.failed += 1
                result.file_results.append(
                    FileResult(
                        source=source_name,
                        success=False,
                        error=load.error or "No document extracted",
                    )
                )
                continue

            try:
                chunks = self.chunker.chunk(load.document)
                written = self.embedding_pipeline.run(chunks)
            except Exception as e:
                # Per-file failure isolation; dead letter queue is step 2.6.4.
                result.failed += 1
                result.file_results.append(
                    FileResult(
                        source=source_name,
                        success=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                continue

            result.succeeded += 1
            result.chunks += written.processed
            result.written += written.written
            result.updated += written.updated
            result.skipped += written.skipped
            result.file_results.append(
                FileResult(
                    source=source_name,
                    success=True,
                    chunks=written.processed,
                    written=written.written,
                    updated=written.updated,
                    skipped=written.skipped,
                )
            )
        return result
