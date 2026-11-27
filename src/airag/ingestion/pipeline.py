"""
Chunk → embedding → document-ID pipeline (step 2.4.3).

Step 2.4.3 provides ``EmbeddingPipeline``: it groups chunks by stable document
ID, embeds each document's chunks in one call through ``EmbeddingService``, and
delegates idempotent storage to ``VectorStoreWriter``.
Step 2.5.1 adds ``IngestionPipeline`` alongside it: a full orchestrator that
loads files/directories, chunks, embeds and stores, with per-file failure
isolation. Directory walking remains a delegate to ``DocumentIngestionService``.
Step 2.5.4 adds opt-in JSON checkpoints (resume) and progress reporting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.dlq import DeadLetterQueue, DLQEntry
from src.airag.ingestion.metrics import (
    CHUNK_SECONDS,
    CHUNKS_CREATED,
    CHUNKS_SKIPPED,
    CHUNKS_UPDATED,
    CHUNKS_WRITTEN,
    EMBED_SECONDS,
    FILES_FAILED,
    FILES_PROCESSED,
    FILES_SKIPPED,
    FILES_SUCCEEDED,
    LOAD_SECONDS,
    STORE_SECONDS,
    MetricsRegistry,
)


def _error_type(message: str) -> str:
    """Derive a coarse error type from a formatted pipeline error message."""
    candidate = message.split(":", 1)[0].strip()
    return candidate if candidate and len(candidate) < 80 else "Error"


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

    def __init__(
        self, embedder: Any, writer: Any, metrics: MetricsRegistry | None = None
    ) -> None:
        """Initialize with an ``EmbeddingService`` and a ``VectorStoreWriter``.

        Args:
            embedder: Object exposing ``embed(texts) -> list[list[float]]``.
            writer: Object exposing ``write(chunks, embeddings)``, ``document_id``
                and ``chunk_id``.
            metrics: Optional registry; a fresh one is created when omitted.
        """
        self.embedder = embedder
        self.writer = writer
        self.metrics = metrics if metrics is not None else MetricsRegistry()

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
            with self.metrics.timer(EMBED_SECONDS):
                vectors = self.embedder.embed([chunk.text for chunk in group])
            with self.metrics.timer(STORE_SECONDS):
                write_result = self.writer.write(group, vectors)

            self.metrics.increment(CHUNKS_CREATED, len(group))
            self.metrics.increment(CHUNKS_WRITTEN, write_result.written)
            self.metrics.increment(CHUNKS_UPDATED, write_result.updated)
            self.metrics.increment(CHUNKS_SKIPPED, write_result.skipped)

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
    resumed: bool = False
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
    resumed: int = 0
    file_results: list[FileResult] = field(default_factory=list)


class CheckpointManager:
    """JSON checkpoint of document hashes that completed the full pipeline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._completed: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        """Read checkpoint state; a missing/corrupt file starts empty."""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            completed = data.get("completed") if isinstance(data, dict) else None
            if isinstance(completed, dict):
                self._completed = {
                    str(key): dict(value)
                    for key, value in completed.items()
                    if isinstance(value, dict)
                }
        except (OSError, ValueError, json.JSONDecodeError):
            self._completed = {}

    def save(self) -> None:
        """Persist state atomically via a temporary file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(
            json.dumps({"version": 1, "completed": self._completed}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def is_completed(self, document_hash: str) -> bool:
        """Whether a document hash was fully ingested before."""
        return document_hash in self._completed

    def mark_completed(self, document_hash: str, source: str) -> None:
        """Record a document hash as fully ingested and persist it."""
        self._completed[document_hash] = {
            "source": source,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def clear(self) -> None:
        """Forget all completed documents and remove the state file."""
        self._completed = {}
        if self.path.exists():
            self.path.unlink()


class NullProgress:
    """Progress reporter that emits nothing (default)."""

    def start(self, total: int) -> None:
        pass

    def advance(self, label: str = "") -> None:
        pass

    def finish(self) -> None:
        pass


class TqdmProgress:
    """Progress reporter backed by tqdm (writes to stderr)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._bar: Any = None

    def start(self, total: int) -> None:
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc="Ingesting", **self.kwargs)

    def advance(self, label: str = "") -> None:
        if self._bar is not None:
            self._bar.update(1)
            if label:
                self._bar.set_postfix_str(label[-60:])

    def finish(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class IngestionPipeline:
    """Orchestrate load → extract → chunk → embed → store for files or directories."""

    def __init__(
        self,
        chunker: Any,
        embedding_pipeline: Any,
        document_service: Any | None = None,
        checkpoint: CheckpointManager | None = None,
        progress: Any | None = None,
        metrics: MetricsRegistry | None = None,
        dlq: DeadLetterQueue | None = None,
    ) -> None:
        """Compose the orchestrator from already-built stages.

        Args:
            chunker: Object exposing ``chunk(document) -> list[Chunk]``.
            embedding_pipeline: ``EmbeddingPipeline`` (or compatible object).
            document_service: ``DocumentIngestionService``; a default recursive
                service is created when omitted.
            checkpoint: Optional ``CheckpointManager`` enabling resume.
            progress: Optional reporter with ``start``/``advance``/``finish``.
            metrics: Optional registry; defaults to the embedding pipeline's
                registry so all stages share one view.
            dlq: Optional dead letter queue for failed files.
        """
        self.chunker = chunker
        self.embedding_pipeline = embedding_pipeline
        self.document_service = document_service or self._default_service()
        self.checkpoint = checkpoint
        self.progress = progress if progress is not None else NullProgress()
        self.metrics = (
            metrics
            if metrics is not None
            else getattr(embedding_pipeline, "metrics", None) or MetricsRegistry()
        )
        self.dlq = dlq

    def _record_failure(
        self, source: str, error: str, content_hash: str = "", chunks: int = 0
    ) -> None:
        """Append a failed file to the DLQ when one is configured."""
        if self.dlq is not None:
            self.dlq.append(
                DLQEntry(
                    source=source,
                    error=error,
                    error_type=_error_type(error),
                    content_hash=content_hash,
                    chunks=chunks,
                )
            )

    @staticmethod
    def _default_service() -> Any:
        from src.airag.ingestion.service import DocumentIngestionService

        return DocumentIngestionService()

    def ingest(self, path: str | Path) -> PipelineRunResult:
        """Ingest one file or a (recursively discovered) directory tree."""
        source = Path(path)
        try:
            with self.metrics.timer(LOAD_SECONDS):
                ingestion = (
                    self.document_service.ingest_directory(source)
                    if source.is_dir()
                    else self.document_service.ingest_files([source])
                )

            result = PipelineRunResult(files=len(ingestion.results))
            self.metrics.increment(FILES_PROCESSED, len(ingestion.results))
            self.progress.start(len(ingestion.results))
            for load in ingestion.results:
                source_name = str(load.source_path)
                if not load.success or load.document is None:
                    result.failed += 1
                    self.metrics.increment(FILES_FAILED)
                    failure = load.error or "No document extracted"
                    self._record_failure(source_name, failure)
                    result.file_results.append(
                        FileResult(
                            source=source_name,
                            success=False,
                            error=failure,
                        )
                    )
                    self.progress.advance(source_name)
                    continue

                document_hash = load.document.content_hash
                if self.checkpoint is not None and self.checkpoint.is_completed(
                    document_hash
                ):
                    result.resumed += 1
                    self.metrics.increment(FILES_SKIPPED)
                    result.file_results.append(
                        FileResult(source=source_name, success=True, resumed=True)
                    )
                    self.progress.advance(source_name)
                    continue

                try:
                    with self.metrics.timer(CHUNK_SECONDS):
                        chunks = self.chunker.chunk(load.document)
                    written = self.embedding_pipeline.run(chunks)
                except Exception as e:
                    # Per-file failure isolation; failures are queued for retry/review.
                    result.failed += 1
                    self.metrics.increment(FILES_FAILED)
                    failure = f"{type(e).__name__}: {e}"
                    self._record_failure(source_name, failure, document_hash)
                    result.file_results.append(
                        FileResult(
                            source=source_name,
                            success=False,
                            error=failure,
                        )
                    )
                    self.progress.advance(source_name)
                    continue

                if self.checkpoint is not None:
                    self.checkpoint.mark_completed(document_hash, source_name)

                result.succeeded += 1
                self.metrics.increment(FILES_SUCCEEDED)
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
                self.progress.advance(source_name)
            return result
        finally:
            self.progress.finish()

    def retry_dlq(self) -> PipelineRunResult:
        """Re-ingest pending DLQ sources, clearing those that now succeed.

        Failed retries stay queued; DLQ deduplication prevents duplicate
        entries across attempts.
        """
        if self.dlq is None:
            return PipelineRunResult()

        sources = self.dlq.pending_sources()
        combined = PipelineRunResult(files=len(sources))
        for source in sources:
            run = self.ingest(source)
            combined.succeeded += run.succeeded
            combined.failed += run.failed
            combined.chunks += run.chunks
            combined.written += run.written
            combined.updated += run.updated
            combined.skipped += run.skipped
            combined.resumed += run.resumed
            combined.file_results.extend(run.file_results)
            if run.failed == 0 and (run.succeeded + run.resumed) > 0:
                self.dlq.remove(source)
        return combined
