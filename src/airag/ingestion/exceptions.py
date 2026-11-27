"""
Custom exceptions for the ingestion pipeline (step 2.6.1, finalized).

Hierarchy::

    Exception
      └── IngestionError
          ├── LoadError (source_path, cause)
          │     └── UnsupportedFormatError
          ├── ChunkError
          ├── EmbeddingError
          ├── ValidationError
          ├── VectorStoreError
          └── PipelineError

Design notes:
    - ``factory.py`` registration errors stay ``TypeError``/``ValueError``
      because they are programming errors, not pipeline failures.
    - Chunkers keep raising ``ChunkError`` for tokenization/embedding failures
      inside their own stage; the standalone ``EmbeddingService`` raises
      ``EmbeddingError``. Both remain subclasses of ``IngestionError``.
    - HTTP status mapping is deferred to the Phase 8 API boundary, where the
      semantics of each error can be decided with the endpoints.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""

    pass


class LoadError(IngestionError):
    """
    Raised when a document fails to load.

    Attributes:
        source_path: Path to the file that failed to load.
        cause: The original exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        source_path: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        if source_path:
            message = f"{message} (source: {source_path})"
        super().__init__(message)
        self.source_path = source_path
        self.cause = cause


class UnsupportedFormatError(LoadError):
    """Raised when no loader is registered for a file's extension."""

    pass


class ChunkError(IngestionError):
    """Raised when document chunking or tokenization fails."""

    pass


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""

    pass


class ValidationError(IngestionError):
    """Raised when configuration or document validation fails."""

    pass


class VectorStoreError(IngestionError):
    """Raised when ChromaDB collection access, lookup or upsert fails."""

    pass


class PipelineError(IngestionError):
    """Raised when pipeline orchestration itself fails."""

    pass


__all__ = [
    "IngestionError",
    "LoadError",
    "UnsupportedFormatError",
    "ChunkError",
    "EmbeddingError",
    "ValidationError",
    "VectorStoreError",
    "PipelineError",
]
