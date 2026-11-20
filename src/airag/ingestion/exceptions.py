"""
Custom exceptions for the ingestion pipeline.

Step 2.6.1: Custom exceptions for the ingestion pipeline.
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
    """Raised when no loader is registered for a file's extension.

    TODO(2.6.1): Final hierarchy placement once the full exception tree lands.
    """

    pass


class ChunkError(IngestionError):
    """Raised when document chunking fails."""
    pass


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""
    pass


class ValidationError(IngestionError):
    """Raised when document validation fails."""
    pass
