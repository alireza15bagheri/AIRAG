"""
Document ingestion service with directory traversal and pipeline orchestration.

Step 2.5.1: Recursive directory traversal to discover supported documents.
Step 2.5.x: Pipeline orchestration - load, hash, deduplicate, and collect results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.airag.ingestion.exceptions import (
    LoadError,
    PipelineError,
    UnsupportedFormatError,
)
from src.airag.ingestion.factory import get_loader, supported_extensions
from src.airag.ingestion.models import IngestionResult, LoadResult

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Service for ingesting documents from directories or file paths.

    Handles recursive directory traversal, loader resolution, content hashing
    for idempotent writes, and result aggregation.
    """

    def __init__(
        self,
        extensions: tuple[str, ...] | None = None,
        recursive: bool = True,
        skip_errors: bool = True,
        deduplicate: bool = True,
        loader_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ingestion service.

        Args:
            extensions: File extensions to process (default: all supported).
            recursive: Whether to traverse subdirectories.
            skip_errors: Whether to skip files that fail to load (vs raising).
            deduplicate: Whether to skip documents with duplicate content hashes.
            loader_options: Extra keyword arguments forwarded to loader constructors.
        """
        self.extensions = extensions or tuple(supported_extensions())
        self.recursive = recursive
        self.skip_errors = skip_errors
        self.deduplicate = deduplicate
        self.loader_options = loader_options or {}
        self._seen_hashes: set[str] = set()

    def ingest_directory(self, directory: str | Path) -> IngestionResult:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to the directory to ingest.

        Returns:
            IngestionResult with aggregated statistics and individual results.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise PipelineError(f"Path is not a directory: {dir_path}")

        files = self._discover_files(dir_path)
        return self._ingest_files(files)

    def ingest_files(self, paths: list[str | Path]) -> IngestionResult:
        """
        Ingest a specific list of file paths.

        Args:
            paths: List of file paths to ingest.

        Returns:
            IngestionResult with aggregated statistics and individual results.
        """
        file_paths = [Path(p) for p in paths]
        return self._ingest_files(file_paths)

    def _discover_files(self, directory: Path) -> list[Path]:
        """
        Recursively discover all supported files in a directory.

        Step 2.5.1: Directory traversal implementation.
        """
        files: list[Path] = []
        pattern = "**/*" if self.recursive else "*"

        for ext in self.extensions:
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() == ext:
                    files.append(file_path)

        logger.info(
            "Discovered %d supported files in %s",
            len(files),
            directory,
            extra={"directory": str(directory), "file_count": len(files)},
        )
        return files

    def _ingest_files(self, files: list[Path]) -> IngestionResult:
        """Process a list of files through the ingestion pipeline."""
        started_at = datetime.now(timezone.utc).isoformat()
        results: list[LoadResult] = []
        loaded = 0
        failed = 0
        skipped = 0

        for file_path in files:
            result = self._load_single(file_path)
            results.append(result)

            if result.success:
                if result.document and self.deduplicate:
                    if result.document.content_hash in self._seen_hashes:
                        skipped += 1
                        logger.info(
                            "Skipped duplicate document: %s",
                            file_path,
                            extra={"source_path": str(file_path), "hash": result.document.content_hash},
                        )
                        continue
                    self._seen_hashes.add(result.document.content_hash)
                loaded += 1
            else:
                failed += 1

        completed_at = datetime.now(timezone.utc).isoformat()

        return IngestionResult(
            loaded=loaded,
            failed=failed,
            skipped=skipped,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _load_single(self, file_path: Path) -> LoadResult:
        """Load a single file using the appropriate loader."""
        try:
            loader = get_loader(file_path, **self.loader_options)
            result = loader.load(file_path)

            if not result.success or result.document is None:
                error_msg = result.error or "Loader returned no document"
                logger.warning(
                    "No document extracted from %s: %s",
                    file_path,
                    error_msg,
                    extra={"source_path": str(file_path)},
                )
                if not self.skip_errors:
                    raise LoadError(error_msg, source_path=str(file_path))
                return LoadResult(
                    document=None,
                    success=False,
                    error=error_msg,
                    source_path=file_path,
                )

            doc = result.document
            logger.debug(
                "Loaded document: %s (hash: %s)",
                file_path,
                doc.content_hash[:16],
                extra={"source_path": str(file_path), "hash": doc.content_hash},
            )
            return result

        except UnsupportedFormatError as e:
            error_msg = f"Unsupported format: {e}"
            logger.warning(
                "Unsupported format for %s: %s",
                file_path,
                e,
                extra={"source_path": str(file_path)},
            )
            if not self.skip_errors:
                raise
            return LoadResult(
                document=None,
                success=False,
                error=error_msg,
                source_path=file_path,
            )

        except LoadError as e:
            error_msg = f"Load failed: {e}"
            logger.error(
                "Failed to load %s: %s",
                file_path,
                e,
                extra={"source_path": str(file_path)},
            )
            if not self.skip_errors:
                raise
            return LoadResult(
                document=None,
                success=False,
                error=error_msg,
                source_path=file_path,
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.exception(
                "Unexpected error loading %s",
                file_path,
                extra={"source_path": str(file_path)},
            )
            if not self.skip_errors:
                raise PipelineError(f"Failed to ingest {file_path}: {e}") from e
            return LoadResult(
                document=None,
                success=False,
                error=error_msg,
                source_path=file_path,
            )

    def reset_deduplication(self) -> None:
        """Clear the deduplication hash set for a new ingestion run."""
        self._seen_hashes.clear()


def create_ingestion_service(
    extensions: tuple[str, ...] | None = None,
    recursive: bool = True,
    skip_errors: bool = True,
    deduplicate: bool = True,
    loader_options: dict[str, Any] | None = None,
) -> DocumentIngestionService:
    """
    Factory function to create a configured DocumentIngestionService.

    Args:
        extensions: File extensions to process (default: all supported).
        recursive: Whether to traverse subdirectories.
        skip_errors: Whether to skip files that fail to load.
        deduplicate: Whether to skip duplicate content.
        loader_options: Extra keyword arguments forwarded to loader constructors.

    Returns:
        Configured DocumentIngestionService instance.
    """
    return DocumentIngestionService(
        extensions=extensions,
        recursive=recursive,
        skip_errors=skip_errors,
        deduplicate=deduplicate,
        loader_options=loader_options,
    )
