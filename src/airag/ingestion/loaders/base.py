"""
Document Loader Protocol (Abstract Base Class)

Defines the interface that all document loaders must implement.
This enables a unified ingestion pipeline regardless of file format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """
    Result of document extraction.

    Attributes:
        text: The extracted text content.
        metadata: Structured metadata extracted from the document.
        pages: Optional list of page-level text for page-aware formats.
    """
    text: str
    metadata: dict[str, Any]
    pages: list[str] | None = None


@dataclass(frozen=True, slots=True)
class LoadResult:
    """
    Result of loading a document.

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


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    All document loaders must implement the `load` method and declare
    which file extensions they support. This enables the LoaderFactory
    to route files to the appropriate loader automatically.
    """

    # Class variable: set of supported file extensions (e.g., {".pdf", ".txt"})
    # Must be overridden by subclasses. Extensions should be lowercase.
    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the loader with optional configuration.

        Args:
            config: Optional configuration dictionary. Keys are loader-specific.
        """
        self._config = config or {}

    @property
    def supported_extensions(self) -> frozenset[str]:
        """
        Return the set of file extensions this loader supports (lowercase).

        For case-insensitive matching, use `validate_file` or `load_with_validation`
        which handle case-insensitive extension matching automatically.
        """
        return self.SUPPORTED_EXTENSIONS

    @property
    def supported_extensions_case_insensitive(self) -> frozenset[str]:
        """
        Return the set of file extensions including both lowercase and uppercase variants.

        This enables case-insensitive membership testing (e.g., ".TXT" in loader.supported_extensions_case_insensitive).
        """
        extensions = set(self.SUPPORTED_EXTENSIONS)
        for ext in self.SUPPORTED_EXTENSIONS:
            extensions.add(ext.upper())
        return frozenset(extensions)

    @abstractmethod
    def load(self, file_path: Path) -> LoadResult:
        """
        Load and extract text from a document.

        Args:
            file_path: Path to the document file.

        Returns:
            LoadResult containing the extracted document or error information.
        """
        ...

    def validate_file(self, file_path: Path) -> tuple[bool, str | None]:
        """
        Validate that the file can be loaded by this loader.

        Args:
            file_path: Path to the file to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False, (
                f"Unsupported extension '{file_path.suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
        return True, None

    def load_with_validation(self, file_path: Path) -> LoadResult:
        """
        Load a file with validation.

        Args:
            file_path: Path to the document file.

        Returns:
            LoadResult containing the extracted document or error information.
        """
        is_valid, error = self.validate_file(file_path)
        if not is_valid:
            return LoadResult(
                success=False,
                document=None,
                error=error,
                source_path=file_path,
            )
        return self.load(file_path)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(extensions={sorted(self.SUPPORTED_EXTENSIONS)})"