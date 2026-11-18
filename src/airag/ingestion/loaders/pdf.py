"""
PDF Document Loader using PyMuPDF (fitz).

Step 2.1.2: PDFLoader implementation using PyMuPDF (fitz).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.airag.ingestion.exceptions import LoadError
from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """
    PDF document loader using PyMuPDF (fitz).

    Extracts text page-by-page, preserves page structure, and extracts
    document metadata including title, author, and creation date.

    Supported extensions: .pdf (case-insensitive)
    """

    SUPPORTED_EXTENSIONS = frozenset({".pdf"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the PDF loader.

        Args:
            config: Optional configuration dictionary.
                Supported keys:
                - extract_images: bool, whether to extract images (default: False)
                - password: str, password for encrypted PDFs (default: "")
        """
        super().__init__(config)
        self._password = config.get("password", "") if config else ""
        self._extract_images = config.get("extract_images", False) if config else False

    def load(self, file_path: Path) -> LoadResult:
        """
        Load and extract text from a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            LoadResult containing the extracted document or error information.
        """
        import fitz  # PyMuPDF

        file_path = Path(file_path)

        # Validate file extension
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. Expected .pdf",
                source_path=str(file_path),
            )

        doc = None
        try:
            doc = fitz.open(file_path)

            # Check if encrypted and requires password
            if doc.is_encrypted and not doc.authenticate(self._password):
                raise LoadError(
                    "PDF is encrypted and requires a password",
                    source_path=str(file_path),
                )

            page_count = doc.page_count
            if page_count == 0:
                raise LoadError(
                    "PDF has zero pages",
                    source_path=str(file_path),
                )

            # Extract text page by page
            pages: list[str] = []
            total_chars = 0

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text("text")
                pages.append(text)
                total_chars += len(text)

            # Combine all pages with double newline separator
            full_text = "\n\n".join(pages)

            # Extract metadata
            metadata = self._extract_metadata(doc, file_path)

            # Check for scanned/empty PDF (no extractable text)
            if total_chars == 0:
                logger.warning(
                    "PDF appears to be scanned or has no extractable text",
                    extra={"source_path": str(file_path), "page_count": page_count},
                )

            logger.info(
                "Successfully loaded PDF",
                extra={
                    "source_path": str(file_path),
                    "page_count": page_count,
                    "char_count": total_chars,
                },
            )

            extracted_doc = ExtractedDocument(
                text=full_text,
                metadata=metadata,
                pages=pages,
            )

            return LoadResult(
                success=True,
                document=extracted_doc,
                error=None,
                source_path=file_path,
            )

        except LoadError:
            # Re-raise LoadError as-is
            raise
        except Exception as e:
            # Wrap any other exception as LoadError
            raise LoadError(
                f"Failed to load PDF: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e
        finally:
            if doc is not None:
                doc.close()

    def _extract_metadata(self, doc: Any, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from the PDF document.

        TODO(2.2.2): Move this to a dedicated extract_metadata function in step 2.2.2.
        Currently extracts basic metadata from PDF info dictionary.

        Args:
            doc: PyMuPDF document object.
            file_path: Path to the source file.

        Returns:
            Dictionary containing document metadata.
        """
        meta = doc.metadata or {}

        return {
            "source": str(file_path),
            "page_count": doc.page_count,
            "format": "pdf",
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "creation_date": meta.get("creationDate"),
            "modification_date": meta.get("modDate"),
        }

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a PDF file without loading full content.

        This is a convenience method for extracting metadata without
        loading the full document content.

        TODO(2.2.2): This will be expanded in step 2.2.2 with more
        comprehensive metadata extraction.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary containing document metadata.

        Raises:
            LoadError: If the file cannot be opened or read.
        """
        import fitz

        file_path = Path(file_path)

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. Expected .pdf",
                source_path=str(file_path),
            )

        doc = None
        try:
            doc = fitz.open(file_path)

            if doc.is_encrypted and not doc.authenticate(self._password):
                raise LoadError(
                    "PDF is encrypted and requires a password",
                    source_path=str(file_path),
                )

            return self._extract_metadata(doc, file_path)

        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to extract metadata from PDF: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e
        finally:
            if doc is not None:
                doc.close()