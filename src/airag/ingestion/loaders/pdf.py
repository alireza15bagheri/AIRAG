"""
PDF Document Loader using PyMuPDF (fitz).

Step 2.1.2: PDFLoader implementation using PyMuPDF (fitz).
Step 2.2.2: PDF metadata extraction (page count, title, author, creation date).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta, timezone
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

    @staticmethod
    def _parse_pdf_date(raw_date: str | None, file_path: Path) -> str | None:
        """
        Parse a PDF date string to ISO-8601 format.

        PDF dates follow the format: D:YYYYMMDDHHmmSS±HH'mm' or D:YYYYMMDDHHmmSSZ
        (per PDF Reference 1.7, section 3.8.2). Examples:
            D:20240115143000+00'00'
            D:20240115143000Z
            D:20240115143000

        Args:
            raw_date: Raw date string from PDF Info dict (may be None or empty).
            file_path: Source file path for logging context.

        Returns:
            ISO-8601 formatted string (e.g., "2024-01-15T14:30:00+00:00"),
            or None if parsing fails or input is None/empty.

        Note:
            This uses only stdlib datetime for parsing. On parse failure,
            logs a WARNING and returns None.
        """
        if not raw_date:
            return None

        # Remove leading "D:" prefix if present
        raw_date = raw_date.removeprefix("D:")

        if not raw_date:
            logger.warning(
                "Empty creation date after stripping 'D:' prefix",
                extra={"source_path": str(file_path)},
            )
            return None

        # Pattern: YYYYMMDDHHMMSS[±HHMM|Z] (optional time components)
        # PDF date format: YYYYMMDDHHMMSS±HH'MM' or YYYYMMDDHHMMSSZ
        # Some PDFs may have incomplete dates (e.g., just YYYYMMDD)
        pattern = r"^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?(Z|([+-])(\d{2})'?(\d{2})?'?)?$"
        match = re.match(pattern, raw_date)

        if not match:
            logger.warning(
                "Malformed PDF creation date",
                extra={"source_path": str(file_path), "raw_date": raw_date},
            )
            return None

        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4) or 0)
            minute = int(match.group(5) or 0)
            second = int(match.group(6) or 0)

            # Handle timezone
            tz_info = None
            tz_match = match.group(7)
            if tz_match:
                if tz_match == "Z":
                    tz_info = UTC
                else:
                    # ±HH'MM' format
                    sign = match.group(8)
                    tz_hour = int(match.group(9) or 0)
                    tz_minute = int(match.group(10) or 0)
                    offset = timedelta(hours=tz_hour, minutes=tz_minute)
                    if sign == "-":
                        offset = -offset
                    tz_info = timezone(offset)

            dt = datetime(year, month, day, hour, minute, second, tzinfo=tz_info)
            return dt.isoformat()

        except (ValueError, OverflowError) as e:
            logger.warning(
                "Failed to parse PDF creation date",
                extra={"source_path": str(file_path), "raw_date": raw_date, "error": str(e)},
            )
            return None

    def _extract_metadata(self, doc: Any, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from the PDF document.

        Populates the following fields (per step 2.2.2):
            - source: original file path
            - page_count: total number of pages
            - format: "pdf"
            - title: document title from Info dict, or None if missing/empty
            - author: document author from Info dict, or None if missing/empty
            - creation_date: creation timestamp as ISO-8601 string (e.g.,
              "2024-01-15T14:30:00+00:00"), or None if missing/malformed

        Missing Info dict fields become None. Empty strings from PyMuPDF are
        normalized to None. Malformed creation dates are logged as WARNING and
        result in creation_date=None.

        Args:
            doc: Open PyMuPDF document object.
            file_path: Path to the source file.

        Returns:
            Dictionary containing document metadata.
        """
        meta = doc.metadata or {}

        # Extract raw values, normalizing empty strings to None
        title = meta.get("title")
        author = meta.get("author")
        raw_creation_date = meta.get("creationDate")

        # Normalize empty strings to None
        if title == "":
            title = None
        if author == "":
            author = None

        # Parse creation date to ISO-8601
        creation_date = self._parse_pdf_date(raw_creation_date, file_path)

        metadata = {
            "source": str(file_path),
            "page_count": doc.page_count,
            "format": "pdf",
            "title": title,
            "author": author,
            "creation_date": creation_date,
        }

        # Log extraction result
        non_none_count = sum(1 for v in (title, author, creation_date) if v is not None)
        logger.debug(
            "Extracted PDF metadata",
            extra={
                "source_path": str(file_path),
                "page_count": doc.page_count,
                "non_none_fields": non_none_count,
            },
        )

        # Check if all meaningful metadata fields are missing/empty
        meaningful_fields = [meta.get("title"), meta.get("author"), meta.get("creationDate"), meta.get("subject"), meta.get("creator"), meta.get("producer")]
        if not any(v for v in meaningful_fields if v):
            logger.warning(
                "PDF has no meaningful Info dictionary metadata; metadata fields will be None",
                extra={"source_path": str(file_path)},
            )

        return metadata

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a PDF file without loading full content.

        This is a convenience method for extracting metadata without
        loading the full document content.

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