"""
DOCX Document Loader using python-docx.

Step 2.1.3: DocxLoader implementation using python-docx.
Step 2.2.3: DOCX metadata extraction (core properties, custom properties).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.airag.ingestion.exceptions import LoadError
from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)

logger = logging.getLogger(__name__)


class DocxLoader(DocumentLoader):
    """
    DOCX document loader using python-docx.

    Walks paragraphs and tables in document order, counts embedded images, and
    extracts core properties. Tracked-change insertions are kept; deletions are
    ignored. Legacy binary .doc files are not supported.

    Supported extensions: .docx (case-insensitive)
    """

    SUPPORTED_EXTENSIONS = frozenset({".docx"})

    def load(self, file_path: Path) -> LoadResult:
        """
        Load and extract text from a DOCX document.

        Args:
            file_path: Path to the DOCX file (str or Path).

        Returns:
            LoadResult containing the extracted document or error information.

        Raises:
            LoadError: Unsupported extension, legacy .doc, or an unreadable /
                malformed document.
        """
        import docx

        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".doc":
            raise LoadError(
                "Legacy .doc format is not supported; convert the file to .docx first",
                source_path=str(file_path),
            )
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. Expected .docx",
                source_path=str(file_path),
            )

        document: Any = None
        try:
            # Read the file exactly once.
            document = docx.Document(str(file_path))
            full_text, paragraph_count, table_count = self._scan_document(document)
            image_count = self._count_images(document)
            metadata = self._extract_metadata(
                document, file_path, paragraph_count, table_count, image_count
            )

            if not full_text:
                logger.warning(
                    "DOCX has no extractable text",
                    extra={"source_path": str(file_path), "char_count": 0},
                )

            logger.info(
                "Successfully loaded DOCX",
                extra={
                    "source_path": str(file_path),
                    "char_count": len(full_text),
                    "paragraph_count": paragraph_count,
                    "table_count": table_count,
                    "image_count": image_count,
                },
            )

            # Non-paginated format: pages holds the single full-text "page"
            # (page_count=1), consistent with PDFLoader's pages contract.
            extracted_doc = ExtractedDocument(
                text=full_text,
                metadata=metadata,
                pages=[full_text],
            )

            return LoadResult(
                success=True,
                document=extracted_doc,
                error=None,
                source_path=file_path,
            )

        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to load DOCX: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e
        finally:
            # python-docx doesn't have a close() method; nothing to clean up.
            pass

    def _scan_document(self, document: Any) -> tuple[str, int, int]:
        """
        Extract text in document order and count paragraphs/tables.

        Args:
            document: An open python-docx Document object.

        Returns:
            Tuple of (full text, paragraph count, table count).
        """
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        parts: list[str] = []
        paragraph_count = 0
        table_count = 0

        # Iterating the body children interleaves paragraphs and tables in the
        # order they actually appear, rather than dumping all paragraphs first.
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                text = self._paragraph_text(paragraph)
                if text:
                    parts.append(text)
                paragraph_count += 1
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                rows = [
                    " | ".join(cell.text for cell in row.cells)
                    for row in table.rows
                ]
                parts.append("\n".join(rows))
                table_count += 1

        return "\n\n".join(parts), paragraph_count, table_count

    @staticmethod
    def _paragraph_text(paragraph: Any) -> str:
        """
        Return paragraph text including tracked-change insertions.

        Deleted runs use w:delText rather than w:t, so walking w:t nodes keeps
        inserted text and ignores deletions.

        Args:
            paragraph: A python-docx Paragraph object.

        Returns:
            The concatenated text of all w:t nodes in the paragraph.
        """
        from docx.oxml.ns import qn

        return "".join(node.text or "" for node in paragraph._p.iter(qn("w:t")))

    @staticmethod
    def _count_images(document: Any) -> int:
        """
        Count embedded image relationships (inline and floating).

        Args:
            document: An open python-docx Document object.

        Returns:
            Number of image parts related to the document, or 0 on failure.
        """
        try:
            return sum(
                1
                for rel in document.part.rels.values()
                if rel.reltype.endswith("/image")
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not count DOCX images",
                extra={"error": str(e)},
            )
            return 0

    @staticmethod
    def _core_prop(core_properties: Any, name: str) -> Any:
        """
        Best-effort read of a single core property.

        Args:
            core_properties: python-docx CoreProperties object (or None).
            name: Name of the property to read.

        Returns:
            The property value as a string/None, with datetimes ISO-formatted;
            None when the property is missing or unreadable.
        """
        try:
            value = getattr(core_properties, name, None)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return value or None

    def _extract_metadata(
        self,
        document: Any,
        file_path: Path,
        paragraph_count: int,
        table_count: int,
        image_count: int,
    ) -> dict[str, Any]:
        """
        Extract metadata from the DOCX document.

        Populates the following fields (per step 2.2.3):
            - source: original file path
            - page_count: always 1 for non-paginated formats
            - format: "docx"
            - title: document title from core properties, or None if missing/empty
            - author: document author from core properties, or None if missing/empty
            - created: creation timestamp as ISO-8601 string (e.g.,
              "2024-01-15T14:30:00+00:00"), or None if missing/malformed
            - modified: last modification timestamp as ISO-8601 string, or None
            - paragraph_count: number of paragraphs in the body
            - table_count: number of tables in the body
            - image_count: number of embedded images

        Missing core properties become None. Empty strings from python-docx are
        normalized to None. All values are JSON-serializable.

        Args:
            document: An open python-docx Document object.
            file_path: Path to the source file.
            paragraph_count: Number of paragraphs found in the body.
            table_count: Number of tables found in the body.
            image_count: Number of embedded images.

        Returns:
            Dictionary containing document metadata.
        """
        try:
            core = document.core_properties
        except Exception:  # noqa: BLE001
            core = None

        # Extract raw values, normalizing empty strings to None
        title = self._core_prop(core, "title")
        author = self._core_prop(core, "author")
        created = self._core_prop(core, "created")
        modified = self._core_prop(core, "modified")

        metadata = {
            "source": str(file_path),
            "page_count": 1,
            "format": "docx",
            "title": title,
            "author": author,
            "created": created,
            "modified": modified,
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "image_count": image_count,
        }

        # Log extraction result
        non_none_count = sum(
            1 for v in (title, author, created, modified) if v is not None
        )
        logger.debug(
            "Extracted DOCX metadata",
            extra={
                "source_path": str(file_path),
                "paragraph_count": paragraph_count,
                "table_count": table_count,
                "image_count": image_count,
                "non_none_fields": non_none_count,
            },
        )

# Check if all meaningful core properties are missing/empty
        meaningful_fields = [
            getattr(core, "title", None),
            getattr(core, "author", None),
            getattr(core, "created", None),
            getattr(core, "modified", None),
            getattr(core, "subject", None),
            getattr(core, "keywords", None),
        ] if core else []
        if not any(v for v in meaningful_fields if v):
            logger.warning(
                "DOCX has no meaningful core properties; metadata fields will be None",
                extra={"source_path": str(file_path)},
            )

        return metadata

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a DOCX file without loading full content.

        This is a convenience method for extracting metadata without
        loading the full document content.

        Args:
            file_path: Path to the DOCX file (str or Path).

        Returns:
            Dictionary containing document metadata.

        Raises:
            LoadError: If the file cannot be opened or read.
        """
        import docx

        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".doc":
            raise LoadError(
                "Legacy .doc format is not supported; convert the file to .docx first",
                source_path=str(file_path),
            )
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. Expected .docx",
                source_path=str(file_path),
            )

        document: Any = None
        try:
            document = docx.Document(str(file_path))
            _, paragraph_count, table_count = self._scan_document(document)
            image_count = self._count_images(document)
            return self._extract_metadata(
                document, file_path, paragraph_count, table_count, image_count
            )
        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to extract metadata from DOCX: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e
        finally:
            # python-docx doesn't have a close() method; nothing to clean up.
            pass