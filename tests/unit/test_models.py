"""
Unit tests for the finalized ingestion models (step 2.2.1).

Covers the ExtractedDocument contract (defaults, equality, re-export identity),
construction from each observed loader-shaped input, and a conformance net
asserting every loader's load() returns the canonical ExtractedDocument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.airag.ingestion.loaders.base import ExtractedDocument as BaseExtractedDocument
from src.airag.ingestion.loaders.docx import DocxLoader
from src.airag.ingestion.loaders.markdown import MarkdownLoader
from src.airag.ingestion.loaders.pdf import PDFLoader
from src.airag.ingestion.loaders.text import TextLoader
from src.airag.ingestion.models import ExtractedDocument


class TestContract:
    """Contract-level guarantees of the finalized dataclass."""

    def test_reexport_identity(self) -> None:
        # base.py re-exports the canonical class; there must be no parallel shape.
        assert BaseExtractedDocument is ExtractedDocument

    def test_default_pages_is_empty_list(self) -> None:
        doc = ExtractedDocument(text="Hello", metadata={})
        assert doc.pages == []
        assert doc.pages is not None

    def test_default_factory_isolation(self) -> None:
        # Each instance must get its own list, not a shared mutable default.
        a = ExtractedDocument(text="a", metadata={})
        b = ExtractedDocument(text="b", metadata={})
        a.pages.append("mutated")
        assert b.pages == []

    def test_equality_by_value(self) -> None:
        stamp = "2026-01-01T00:00:00+00:00"
        a = ExtractedDocument(
            text="x", metadata={"k": 1}, pages=["x"], ingested_at=stamp
        )
        b = ExtractedDocument(
            text="x", metadata={"k": 1}, pages=["x"], ingested_at=stamp
        )
        c = ExtractedDocument(
            text="y", metadata={"k": 1}, pages=["x"], ingested_at=stamp
        )
        assert a == b
        assert a != c


# Metadata shapes observed in the four loaders during pre-flight (2.2.1).
LOADER_SHAPES: list[tuple[str, str, list[str], dict[str, Any]]] = [
    (
        "pdf",
        "Page one\n\nPage two",
        ["Page one", "Page two"],
        {
            "source": "sample.pdf",
            "page_count": 2,
            "format": "pdf",
            "title": None,
            "author": None,
            "subject": None,
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
        },
    ),
    (
        "docx",
        "A paragraph",
        ["A paragraph"],
        {
            "source": "sample.docx",
            "page_count": 1,
            "format": "docx",
            "title": None,
            "author": None,
            "created": None,
            "modified": None,
            "table_count": 0,
            "image_count": 0,
            "paragraph_count": 1,
        },
    ),
    (
        "markdown",
        "Body text",
        ["Body text"],
        {
            "source": "sample.md",
            "page_count": 1,
            "format": "markdown",
            "frontmatter": {},
            "headings": [{"level": 1, "text": "Heading"}],
            "heading_count": 1,
            "table_count": 0,
            "has_code_blocks": False,
        },
    ),
    (
        "text",
        "plain line",
        ["plain line"],
        {
            "source": "sample.txt",
            "page_count": 1,
            "format": "text",
            "encoding": "utf_8",
            "detection_method": "utf-8-fallback",
            "line_count": 1,
            "char_count": 10,
            "word_count": 2,
        },
    ),
]


class TestLoaderShapedConstruction:
    """Construction from each observed loader-shaped input."""

    @pytest.mark.parametrize(
        ("fmt", "text", "pages", "metadata"),
        LOADER_SHAPES,
        ids=[shape[0] for shape in LOADER_SHAPES],
    )
    def test_round_trip(
        self, fmt: str, text: str, pages: list[str], metadata: dict[str, Any]
    ) -> None:
        doc = ExtractedDocument(text=text, metadata=metadata, pages=pages)
        assert doc.text == text
        assert doc.pages == pages
        assert doc.metadata == metadata
        assert doc.metadata["format"] == fmt


def _make_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Conformance PDF text")
    doc.save(str(path))
    doc.close()


def _make_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    document.add_paragraph("Conformance DOCX text")
    document.save(str(path))


def _make_markdown(path: Path) -> None:
    path.write_text("# Heading\n\nConformance markdown text\n", encoding="utf-8")


def _make_text(path: Path) -> None:
    path.write_text("Conformance plain text\n", encoding="utf-8")


class TestLoaderConformance:
    """Every loader's load() must return the canonical ExtractedDocument."""

    @pytest.mark.parametrize(
        ("loader_cls", "suffix", "builder"),
        [
            (PDFLoader, ".pdf", _make_pdf),
            (DocxLoader, ".docx", _make_docx),
            (MarkdownLoader, ".md", _make_markdown),
            (TextLoader, ".txt", _make_text),
        ],
        ids=["pdf", "docx", "markdown", "text"],
    )
    def test_load_returns_canonical_model(
        self,
        tmp_path: Path,
        loader_cls: type,
        suffix: str,
        builder: Any,
    ) -> None:
        file_path = tmp_path / f"conformance{suffix}"
        builder(file_path)

        result = loader_cls().load(file_path)

        assert result.success is True
        doc = result.document
        assert isinstance(doc, ExtractedDocument)
        assert isinstance(doc.text, str)
        assert isinstance(doc.metadata, dict)
        assert isinstance(doc.pages, list)
        assert all(isinstance(page, str) for page in doc.pages)
        assert len(doc.pages) >= 1
