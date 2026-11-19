"""
Unit tests for the step 2.1.3 loaders: DocxLoader, MarkdownLoader, TextLoader.

Fixtures are generated in tmp_path (no committed binaries), following the
PDFLoader test strategy from step 2.1.2.
"""

from __future__ import annotations

import logging
import re
import struct
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import docx
import pytest
from docx.oxml import OxmlElement
from docx.shared import Inches

from src.airag.ingestion.exceptions import LoadError
from src.airag.ingestion.loaders.base import DocumentLoader, LoadResult
from src.airag.ingestion.loaders.docx import DocxLoader
from src.airag.ingestion.loaders.markdown import MarkdownLoader, _load_yaml
from src.airag.ingestion.loaders.text import TextLoader, _detection_confidence


def _png_bytes() -> bytes:
    """Build a minimal valid 1x1 red PNG in memory."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib_crc32(tag + data))
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"
    return header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib_compress(raw)) + chunk(b"IEND", b"")


def zlib_crc32(data: bytes) -> int:
    """CRC32 helper for PNG chunks."""
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF


def zlib_compress(data: bytes) -> bytes:
    """zlib helper for PNG chunks."""
    import zlib

    return zlib.compress(data)


class TestDocxLoader:
    """Tests for DocxLoader."""

    @pytest.fixture
    def loader(self) -> DocxLoader:
        """Create a DocxLoader instance."""
        return DocxLoader()

    @pytest.fixture
    def simple_docx(self, tmp_path: Path) -> Path:
        """Create a simple multi-paragraph DOCX."""
        path = tmp_path / "simple.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")
        document.save(path)
        return path

    def test_load_valid_docx(self, loader: DocxLoader, simple_docx: Path) -> None:
        """Valid DOCX extracts paragraphs with correct metadata and pages."""
        result = loader.load(simple_docx)

        assert result.success is True
        assert result.error is None
        assert result.document is not None
        assert "First paragraph" in result.document.text
        assert "Second paragraph" in result.document.text
        assert result.document.pages == [result.document.text]
        meta = result.document.metadata
        assert meta["format"] == "docx"
        assert meta["source"] == str(simple_docx)
        assert meta["page_count"] == 1
        assert meta["paragraph_count"] == 2
        assert meta["table_count"] == 0
        assert meta["image_count"] == 0

    def test_table_text_in_document_order(
        self, loader: DocxLoader, tmp_path: Path
    ) -> None:
        """Table rows are rendered in order relative to surrounding paragraphs."""
        path = tmp_path / "tables.docx"
        document = docx.Document()
        document.add_paragraph("Alpha")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "A1"
        table.cell(0, 1).text = "B1"
        table.cell(1, 0).text = "A2"
        table.cell(1, 1).text = "B2"
        document.add_paragraph("Omega")
        document.save(path)

        result = loader.load(path)
        assert result.success is True
        text = result.document.text
        assert "A1 | B1" in text
        assert "A2 | B2" in text
        assert text.index("Alpha") < text.index("A1 | B1") < text.index("Omega")
        assert result.document.metadata["table_count"] == 1

    def test_images_counted(self, loader: DocxLoader, tmp_path: Path) -> None:
        """Embedded images are counted without extracting them."""
        import io

        path = tmp_path / "image.docx"
        document = docx.Document()
        document.add_paragraph("With an image")
        document.add_picture(io.BytesIO(_png_bytes()), width=Inches(1))
        document.save(path)

        result = loader.load(path)
        assert result.success is True
        assert result.document.metadata["image_count"] == 1

    def test_tracked_changes(self, loader: DocxLoader, tmp_path: Path) -> None:
        """Inserted runs are kept; deleted runs are ignored."""
        path = tmp_path / "tracked.docx"
        document = docx.Document()
        paragraph = document.add_paragraph("Base")
        inserted = OxmlElement("w:ins")
        run = OxmlElement("w:r")
        text_node = OxmlElement("w:t")
        text_node.text = " INSERTED"
        run.append(text_node)
        inserted.append(run)
        paragraph._p.append(inserted)

        deleted = OxmlElement("w:del")
        run2 = OxmlElement("w:r")
        del_text = OxmlElement("w:delText")
        del_text.text = " DELETED"
        run2.append(del_text)
        deleted.append(run2)
        paragraph._p.append(deleted)
        document.save(path)

        result = loader.load(path)
        assert result.success is True
        assert "Base INSERTED" in result.document.text
        assert "DELETED" not in result.document.text

    def test_empty_docx_succeeds(
        self, loader: DocxLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A valid empty DOCX succeeds with empty text and a warning."""
        path = tmp_path / "empty.docx"
        docx.Document().save(path)

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.success is True
        assert result.document is not None
        assert result.document.text == ""
        assert result.document.pages == [""]
        assert any("no extractable text" in r.message.lower() for r in caplog.records)

    def test_missing_core_properties(
        self, loader: DocxLoader, tmp_path: Path
    ) -> None:
        """Missing core properties degrade to None without raising."""
        path = tmp_path / "no_core.docx"
        stripped = tmp_path / "stripped.docx"
        document = docx.Document()
        document.add_paragraph("Body")
        document.save(path)

        with zipfile.ZipFile(path) as source, zipfile.ZipFile(
            stripped, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                if item.filename == "docProps/core.xml":
                    continue
                data = source.read(item.filename)
                if item.filename == "_rels/.rels":
                    rels = data.decode("utf-8")
                    rels = re.sub(
                        r'<Relationship[^>]*Type="[^"]*/core-properties"[^>]*/>',
                        "",
                        rels,
                    )
                    data = rels.encode("utf-8")
                target.writestr(item, data)

        result = loader.load(stripped)
        assert result.success is True
        meta = result.document.metadata
        for key in ("title", "author", "created", "modified"):
            assert key in meta
            assert meta[key] is None or isinstance(meta[key], str)

    def test_core_prop_swallows_property_errors(self) -> None:
        """A property that raises is reported as None."""

        class Broken:
            @property
            def title(self) -> str:
                raise RuntimeError("broken property")

        assert DocxLoader._core_prop(Broken(), "title") is None

    def test_corrupt_docx(self, loader: DocxLoader, tmp_path: Path) -> None:
        """A non-zip file renamed to .docx raises LoadError with a cause."""
        path = tmp_path / "corrupt.docx"
        path.write_bytes(b"this is not a zip archive")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "failed to load docx" in str(exc_info.value).lower()
        assert exc_info.value.cause is not None

    def test_legacy_doc_rejected(self, loader: DocxLoader, tmp_path: Path) -> None:
        """Legacy .doc files are rejected with a conversion hint."""
        path = tmp_path / "legacy.doc"
        path.write_bytes(b"old binary")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "legacy" in str(exc_info.value).lower()
        assert "convert" in str(exc_info.value).lower()
        assert ".docx" in str(exc_info.value).lower()

    def test_wrong_extension(self, loader: DocxLoader, tmp_path: Path) -> None:
        """A non-DOCX extension raises LoadError."""
        path = tmp_path / "notes.md"
        path.write_text("# Notes")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "unsupported" in str(exc_info.value).lower()

    def test_case_insensitive_extension_and_str_path(
        self, loader: DocxLoader, tmp_path: Path
    ) -> None:
        """Uppercase extension and str paths are accepted."""
        path = tmp_path / "UPPER.DOCX"
        docx.Document().save(path)

        result = loader.load(str(path))
        assert result.success is True
        assert result.document.metadata["source"] == str(path)

    def test_extract_metadata(self, loader: DocxLoader, simple_docx: Path) -> None:
        """The metadata-only convenience method returns the stub fields."""
        meta = loader.extract_metadata(simple_docx)
        assert meta["format"] == "docx"
        assert meta["paragraph_count"] == 2
        assert meta["source"] == str(simple_docx)


    def test_count_images_degrades(self, caplog: pytest.LogCaptureFixture) -> None:
        """An object without image relationships yields count 0 and a warning."""
        with caplog.at_level(logging.WARNING):
            assert DocxLoader._count_images(object()) == 0
        assert any("count docx images" in r.message.lower() for r in caplog.records)

    def test_extract_metadata_unreadable_core_properties(self, tmp_path: Path) -> None:
        """Core-property access errors degrade to None values."""

        class BadDocument:
            @property
            def core_properties(self) -> Any:
                raise RuntimeError("no core")

        meta = DocxLoader()._extract_metadata(
            BadDocument(), tmp_path / "bad.docx", 1, 2, 3
        )
        assert meta["title"] is None
        assert meta["author"] is None
        assert meta["paragraph_count"] == 1
        assert meta["table_count"] == 2
        assert meta["image_count"] == 3

    def test_load_error_from_helper_is_reraised(
        self, loader: DocxLoader, simple_docx: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LoadError raised inside the load try-block passes through unchanged."""

        def boom(document: Any) -> int:
            raise LoadError("helper failed", source_path=str(simple_docx))

        monkeypatch.setattr(loader, "_count_images", boom)
        with pytest.raises(LoadError, match="helper failed"):
            loader.load(simple_docx)

    def test_extract_metadata_rejects_legacy_and_wrong_extension(
        self, loader: DocxLoader, tmp_path: Path
    ) -> None:
        """The metadata-only method applies the same extension guard."""
        legacy = tmp_path / "legacy.doc"
        legacy.write_bytes(b"old")
        wrong = tmp_path / "notes.md"
        wrong.write_text("# x")

        with pytest.raises(LoadError, match="(?i)legacy"):
            loader.extract_metadata(legacy)
        with pytest.raises(LoadError, match="(?i)unsupported"):
            loader.extract_metadata(wrong)

    def test_extract_metadata_corrupt_file(
        self, loader: DocxLoader, tmp_path: Path
    ) -> None:
        """A corrupt DOCX raises LoadError from extract_metadata."""
        path = tmp_path / "corrupt.docx"
        path.write_bytes(b"not a zip")

        with pytest.raises(LoadError) as exc_info:
            loader.extract_metadata(path)

        assert "failed to extract metadata" in str(exc_info.value).lower()
        assert exc_info.value.cause is not None

    def test_extract_metadata_reraises_load_error(
        self, loader: DocxLoader, simple_docx: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LoadError from a helper passes through extract_metadata unchanged."""

        def boom(document: Any) -> tuple[str, int, int]:
            raise LoadError("scan failed", source_path=str(simple_docx))

        monkeypatch.setattr(loader, "_scan_document", boom)
        with pytest.raises(LoadError, match="scan failed"):
            loader.extract_metadata(simple_docx)


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    @pytest.fixture
    def loader(self) -> MarkdownLoader:
        """Create a MarkdownLoader instance."""
        return MarkdownLoader()

    def test_frontmatter_stripped_and_parsed(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """Valid YAML frontmatter is removed from the body and kept in metadata."""
        path = tmp_path / "front.md"
        content = "---\ntitle: Hello\ntags:\n  - alpha\n  - beta\n---\n# Intro\nBody text\n"
        path.write_text(content, encoding="utf-8")

        result = loader.load(path)
        assert result.success is True
        doc = result.document
        assert doc.text.startswith("# Intro")
        assert "title: Hello" not in doc.text
        assert doc.metadata["frontmatter"] == {
            "title": "Hello",
            "tags": ["alpha", "beta"],
        }
        assert doc.pages == [doc.text]

    def test_no_frontmatter(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """A file without frontmatter keeps its body verbatim."""
        path = tmp_path / "plain.md"
        content = "# Title\n\nSome *text*.\n"
        path.write_text(content, encoding="utf-8")

        result = loader.load(path)
        assert result.document.text == content
        assert result.document.metadata["frontmatter"] == {}

    def test_malformed_yaml_warns_and_preserves_body(
        self, loader: MarkdownLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed YAML frontmatter warns but stays verbatim in the body."""
        path = tmp_path / "bad.md"
        content = "---\ntitle: [unclosed\n---\n# Head\n"
        path.write_text(content, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.success is True
        assert result.document.text == content
        assert result.document.metadata["frontmatter"] == {}
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_non_mapping_frontmatter_warns(
        self, loader: MarkdownLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A scalar frontmatter document is treated as body text."""
        path = tmp_path / "scalar.md"
        content = "---\njust a scalar\n---\n# Head\n"
        path.write_text(content, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.document.text == content
        assert result.document.metadata["frontmatter"] == {}
        assert any("not a mapping" in r.message.lower() for r in caplog.records)

    def test_heading_hierarchy(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """Heading levels and text are captured in document order."""
        path = tmp_path / "headings.md"
        path.write_text("# One\n\n## Two\n\n### Three\n\n###### Six\n", encoding="utf-8")

        result = loader.load(path)
        assert result.document.metadata["headings"] == [
            {"level": 1, "text": "One"},
            {"level": 2, "text": "Two"},
            {"level": 3, "text": "Three"},
            {"level": 6, "text": "Six"},
        ]
        assert result.document.metadata["heading_count"] == 4

    def test_fenced_code_preserved_verbatim(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """Fenced code keeps # and | characters and is not scanned as structure."""
        path = tmp_path / "code.md"
        code = "```python\n# not a heading\n| not | a table |\nprint('# still code')\n```"
        path.write_text(f"# Real\n\n{code}\n", encoding="utf-8")

        result = loader.load(path)
        assert code in result.document.text
        assert result.document.metadata["headings"] == [
            {"level": 1, "text": "Real"}
        ]
        assert result.document.metadata["table_count"] == 0
        assert result.document.metadata["has_code_blocks"] is True

    def test_indented_code_detected(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """Indented code blocks set has_code_blocks."""
        path = tmp_path / "indent.md"
        path.write_text("Text\n\n    indented = True\n", encoding="utf-8")

        result = loader.load(path)
        assert result.document.metadata["has_code_blocks"] is True

    def test_pipe_table_preserved_and_counted(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """Pipe tables stay in the body and are counted."""
        path = tmp_path / "table.md"
        table = "| Name | Value |\n| ---- | ----- |\n| a | b |\n"
        path.write_text(table, encoding="utf-8")

        result = loader.load(path)
        assert table in result.document.text
        assert result.document.metadata["table_count"] == 1

    def test_empty_file(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """An empty markdown file is valid."""
        path = tmp_path / "empty.md"
        path.write_text("", encoding="utf-8")

        result = loader.load(path)
        assert result.success is True
        assert result.document.text == ""
        assert result.document.pages == [""]

    def test_bom_tolerated(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """A UTF-8 BOM does not leak into the body text."""
        path = tmp_path / "bom.md"
        path.write_bytes(b"\xef\xbb\xbf# Bom\n")

        result = loader.load(path)
        assert result.document.text == "# Bom\n"

    def test_wrong_extension(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """Unsupported extensions (including .mdx) raise LoadError."""
        path = tmp_path / "doc.mdx"
        path.write_text("# MDX", encoding="utf-8")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "unsupported" in str(exc_info.value).lower()

    def test_markdown_extension_and_uppercase(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """Both .md and uppercase .MARKDOWN are accepted."""
        md = tmp_path / "a.md"
        md.write_text("# A", encoding="utf-8")
        markdown = tmp_path / "B.MARKDOWN"
        markdown.write_text("## B", encoding="utf-8")

        assert loader.load(md).success is True
        assert loader.load(markdown).success is True

    def test_extract_metadata(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """The metadata-only convenience method records format and headings."""
        path = tmp_path / "meta.markdown"
        path.write_text("# Title\n", encoding="utf-8")

        meta = loader.extract_metadata(path)
        assert meta["format"] == "markdown"
        assert meta["heading_count"] == 1
        assert meta["source"] == str(path)


    def test_no_yaml_loader_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A yaml module without any safe loader raises a clear AttributeError."""

        class NoSafeYaml:
            @staticmethod
            def safe_load(block: str) -> Any:
                raise AttributeError("safe_load removed")

        monkeypatch.setitem(sys.modules, "yaml", NoSafeYaml())
        with pytest.raises(AttributeError, match="No safe YAML loader"):
            _load_yaml("a: b")

    def test_load_unreadable_path(self, loader: MarkdownLoader, tmp_path: Path) -> None:
        """A directory with a .md suffix raises LoadError."""
        directory = tmp_path / "bad.md"
        directory.mkdir()

        with pytest.raises(LoadError) as exc_info:
            loader.load(directory)

        assert "failed to load markdown" in str(exc_info.value).lower()

    def test_load_error_from_parse_is_reraised(
        self, loader: MarkdownLoader, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A LoadError from parsing passes through unchanged."""
        path = tmp_path / "x.md"
        path.write_text("# x", encoding="utf-8")

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise LoadError("parse failed", source_path=str(path))

        monkeypatch.setattr(loader, "_parse", boom)
        with pytest.raises(LoadError, match="parse failed"):
            loader.load(path)

    def test_extract_metadata_wrong_extension(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """The metadata-only method applies the extension guard."""
        path = tmp_path / "doc.mdx"
        path.write_text("# x", encoding="utf-8")

        with pytest.raises(LoadError, match="(?i)unsupported"):
            loader.extract_metadata(path)

    def test_extract_metadata_unreadable_path(
        self, loader: MarkdownLoader, tmp_path: Path
    ) -> None:
        """An unreadable path raises LoadError from extract_metadata."""
        directory = tmp_path / "dir.md"
        directory.mkdir()

        with pytest.raises(LoadError) as exc_info:
            loader.extract_metadata(directory)

        assert "failed to extract metadata" in str(exc_info.value).lower()

    def test_extract_metadata_reraises_load_error(
        self,
        loader: MarkdownLoader,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LoadError from parsing passes through extract_metadata unchanged."""
        path = tmp_path / "x.md"
        path.write_text("# x", encoding="utf-8")

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise LoadError("parse failed", source_path=str(path))

        monkeypatch.setattr(loader, "_parse", boom)
        with pytest.raises(LoadError, match="parse failed"):
            loader.extract_metadata(path)


class TestTextLoader:
    """Tests for TextLoader."""

    @pytest.fixture
    def loader(self) -> TextLoader:
        """Create a TextLoader instance."""
        return TextLoader()

    def test_utf8(self, loader: TextLoader, tmp_path: Path) -> None:
        """UTF-8 text is decoded and counted."""
        path = tmp_path / "utf8.txt"
        content = "Café naïve — ok\nSecond line\n"
        path.write_bytes(content.encode("utf-8"))

        result = loader.load(path)
        assert result.success is True
        assert result.document.text == content
        meta = result.document.metadata
        assert meta["format"] == "text"
        assert meta["source"] == str(path)
        assert meta["page_count"] == 1
        assert meta["detection_method"] == "charset-normalizer"
        assert meta["line_count"] == 2
        assert meta["word_count"] == 6  # "Café", "naïve", "—", "ok", "Second", "line"
        assert result.document.pages == [content]

    def test_utf16_with_bom(self, loader: TextLoader, tmp_path: Path) -> None:
        """UTF-16 with BOM is decoded correctly."""
        path = tmp_path / "utf16.txt"
        content = "Hello café\n"
        path.write_bytes(content.encode("utf-16"))

        result = loader.load(path)
        assert result.document.text == content
        assert result.document.metadata["encoding"].lower().replace("_", "-") in {
            "utf-16",
            "utf-16-le",
            "utf-16-be",
        }

    def test_cp1252_bytes_load(
        self, loader: TextLoader, tmp_path: Path
    ) -> None:
        """Non-ASCII single-byte content loads with a recorded encoding."""
        path = tmp_path / "latin.txt"
        path.write_bytes("Café déjà vu résumé naïve façade\n".encode("cp1252"))

        result = loader.load(path)
        assert result.success is True
        assert result.document.text.strip()
        assert isinstance(result.document.metadata["encoding"], str)
        assert result.document.metadata["detection_method"] in {
            "charset-normalizer",
            "chardet",
            "utf-8-fallback",
        }

    def test_explicit_encoding_via_monkeypatch(
        self, loader: TextLoader, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detected cp1252 encoding decodes non-ASCII bytes correctly."""

        class FakeMatch:
            encoding = "cp1252"
            confidence = 1.0
            percent_chaos = 0.0

        class FakeMatches:
            def best(self) -> FakeMatch:
                return FakeMatch()

        monkeypatch.setattr(
            "charset_normalizer.from_bytes", lambda data: FakeMatches()
        )
        path = tmp_path / "cp1252.txt"
        content = "Café déjà vu\n"
        path.write_bytes(content.encode("cp1252"))

        result = loader.load(path)
        assert result.document.text == content
        assert result.document.metadata["encoding"] == "cp1252"

    def test_line_endings_normalized(self, loader: TextLoader, tmp_path: Path) -> None:
        """CRLF and CR are normalized to LF."""
        path = tmp_path / "lines.txt"
        path.write_bytes(b"a\r\nb\rc\n")

        result = loader.load(path)
        assert result.document.text == "a\nb\nc\n"
        assert result.document.metadata["line_count"] == 3

    def test_empty_file(self, loader: TextLoader, tmp_path: Path) -> None:
        """A zero-byte text file is valid."""
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")

        result = loader.load(path)
        assert result.success is True
        assert result.document.text == ""
        assert result.document.pages == [""]
        meta = result.document.metadata
        assert meta["line_count"] == 0
        assert meta["char_count"] == 0
        assert meta["word_count"] == 0
        assert meta["detection_method"] == "empty-file"

    def test_large_file(self, loader: TextLoader, tmp_path: Path) -> None:
        """Larger files are loaded with correct statistics."""
        path = tmp_path / "large.txt"
        content = "word " * 20000
        path.write_bytes(content.encode("utf-8"))

        result = loader.load(path)
        assert result.success is True
        assert result.document.metadata["word_count"] == 20000
        assert result.document.metadata["char_count"] == len(content)

    def test_low_confidence_fallback_warns(
        self,
        loader: TextLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No detection match falls back to UTF-8 with a warning."""

        class NoMatches:
            def best(self) -> None:
                return None

        monkeypatch.setattr(
            "charset_normalizer.from_bytes", lambda data: NoMatches()
        )
        path = tmp_path / "fallback.text"
        path.write_bytes(b"plain \xff text")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.success is True
        assert "\ufffd" in result.document.text
        assert result.document.metadata["detection_method"] == "utf-8-fallback"
        assert any("low-confidence" in r.message.lower() for r in caplog.records)

    def test_binary_content_rejected(self, loader: TextLoader, tmp_path: Path) -> None:
        """NUL bytes indicate binary content and raise LoadError."""
        path = tmp_path / "binary.txt"
        path.write_bytes(b"abc\x00def")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "binary" in str(exc_info.value).lower()

    def test_unreadable_path(self, loader: TextLoader, tmp_path: Path) -> None:
        """A directory masquerading as a .txt file raises LoadError."""
        directory = tmp_path / "directory.txt"
        directory.mkdir()

        with pytest.raises(LoadError) as exc_info:
            loader.load(directory)

        assert "failed to load text" in str(exc_info.value).lower()

    def test_wrong_extension(self, loader: TextLoader, tmp_path: Path) -> None:
        """Unsupported extensions raise LoadError."""
        path = tmp_path / "data.csv"
        path.write_bytes(b"a,b\n")

        with pytest.raises(LoadError) as exc_info:
            loader.load(path)

        assert "unsupported" in str(exc_info.value).lower()

    def test_uppercase_extension_and_str_path(
        self, loader: TextLoader, tmp_path: Path
    ) -> None:
        """Uppercase extensions and str paths are accepted."""
        path = tmp_path / "UPPER.TXT"
        path.write_bytes(b"content")

        result = loader.load(str(path))
        assert result.success is True
        assert result.document.metadata["source"] == str(path)

    def test_extract_metadata(self, loader: TextLoader, tmp_path: Path) -> None:
        """The metadata-only convenience method returns text statistics."""
        path = tmp_path / "meta.txt"
        path.write_bytes(b"one two three\n")

        meta = loader.extract_metadata(path)
        assert meta["format"] == "text"
        assert meta["word_count"] == 3
        assert meta["source"] == str(path)


    def test_detection_confidence_helpers(self) -> None:
        """Confidence is normalized across charset-normalizer versions."""

        class Modern:
            confidence = 0.75

        class Newer:
            percent_chaos = 25.0

        class Unknown:
            pass

        assert _detection_confidence(Modern()) == 0.75
        assert _detection_confidence(Newer()) == 0.75
        assert _detection_confidence(Unknown()) == 1.0

    def test_charset_normalizer_unavailable(
        self,
        loader: TextLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing charset-normalizer falls back to UTF-8 with a warning."""
        path = tmp_path / "plain.txt"
        path.write_bytes(b"plain ascii text")

        monkeypatch.setitem(sys.modules, "charset_normalizer", None)
        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.success is True
        assert result.document.text == "plain ascii text"
        assert any("low-confidence" in r.message.lower() for r in caplog.records)

    def test_detector_exception_falls_back(
        self,
        loader: TextLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An exception in detection logs a warning and falls back."""

        def boom(data: bytes) -> Any:
            raise RuntimeError("detector exploded")

        monkeypatch.setattr("charset_normalizer.from_bytes", boom)
        path = tmp_path / "fallback.txt"
        path.write_bytes(b"plain ascii")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.document.text == "plain ascii"
        assert any("encoding detection failed" in r.message.lower() for r in caplog.records)

    def test_chardet_fallback_success(
        self,
        loader: TextLoader,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """chardet is used when charset-normalizer finds no match."""

        class NoMatches:
            def best(self) -> None:
                return None

        fake_chardet = types.ModuleType("chardet")
        fake_chardet.detect = lambda data: {  # type: ignore[attr-defined]
            "encoding": "cp1252",
            "confidence": 0.9,
        }
        monkeypatch.setattr("charset_normalizer.from_bytes", lambda data: NoMatches())
        monkeypatch.setitem(sys.modules, "chardet", fake_chardet)

        path = tmp_path / "chardet.txt"
        content = "Café déjà vu\n"
        path.write_bytes(content.encode("cp1252"))

        result = loader.load(path)
        assert result.document.text == content
        assert result.document.metadata["detection_method"] == "chardet"

    def test_chardet_exception_falls_back(
        self,
        loader: TextLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chardet failure logs a warning and falls back to UTF-8."""

        class NoMatches:
            def best(self) -> None:
                return None

        fake_chardet = types.ModuleType("chardet")

        def boom(data: bytes) -> Any:
            raise RuntimeError("chardet exploded")

        fake_chardet.detect = boom  # type: ignore[attr-defined]
        monkeypatch.setattr("charset_normalizer.from_bytes", lambda data: NoMatches())
        monkeypatch.setitem(sys.modules, "chardet", fake_chardet)

        path = tmp_path / "chardet_fail.txt"
        path.write_bytes(b"plain ascii")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert result.document.text == "plain ascii"
        assert any("chardet detection failed" in r.message.lower() for r in caplog.records)

    def test_bad_detected_encoding_falls_back(
        self,
        loader: TextLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown detected codec falls back to UTF-8 with replacements."""

        class BadMatch:
            encoding = "definitely-not-a-codec"
            confidence = 1.0
            percent_chaos = 0.0

        class Matches:
            def best(self) -> BadMatch:
                return BadMatch()

        monkeypatch.setattr("charset_normalizer.from_bytes", lambda data: Matches())
        path = tmp_path / "bad_codec.txt"
        path.write_bytes(b"plain \xff text")

        with caplog.at_level(logging.WARNING):
            result = loader.load(path)

        assert "\ufffd" in result.document.text
        assert result.document.metadata["encoding"] == "utf-8"
        assert result.document.metadata["detection_method"] == "utf-8-fallback"
        assert any("could not decode" in r.message.lower() for r in caplog.records)

    def test_extract_metadata_wrong_extension(
        self, loader: TextLoader, tmp_path: Path
    ) -> None:
        """The metadata-only method applies the extension guard."""
        path = tmp_path / "data.csv"
        path.write_bytes(b"a,b")

        with pytest.raises(LoadError, match="(?i)unsupported"):
            loader.extract_metadata(path)

    def test_extract_metadata_unreadable_path(
        self, loader: TextLoader, tmp_path: Path
    ) -> None:
        """An unreadable path raises LoadError from extract_metadata."""
        directory = tmp_path / "dir.txt"
        directory.mkdir()

        with pytest.raises(LoadError, match="(?i)failed to extract metadata"):
            loader.extract_metadata(directory)

    def test_extract_metadata_binary_rejected(
        self, loader: TextLoader, tmp_path: Path
    ) -> None:
        """A LoadError from decoding passes through extract_metadata."""
        path = tmp_path / "binary.txt"
        path.write_bytes(b"abc\x00def")

        with pytest.raises(LoadError, match="binary"):
            loader.extract_metadata(path)


class TestLoaderInterface:
    """Cross-loader checks that all three satisfy DocumentLoader."""

    @pytest.mark.parametrize(
        ("loader_cls", "filename", "setup_file"),
        [
            (
                DocxLoader,
                "sample.docx",
                lambda path: docx.Document().save(path),
            ),
            (
                MarkdownLoader,
                "sample.md",
                lambda path: path.write_text("# Heading\n", encoding="utf-8"),
            ),
            (
                TextLoader,
                "sample.txt",
                lambda path: path.write_bytes(b"content\n"),
            ),
        ],
    )
    def test_interface(
        self,
        loader_cls: type[DocumentLoader],
        filename: str,
        setup_file: Any,
        tmp_path: Path,
    ) -> None:
        """Each loader subclasses DocumentLoader and returns a LoadResult."""
        path = tmp_path / filename
        setup_file(path)

        loader = loader_cls()
        assert isinstance(loader, DocumentLoader)
        assert loader.supported_extensions

        result = loader.load(path)
        assert isinstance(result, LoadResult)
        assert result.success is True
        assert result.document is not None
        assert result.document.pages == [result.document.text]
        assert result.document.metadata["page_count"] == 1
        assert result.document.metadata["source"] == str(path)
