"""
Unit tests for PDFLoader.

Step 2.1.2: Tests for PDFLoader implementation using PyMuPDF.
Step 2.2.2: Tests for PDF metadata extraction (page count, title, author, creation date).
"""

import logging
import re
import pytest
from pathlib import Path

import fitz  # PyMuPDF

from src.airag.ingestion.loaders.pdf import PDFLoader
from src.airag.ingestion.exceptions import LoadError


class TestPDFLoader:
    """Tests for PDFLoader class."""

    @pytest.fixture
    def loader(self) -> PDFLoader:
        """Create a PDFLoader instance."""
        return PDFLoader()

    @pytest.fixture
    def sample_pdf(self, tmp_path: Path) -> Path:
        """Create a simple multi-page PDF for testing."""
        pdf_path = tmp_path / "sample.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} content\nLine 2\nLine 3")
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    @pytest.fixture
    def pdf_with_metadata(self, tmp_path: Path) -> Path:
        """Create a PDF with title, author, and creation date metadata."""
        pdf_path = tmp_path / "metadata.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Document with metadata")
        # Set metadata
        doc.set_metadata({
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "D:20240115143000+00'00'",
            "modDate": "D:20240115150000+00'00'",
        })
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    @pytest.fixture
    def encrypted_pdf(self, tmp_path: Path) -> Path:
        """Create an encrypted PDF for testing."""
        pdf_path = tmp_path / "encrypted.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        doc.close()
        return pdf_path

    @pytest.fixture
    def corrupted_pdf(self, tmp_path: Path) -> Path:
        """Create a corrupted/truncated PDF."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%truncated")
        return pdf_path

    @pytest.fixture
    def non_pdf_file(self, tmp_path: Path) -> Path:
        """Create a non-PDF file."""
        txt_path = tmp_path / "document.txt"
        txt_path.write_text("This is not a PDF")
        return txt_path

    def test_load_valid_pdf(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test loading a valid multi-page PDF."""
        result = loader.load(sample_pdf)

        assert result.success is True
        assert result.document is not None
        assert result.error is None
        assert result.document.pages is not None
        assert len(result.document.pages) == 3
        assert "Page 1 content" in result.document.pages[0]
        assert "Page 2 content" in result.document.pages[1]
        assert "Page 3 content" in result.document.pages[2]

        # Check combined text
        assert "Page 1 content" in result.document.text
        assert "Page 2 content" in result.document.text
        assert "Page 3 content" in result.document.text

        # Check metadata
        assert result.document.metadata["format"] == "pdf"
        assert result.document.metadata["page_count"] == 3
        assert result.document.metadata["source"] == str(sample_pdf)

    def test_load_encrypted_pdf_wrong_password(self, loader: PDFLoader, encrypted_pdf: Path) -> None:
        """Test loading encrypted PDF with wrong password raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(encrypted_pdf)

        assert "encrypted" in str(exc_info.value).lower()
        assert "password" in str(exc_info.value).lower()

    def test_load_encrypted_pdf_correct_password(self, tmp_path: Path) -> None:
        """Test loading encrypted PDF with correct password succeeds."""
        pdf_path = tmp_path / "encrypted2.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        doc.close()

        loader = PDFLoader(config={"password": "secret"})
        result = loader.load(pdf_path)

        assert result.success is True
        assert result.document is not None
        assert "Secret content" in result.document.text

    def test_load_scanned_pdf_no_text(self, loader: PDFLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test loading scanned/image-only PDF returns empty text with warning."""
        # Create a PDF with an image but no extractable text
        # Use a simple approach: create a page with a colored rectangle (no text)
        pdf_path = tmp_path / "scanned.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Draw a rectangle (no text) - this simulates a scanned page
        rect = fitz.Rect(0, 0, 612, 792)
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0))
        doc.save(pdf_path)
        doc.close()

        with caplog.at_level(logging.WARNING):
            result = loader.load(pdf_path)

        assert result.success is True
        assert result.document is not None
        # Should have empty or minimal text
        assert any("scanned" in record.message.lower() or "empty" in record.message.lower()
                   for record in caplog.records)

    def test_load_corrupted_pdf(self, loader: PDFLoader, corrupted_pdf: Path) -> None:
        """Test loading corrupted/truncated PDF raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(corrupted_pdf)

        assert "failed to load" in str(exc_info.value).lower() or "corrupt" in str(exc_info.value).lower()

    def test_load_non_pdf_extension(self, loader: PDFLoader, non_pdf_file: Path) -> None:
        """Test loading non-PDF file raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(non_pdf_file)

        assert "unsupported" in str(exc_info.value).lower()
        assert ".txt" in str(exc_info.value).lower()

    def test_load_nonexistent_file(self, loader: PDFLoader) -> None:
        """Test loading non-existent file raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(Path("/nonexistent/path.pdf"))

        assert "no such file" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_load_case_insensitive_extension(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test loading PDF with uppercase extension."""
        pdf_path = tmp_path / "TEST.PDF"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Uppercase extension test")
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        assert result.success is True
        assert "Uppercase extension test" in result.document.text

    def test_metadata_extraction(self, loader: PDFLoader, pdf_with_metadata: Path) -> None:
        """Test metadata extraction from PDF with full Info dict."""
        result = loader.load(pdf_with_metadata)

        assert result.success is True
        meta = result.document.metadata
        assert meta["format"] == "pdf"
        assert meta["page_count"] == 1
        assert meta["source"] == str(pdf_with_metadata)
        assert meta["title"] == "Test Document"
        assert meta["author"] == "Test Author"
        assert meta["creation_date"] is not None
        # Verify ISO-8601 format (time is 00:00:00 because PyMuPDF stores only date in this test)
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", meta["creation_date"])

    def test_extract_metadata_method(self, loader: PDFLoader, pdf_with_metadata: Path) -> None:
        """Test extract_metadata method returns correct fields."""
        meta = loader.extract_metadata(pdf_with_metadata)

        assert meta["format"] == "pdf"
        assert meta["page_count"] == 1
        assert meta["source"] == str(pdf_with_metadata)
        assert meta["title"] == "Test Document"
        assert meta["author"] == "Test Author"
        assert meta["creation_date"] is not None
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", meta["creation_date"])

    def test_extract_metadata_nonexistent(self, loader: PDFLoader) -> None:
        """Test extract_metadata on nonexistent file raises LoadError."""
        with pytest.raises(LoadError):
            loader.extract_metadata(Path("/nonexistent.pdf"))

    def test_cjk_text_extraction(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test Unicode text extraction (Latin extended)."""
        pdf_path = tmp_path / "unicode.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Use Latin extended characters that are more likely to render
        page.insert_text((72, 72), "Café naïve résumé\nМосква 東京")
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)

        assert result.success is True
        # Check that Unicode text was extracted (may not render perfectly but should be present)
        assert "Café" in result.document.text or "caf" in result.document.text.lower()

    def test_large_pdf_memory(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test loading a larger PDF (50 pages) doesn't exceed memory."""
        import psutil
        import os

        pdf_path = tmp_path / "large.pdf"
        doc = fitz.open()
        for i in range(50):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}\n" + "Content " * 100)
        doc.save(pdf_path)
        doc.close()

        # Track memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = loader.load(pdf_path)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before

        assert result.success is True
        assert result.document is not None
        assert len(result.document.pages) == 50
        assert mem_diff < 200  # Should not exceed 200MB increase

    def test_chained_exception_chaining(self, loader: PDFLoader, corrupted_pdf: Path) -> None:
        """Test that exceptions are properly chained."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(corrupted_pdf)

        assert exc_info.value.cause is not None

    def test_empty_password_on_encrypted(self, tmp_path: Path) -> None:
        """Test loading encrypted PDF with empty password fails."""
        pdf_path = tmp_path / "encrypted3.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Content")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="password", owner_pw="owner")
        doc.close()

        loader = PDFLoader(config={"password": ""})
        with pytest.raises(LoadError) as exc_info:
            loader.load(pdf_path)

        assert "password" in str(exc_info.value).lower()

    # ===== Step 2.2.2: Metadata-specific tests =====

    def test_metadata_full_info_dict(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test metadata extraction with complete Info dict."""
        pdf_path = tmp_path / "full_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Full metadata test")
        doc.set_metadata({
            "title": "Complete Document",
            "author": "John Doe",
            "creationDate": "D:20240615103000+00'00'",
        })
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["title"] == "Complete Document"
        assert meta["author"] == "John Doe"
        assert meta["creation_date"] == "2024-06-15T10:30:00+00:00"

    def test_metadata_missing_info_dict(self, tmp_path: Path, loader: PDFLoader, caplog: pytest.LogCaptureFixture) -> None:
        """Test metadata extraction when Info dict is missing entirely."""
        pdf_path = tmp_path / "no_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "No metadata")
        # Don't set any metadata
        doc.save(pdf_path)
        doc.close()

        with caplog.at_level(logging.WARNING):
            result = loader.load(pdf_path)

        meta = result.document.metadata
        assert meta["title"] is None
        assert meta["author"] is None
        assert meta["creation_date"] is None
        assert any("meaningful info dictionary metadata" in r.message.lower() for r in caplog.records)

    def test_metadata_partial_info_dict(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test metadata extraction with partial Info dict (title only)."""
        pdf_path = tmp_path / "partial_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Partial metadata")
        doc.set_metadata({"title": "Only Title"})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["title"] == "Only Title"
        assert meta["author"] is None
        assert meta["creation_date"] is None

    def test_metadata_empty_strings_normalized_to_none(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test that empty strings in title/author are normalized to None."""
        pdf_path = tmp_path / "empty_strings.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Empty strings test")
        # PyMuPDF returns empty strings for missing fields
        doc.set_metadata({"title": "", "author": ""})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["title"] is None
        assert meta["author"] is None

    def test_creation_date_utc_z_suffix(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test parsing creation date with 'Z' suffix (UTC)."""
        pdf_path = tmp_path / "utc_date.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "UTC date test")
        doc.set_metadata({"creationDate": "D:20240701120000Z"})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["creation_date"] == "2024-07-01T12:00:00+00:00"

    def test_creation_date_with_offset(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test parsing creation date with offset (+HH'MM')."""
        pdf_path = tmp_path / "offset_date.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Offset date test")
        # Format: D:YYYYMMDDHHmmSS+HH'MM'
        doc.set_metadata({"creationDate": "D:20240615143000+05'30'"})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        # Should parse offset +05:30
        assert meta["creation_date"] == "2024-06-15T14:30:00+05:30"

    def test_creation_date_negative_offset(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test parsing creation date with negative offset (-HH'MM')."""
        pdf_path = tmp_path / "neg_offset.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Negative offset test")
        doc.set_metadata({"creationDate": "D:20240615143000-05'00'"})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["creation_date"] == "2024-06-15T14:30:00-05:00"

    def test_malformed_creation_date_warns_and_none(self, tmp_path: Path, loader: PDFLoader, caplog: pytest.LogCaptureFixture) -> None:
        """Test malformed creation date logs warning and returns None."""
        pdf_path = tmp_path / "bad_date.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Bad date test")
        doc.set_metadata({"creationDate": "D:notadate"})
        doc.save(pdf_path)
        doc.close()

        with caplog.at_level(logging.WARNING):
            result = loader.load(pdf_path)

        meta = result.document.metadata
        assert meta["creation_date"] is None
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_empty_creation_date_returns_none(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test empty creation date string returns None."""
        pdf_path = tmp_path / "empty_date.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Empty date test")
        doc.set_metadata({"creationDate": ""})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["creation_date"] is None

    def test_creation_date_iso_without_d_prefix(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test creation date without 'D:' prefix (still parses)."""
        pdf_path = tmp_path / "no_d_prefix.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "No D prefix test")
        # Some PDFs may omit the D: prefix
        doc.set_metadata({"creationDate": "20240615143000Z"})
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        # Should still parse (with warning)
        assert meta["creation_date"] is not None

    def test_non_ascii_title_author(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test Unicode title and author round-trip."""
        pdf_path = tmp_path / "unicode_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Unicode test")
        doc.set_metadata({
            "title": "Café naïve résumé",
            "author": "Москва 東京",
        })
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        assert meta["title"] == "Café naïve résumé"
        assert meta["author"] == "Москва 東京"

    def test_encrypted_but_openable_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction from encrypted-but-openable PDF."""
        pdf_path = tmp_path / "encrypted_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Encrypted with metadata")
        doc.set_metadata({
            "title": "Secret Doc",
            "author": "Agent X",
            "creationDate": "D:20240101000000Z",
        })
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        doc.close()

        loader = PDFLoader(config={"password": "secret"})
        result = loader.load(pdf_path)

        assert result.success is True
        meta = result.document.metadata
        assert meta["title"] == "Secret Doc"
        assert meta["author"] == "Agent X"
        assert meta["creation_date"] is not None

    def test_scanned_pdf_metadata_preserved(self, tmp_path: Path, loader: PDFLoader, caplog: pytest.LogCaptureFixture) -> None:
        """Test scanned PDF (no text) still extracts metadata correctly."""
        pdf_path = tmp_path / "scanned_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Draw a rectangle (no text) - simulates scanned page
        rect = fitz.Rect(0, 0, 612, 792)
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0))
        doc.set_metadata({
            "title": "Scanned Document",
            "author": "Scanner",
            "creationDate": "D:20240101000000Z",
        })
        doc.save(pdf_path)
        doc.close()

        with caplog.at_level(logging.WARNING):
            result = loader.load(pdf_path)

        meta = result.document.metadata
        assert meta["title"] == "Scanned Document"
        assert meta["author"] == "Scanner"
        assert meta["creation_date"] is not None
        assert any("scanned" in r.message.lower() or "empty" in r.message.lower()
                   for r in caplog.records)

    def test_metadata_json_serializable(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test that metadata dict is JSON-serializable (2.6.2 / ChromaDB compatibility)."""
        import json
        pdf_path = tmp_path / "serializable.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Serializable test")
        doc.set_metadata({
            "title": "Test",
            "author": "Author",
            "creationDate": "D:20240101000000Z",
        })
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        # Should not raise
        json_str = json.dumps(meta)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["title"] == "Test"

    def test_metadata_keys_exact_set(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test metadata dict has exactly the 6 required keys, no more no less."""
        pdf_path = tmp_path / "keys_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Keys test")
        doc.set_metadata({
            "title": "Test",
            "author": "Author",
            "creationDate": "D:20240101000000Z",
            "subject": "Should not appear",  # Out of scope
            "creator": "Should not appear",   # Out of scope
            "producer": "Should not appear",  # Out of scope
            "modDate": "D:20240101000000Z",   # Out of scope
        })
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        meta = result.document.metadata

        expected_keys = {"source", "page_count", "format", "title", "author", "creation_date"}
        assert set(meta.keys()) == expected_keys

    def test_extract_metadata_matches_load_metadata(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test extract_metadata() returns same fields as load()."""
        pdf_path = tmp_path / "consistency.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Consistency test")
        doc.set_metadata({
            "title": "Consistent",
            "author": "Tester",
            "creationDate": "D:20240101000000Z",
        })
        doc.save(pdf_path)
        doc.close()

        # Load via load()
        load_result = loader.load(pdf_path)
        load_meta = load_result.document.metadata

        # Load via extract_metadata()
        extract_meta = loader.extract_metadata(pdf_path)

        # Should have same keys and values
        assert load_meta.keys() == extract_meta.keys()
        for key in load_meta:
            assert load_meta[key] == extract_meta[key]

    # Existing tests that must continue to pass
    def test_load_valid_pdf(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test loading a valid multi-page PDF."""
        result = loader.load(sample_pdf)

        assert result.success is True
        assert result.document is not None
        assert result.error is None
        assert result.document.pages is not None
        assert len(result.document.pages) == 3
        assert "Page 1 content" in result.document.pages[0]
        assert "Page 2 content" in result.document.pages[1]
        assert "Page 3 content" in result.document.pages[2]

        # Check combined text
        assert "Page 1 content" in result.document.text
        assert "Page 2 content" in result.document.text
        assert "Page 3 content" in result.document.text

        # Check metadata
        assert result.document.metadata["format"] == "pdf"
        assert result.document.metadata["page_count"] == 3
        assert result.document.metadata["source"] == str(sample_pdf)

    def test_load_encrypted_pdf_wrong_password(self, loader: PDFLoader, encrypted_pdf: Path) -> None:
        """Test loading encrypted PDF with wrong password raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(encrypted_pdf)

        assert "encrypted" in str(exc_info.value).lower()
        assert "password" in str(exc_info.value).lower()

    def test_load_encrypted_pdf_correct_password(self, tmp_path: Path) -> None:
        """Test loading encrypted PDF with correct password succeeds."""
        pdf_path = tmp_path / "encrypted2.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        doc.close()

        loader = PDFLoader(config={"password": "secret"})
        result = loader.load(pdf_path)

        assert result.success is True
        assert result.document is not None
        assert "Secret content" in result.document.text

    def test_load_scanned_pdf_no_text(self, loader: PDFLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test loading scanned/image-only PDF returns empty text with warning."""
        pdf_path = tmp_path / "scanned.pdf"
        doc = fitz.open()
        page = doc.new_page()
        rect = fitz.Rect(0, 0, 612, 792)
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0))
        doc.save(pdf_path)
        doc.close()

        with caplog.at_level(logging.WARNING):
            result = loader.load(pdf_path)

        assert result.success is True
        assert result.document is not None
        assert any("scanned" in record.message.lower() or "empty" in record.message.lower()
                   for record in caplog.records)

    def test_load_corrupted_pdf(self, loader: PDFLoader, corrupted_pdf: Path) -> None:
        """Test loading corrupted/truncated PDF raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(corrupted_pdf)

        assert "failed to load" in str(exc_info.value).lower() or "corrupt" in str(exc_info.value).lower()

    def test_load_non_pdf_extension(self, loader: PDFLoader, non_pdf_file: Path) -> None:
        """Test loading non-PDF file raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(non_pdf_file)

        assert "unsupported" in str(exc_info.value).lower()
        assert ".txt" in str(exc_info.value).lower()

    def test_load_nonexistent_file(self, loader: PDFLoader) -> None:
        """Test loading non-existent file raises LoadError."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(Path("/nonexistent/path.pdf"))

        assert "no such file" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_load_case_insensitive_extension(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test loading PDF with uppercase extension."""
        pdf_path = tmp_path / "TEST.PDF"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Uppercase extension test")
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)
        assert result.success is True
        assert "Uppercase extension test" in result.document.text

    def test_cjk_text_extraction(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test Unicode text extraction (Latin extended)."""
        pdf_path = tmp_path / "unicode.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Café naïve résumé\nМосква 東京")
        doc.save(pdf_path)
        doc.close()

        result = loader.load(pdf_path)

        assert result.success is True
        assert "Café" in result.document.text or "caf" in result.document.text.lower()

    def test_large_pdf_memory(self, tmp_path: Path, loader: PDFLoader) -> None:
        """Test loading a larger PDF (50 pages) doesn't exceed memory."""
        import psutil
        import os

        pdf_path = tmp_path / "large.pdf"
        doc = fitz.open()
        for i in range(50):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}\n" + "Content " * 100)
        doc.save(pdf_path)
        doc.close()

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = loader.load(pdf_path)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before

        assert result.success is True
        assert result.document is not None
        assert len(result.document.pages) == 50
        assert mem_diff < 200

    def test_chained_exception_chaining(self, loader: PDFLoader, corrupted_pdf: Path) -> None:
        """Test that exceptions are properly chained."""
        with pytest.raises(LoadError) as exc_info:
            loader.load(corrupted_pdf)

        assert exc_info.value.cause is not None

    def test_empty_password_on_encrypted(self, tmp_path: Path) -> None:
        """Test loading encrypted PDF with empty password fails."""
        pdf_path = tmp_path / "encrypted3.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Content")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="password", owner_pw="owner")
        doc.close()

        loader = PDFLoader(config={"password": ""})
        with pytest.raises(LoadError) as exc_info:
            loader.load(pdf_path)

        assert "password" in str(exc_info.value).lower()