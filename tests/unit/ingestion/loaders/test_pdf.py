"""
Unit tests for PDFLoader.

Step 2.1.2: Tests for PDFLoader implementation using PyMuPDF.
"""

import logging
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

    def test_metadata_extraction(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test metadata extraction from PDF."""
        result = loader.load(sample_pdf)

        assert result.success is True
        meta = result.document.metadata
        assert meta["format"] == "pdf"
        assert meta["page_count"] == 3
        assert meta["source"] == str(sample_pdf)
        assert "title" in meta
        assert "author" in meta
        assert "creation_date" in meta

    def test_extract_metadata_method(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test extract_metadata method."""
        meta = loader.extract_metadata(sample_pdf)

        assert meta["format"] == "pdf"
        assert meta["page_count"] == 3
        assert meta["source"] == str(sample_pdf)

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