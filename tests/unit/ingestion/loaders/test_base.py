"""
Unit tests for the DocumentLoader base protocol and dataclasses.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)


class TestExtractedDocument:
    """Tests for ExtractedDocument dataclass."""

    def test_create_minimal(self) -> None:
        doc = ExtractedDocument(text="Hello", metadata={})
        assert doc.text == "Hello"
        assert doc.metadata == {}
        assert doc.pages == []

    def test_create_with_pages(self) -> None:
        doc = ExtractedDocument(
            text="Page 1\nPage 2",
            metadata={"pages": 2},
            pages=["Page 1", "Page 2"],
        )
        assert doc.pages == ["Page 1", "Page 2"]
        assert doc.metadata["pages"] == 2

    def test_frozen_immutable(self) -> None:
        doc = ExtractedDocument(text="Hello", metadata={})
        with pytest.raises(FrozenInstanceError):
            doc.text = "Modified"

    def test_slots_memory_efficiency(self) -> None:
        doc = ExtractedDocument(text="x" * 1000, metadata={})
        # Verify no __dict__ (slots optimization)
        assert not hasattr(doc, "__dict__")


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_create_success(self) -> None:
        doc = ExtractedDocument(text="Hello", metadata={})
        result = LoadResult(
            success=True,
            document=doc,
            error=None,
            source_path=Path("/test/file.txt"),
        )
        assert result.success is True
        assert result.document is doc
        assert result.error is None

    def test_create_failure(self) -> None:
        result = LoadResult(
            success=False,
            document=None,
            error="File not found",
            source_path=Path("/missing.txt"),
        )
        assert result.success is False
        assert result.document is None
        assert result.error == "File not found"


class ConcreteLoader(DocumentLoader):
    """Concrete implementation for testing."""
    SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})

    def load(self, file_path: Path) -> LoadResult:
        return LoadResult(
            success=True,
            document=ExtractedDocument(text="content", metadata={}),
            error=None,
            source_path=file_path,
        )


class TestDocumentLoader:
    """Tests for DocumentLoader abstract base class."""

    def test_instantiation(self) -> None:
        loader = ConcreteLoader()
        assert loader.supported_extensions == frozenset({".txt", ".md"})

    def test_instantiation_with_config(self) -> None:
        loader = ConcreteLoader(config={"option": "value"})
        assert loader.get_config("option") == "value"
        assert loader.get_config("missing", "default") == "default"

    def test_supported_extensions_property(self) -> None:
        loader = ConcreteLoader()
        assert loader.supported_extensions == frozenset({".txt", ".md"})
        # Case insensitive - use the case-insensitive property
        assert ".TXT" in loader.supported_extensions_case_insensitive
        assert ".MD" in loader.supported_extensions_case_insensitive

    def test_validate_file_success(self, tmp_path: Path) -> None:
        loader = ConcreteLoader()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        is_valid, error = loader.validate_file(test_file)
        assert is_valid is True
        assert error is None

    def test_validate_file_not_exists(self) -> None:
        loader = ConcreteLoader()
        is_valid, error = loader.validate_file(Path("/nonexistent.txt"))
        assert is_valid is False
        assert "does not exist" in error

    def test_validate_file_is_directory(self, tmp_path: Path) -> None:
        loader = ConcreteLoader()
        is_valid, error = loader.validate_file(tmp_path)
        assert is_valid is False
        assert "not a file" in error

    def test_validate_file_unsupported_extension(self, tmp_path: Path) -> None:
        loader = ConcreteLoader()
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        is_valid, error = loader.validate_file(test_file)
        assert is_valid is False
        assert "Unsupported extension" in error
        assert ".xyz" in error

    def test_load_with_validation_success(self, tmp_path: Path) -> None:
        loader = ConcreteLoader()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = loader.load_with_validation(test_file)
        assert result.success is True
        assert result.document is not None

    def test_load_with_validation_failure(self) -> None:
        loader = ConcreteLoader()
        result = loader.load_with_validation(Path("/nonexistent.txt"))
        assert result.success is False
        assert result.document is None
        assert result.error is not None

    def test_abstract_load_method(self) -> None:
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            DocumentLoader()

    def test_repr(self) -> None:
        loader = ConcreteLoader()
        repr_str = repr(loader)
        assert "ConcreteLoader" in repr_str
        assert ".md" in repr_str
        assert ".txt" in repr_str


class TestDocumentLoaderSubclassing:
    """Tests ensuring proper subclassing behavior."""

    def test_must_define_supported_extensions(self) -> None:
        class BadLoader(DocumentLoader):
            def load(self, file_path):
                pass

        # Should be able to instantiate but extensions will be empty
        loader = BadLoader()
        assert loader.supported_extensions == frozenset()

    def test_load_method_signature(self) -> None:
        """Verify load method accepts Path and returns LoadResult."""
        import inspect

        sig = inspect.signature(ConcreteLoader.load)
        params = list(sig.parameters.keys())
        assert params == ["self", "file_path"]
        assert sig.parameters["file_path"].annotation == Path
        assert sig.return_annotation == LoadResult
