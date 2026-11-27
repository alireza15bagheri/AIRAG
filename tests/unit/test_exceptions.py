"""
Unit tests for the finalized exception hierarchy (step 2.6.1).

Verifies inheritance, LoadError attributes, typed callers and package exports.
Existing loader/chunker/embedding tests continue to assert their original
exception types and are intentionally left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.airag.ingestion import exceptions as exc
from src.airag.ingestion.factory import get_loader
from src.airag.ingestion.service import DocumentIngestionService
from src.airag.ingestion.vector_store import VectorStoreWriter


@pytest.mark.parametrize(
    "name",
    [
        "LoadError",
        "UnsupportedFormatError",
        "ChunkError",
        "EmbeddingError",
        "ValidationError",
        "VectorStoreError",
        "PipelineError",
    ],
)
def test_all_domain_errors_share_base(name: str) -> None:
    assert issubclass(getattr(exc, name), exc.IngestionError)


def test_unsupported_format_remains_load_error() -> None:
    assert issubclass(exc.UnsupportedFormatError, exc.LoadError)
    assert not issubclass(exc.UnsupportedFormatError, exc.VectorStoreError)


def test_new_errors_are_not_load_errors() -> None:
    assert not issubclass(exc.VectorStoreError, exc.LoadError)
    assert not issubclass(exc.PipelineError, exc.LoadError)


def test_load_error_attributes() -> None:
    cause = RuntimeError("original")
    error = exc.LoadError("Could not read", source_path="doc.pdf", cause=cause)

    assert str(error) == "Could not read (source: doc.pdf)"
    assert error.source_path == "doc.pdf"
    assert error.cause is cause
    assert isinstance(error, exc.IngestionError)


def test_load_error_without_source() -> None:
    error = exc.LoadError("Could not read")
    assert str(error) == "Could not read"
    assert error.source_path is None
    assert error.cause is None


def test_all_exports() -> None:
    for name in (
        "IngestionError",
        "LoadError",
        "UnsupportedFormatError",
        "ChunkError",
        "EmbeddingError",
        "ValidationError",
        "VectorStoreError",
        "PipelineError",
    ):
        assert name in exc.__all__


def test_vector_store_uses_typed_error() -> None:
    with pytest.raises(exc.VectorStoreError, match="batch_size must be greater"):
        VectorStoreWriter(batch_size=0)


def test_service_uses_typed_error(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(exc.PipelineError, match="not a directory"):
        DocumentIngestionService().ingest_directory(file_path)


def test_factory_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(exc.UnsupportedFormatError):
        get_loader(tmp_path / "unknown.bin")


def test_package_reexports() -> None:
    import src.airag.ingestion as ingestion

    assert ingestion.VectorStoreError is exc.VectorStoreError
    assert ingestion.PipelineError is exc.PipelineError
