"""
Unit tests for the document ingestion service (steps 2.5.1 / 2.5.2 equivalent).

Covers recursive discovery, extension filtering, aggregation, content-hash
deduplication, error policies and the service factory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.airag.ingestion.exceptions import IngestionError, LoadError
from src.airag.ingestion.models import ExtractedDocument, LoadResult
from src.airag.ingestion.service import DocumentIngestionService, create_ingestion_service

ALPHA = "alpha\n"
BODY = "# Heading\n\nbody text\n"
DELTA = "delta\n"


@pytest.fixture
def document_tree(tmp_path: Path) -> Path:
    """Small mixed tree: two top-level docs, a binary file and nested docs."""
    (tmp_path / "a.txt").write_text(ALPHA, encoding="utf-8")
    (tmp_path / "b.md").write_text(BODY, encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\x00")
    (tmp_path / "ignored.bin").write_bytes(b"not a document")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (tmp_path / "sub" / "c.txt").write_text(ALPHA, encoding="utf-8")
    (nested / "d.text").write_text(DELTA, encoding="utf-8")
    return tmp_path


class TestDiscovery:
    def test_recursive_discovers_all_supported_files(self, document_tree: Path) -> None:
        service = DocumentIngestionService(deduplicate=False)
        result = service.ingest_directory(document_tree)

        assert result.loaded == 4
        assert result.failed == 1  # bad.txt contains NUL bytes
        assert result.skipped == 0
        assert len(result.results) == 5
        assert {r.source_path.name for r in result.results} == {
            "a.txt", "b.md", "bad.txt", "c.txt", "d.text",
        }

    def test_non_recursive_only_top_level(self, document_tree: Path) -> None:
        service = DocumentIngestionService(recursive=False, deduplicate=False)
        result = service.ingest_directory(document_tree)

        assert result.loaded == 2
        assert result.failed == 1
        assert {r.source_path.name for r in result.results} == {
            "a.txt", "b.md", "bad.txt",
        }

    def test_extension_filter(self, document_tree: Path) -> None:
        service = DocumentIngestionService(extensions=(".md",))
        result = service.ingest_directory(document_tree)

        assert result.loaded == 1
        assert result.failed == 0
        assert result.results[0].document is not None
        assert result.results[0].document.metadata["format"] == "markdown"

    def test_non_directory_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("x", encoding="utf-8")
        with pytest.raises(IngestionError, match="not a directory"):
            DocumentIngestionService().ingest_directory(file_path)


class TestAggregation:
    def test_content_hash_and_timestamp_populated(self, document_tree: Path) -> None:
        result = DocumentIngestionService().ingest_files([document_tree / "a.txt"])

        doc = result.results[0].document
        assert isinstance(doc, ExtractedDocument)
        assert doc.content_hash == hashlib.sha256(ALPHA.encode("utf-8")).hexdigest()
        assert doc.ingested_at
        assert doc.pages == [ALPHA]

    def test_duplicate_content_is_skipped(self, document_tree: Path) -> None:
        service = DocumentIngestionService()
        files = [document_tree / "a.txt", document_tree / "sub" / "c.txt"]

        result = service.ingest_files(files)
        assert result.loaded == 1
        assert result.skipped == 1
        assert result.failed == 0

        service.reset_deduplication()
        again = service.ingest_files(files)
        assert again.loaded == 1
        assert again.skipped == 1

    def test_deduplication_can_be_disabled(self, document_tree: Path) -> None:
        service = DocumentIngestionService(deduplicate=False)
        files = [document_tree / "a.txt", document_tree / "sub" / "c.txt"]

        result = service.ingest_files(files)
        assert result.loaded == 2
        assert result.skipped == 0

    def test_started_and_completed_timestamps(self, document_tree: Path) -> None:
        result = DocumentIngestionService().ingest_files([document_tree / "a.txt"])
        assert result.started_at
        assert result.completed_at
        assert result.started_at <= result.completed_at


class TestErrorPolicies:
    def test_failed_loads_are_reported_when_skipping(self, document_tree: Path) -> None:
        result = DocumentIngestionService().ingest_files([document_tree / "bad.txt"])

        assert result.failed == 1
        assert result.results[0].success is False
        assert "load failed" in (result.results[0].error or "").lower()

    def test_unsupported_extension_is_reported(self, document_tree: Path) -> None:
        result = DocumentIngestionService().ingest_files([document_tree / "ignored.bin"])

        assert result.failed == 1
        assert "unsupported format" in (result.results[0].error or "").lower()

    def test_fail_fast_raises_load_error(self, document_tree: Path) -> None:
        service = DocumentIngestionService(skip_errors=False)
        with pytest.raises(LoadError):
            service.ingest_files([document_tree / "bad.txt"])

    def test_fail_fast_raises_unsupported_format(self, document_tree: Path) -> None:
        from src.airag.ingestion.exceptions import UnsupportedFormatError

        service = DocumentIngestionService(skip_errors=False)
        with pytest.raises(UnsupportedFormatError):
            service.ingest_files([document_tree / "ignored.bin"])


def test_service_factory() -> None:
    service = create_ingestion_service(
        extensions=(".txt",), recursive=False, skip_errors=False, deduplicate=False
    )
    assert isinstance(service, DocumentIngestionService)
    assert service.extensions == (".txt",)
    assert service.recursive is False
    assert service.skip_errors is False
    assert service.deduplicate is False


class TestLoaderFailureBranches:
    def test_loader_failure_result_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.airag.ingestion import service as service_module

        class BrokenLoader:
            def load(self, file_path: Path) -> LoadResult:
                return LoadResult(
                    success=False,
                    document=None,
                    error="no document",
                    source_path=file_path,
                )

        monkeypatch.setattr(service_module, "get_loader", lambda path: BrokenLoader())
        path = tmp_path / "file.txt"
        path.write_text("x", encoding="utf-8")

        result = service_module.DocumentIngestionService().ingest_files([path])
        assert result.failed == 1
        assert result.results[0].error == "no document"

        with pytest.raises(LoadError, match="no document"):
            service_module.DocumentIngestionService(skip_errors=False).ingest_files([path])

    def test_unexpected_loader_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.airag.ingestion import service as service_module

        def boom(path: Path) -> None:
            raise RuntimeError("exploded")

        monkeypatch.setattr(service_module, "get_loader", boom)
        path = tmp_path / "file.txt"
        path.write_text("x", encoding="utf-8")

        result = service_module.DocumentIngestionService().ingest_files([path])
        assert result.failed == 1
        assert "unexpected error" in (result.results[0].error or "").lower()

        with pytest.raises(IngestionError, match="exploded"):
            service_module.DocumentIngestionService(skip_errors=False).ingest_files([path])
