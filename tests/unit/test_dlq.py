"""
Unit tests for the dead letter queue (step 2.6.4).

Covers JSONL persistence, deduplication, review/retry API, pipeline wiring,
config/CLI flags and package exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.airag.ingestion import dlq as dlq_module
from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.cli import create_parser, main
from src.airag.ingestion.config import IngestionConfig
from src.airag.ingestion.dlq import DeadLetterQueue, DLQEntry
from src.airag.ingestion.exceptions import ValidationError
from src.airag.ingestion.metrics import MetricsRegistry
from src.airag.ingestion.models import ExtractedDocument, IngestionResult, LoadResult
from src.airag.ingestion.pipeline import IngestionPipeline, PipelineResult


def make_entry(source: str = "a.txt", error: str = "Load failed: bad") -> DLQEntry:
    return DLQEntry(source=source, error=error, content_hash="abc123", chunks=2)


class TestDeadLetterQueue:
    def test_default_path(self) -> None:
        assert DeadLetterQueue().path == Path(dlq_module.DEFAULT_DLQ_FILENAME)

    def test_append_and_entries(self, tmp_path: Path) -> None:
        queue = DeadLetterQueue(tmp_path / "dlq.jsonl")
        assert queue.append(make_entry()) is True

        entries = queue.entries()
        assert len(entries) == 1
        assert entries[0].source == "a.txt"
        assert entries[0].error_type == "Load failed"
        assert entries[0].content_hash == "abc123"
        assert entries[0].chunks == 2
        assert entries[0].timestamp
        assert json.loads((tmp_path / "dlq.jsonl").read_text(encoding="utf-8").strip())

    def test_identical_failure_is_deduplicated(self, tmp_path: Path) -> None:
        queue = DeadLetterQueue(tmp_path / "dlq.jsonl")
        entry = make_entry()

        assert queue.append(entry) is True
        assert queue.append(entry) is False
        assert len(queue.entries()) == 1

    def test_different_error_for_same_source_is_kept(self, tmp_path: Path) -> None:
        queue = DeadLetterQueue(tmp_path / "dlq.jsonl")
        queue.append(make_entry(error="Load failed: bad"))
        queue.append(make_entry(error="ValueError: boom"))
        assert len(queue.entries()) == 2

    def test_corrupt_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "dlq.jsonl"
        path.write_text(
            "not-json\n"
            "\n"
            + json.dumps({"source": "good.txt"}) + "\n"
            + json.dumps({"error": "no source"}) + "\n",
            encoding="utf-8",
        )
        entries = DeadLetterQueue(path).entries()
        assert len(entries) == 1
        assert entries[0].source == "good.txt"
        assert entries[0].error == ""

    def test_pending_sources_are_unique_in_order(self, tmp_path: Path) -> None:
        queue = DeadLetterQueue(tmp_path / "dlq.jsonl")
        for source in ("b.txt", "a.txt", "b.txt"):
            queue.append(make_entry(source=source))
        assert queue.pending_sources() == ["b.txt", "a.txt"]

    def test_remove_and_clear(self, tmp_path: Path) -> None:
        path = tmp_path / "dlq.jsonl"
        queue = DeadLetterQueue(path)
        queue.append(make_entry(source="a.txt"))
        queue.append(make_entry(source="b.txt"))

        assert queue.remove("a.txt") == 1
        assert [entry.source for entry in queue.entries()] == ["b.txt"]
        assert queue.remove("missing.txt") == 0

        queue.clear()
        assert not path.exists()
        assert queue.entries() == []

    def test_package_reexports(self) -> None:
        import src.airag.ingestion as ingestion

        assert ingestion.DeadLetterQueue is DeadLetterQueue
        assert ingestion.DLQEntry is DLQEntry


class TestPipelineDLQ:
    def _pipeline(
        self,
        dlq: DeadLetterQueue,
        *,
        loads_ok: bool = True,
        chunker_fails: bool = False,
    ) -> IngestionPipeline:
        document = ExtractedDocument(text="hello", metadata={"source": "a.txt"})
        load = LoadResult(
            success=loads_ok,
            document=document if loads_ok else None,
            error=None if loads_ok else "Load failed: bad",
            source_path=Path("a.txt"),
        )

        class FakeService:
            def ingest_files(self, paths: list[str | Path]) -> IngestionResult:
                return IngestionResult(1, 0, 0, [load], "", "")

            def ingest_directory(self, path: str | Path) -> IngestionResult:
                return self.ingest_files([path])

        class FakeChunker:
            def chunk(self, document: ExtractedDocument) -> list[Chunk]:
                if chunker_fails:
                    raise ValueError("boom")
                return [Chunk(text=document.text)]

        class FakeEmbeddingPipeline:
            metrics = MetricsRegistry()

            def run(self, chunks: list[Chunk]) -> PipelineResult:
                return PipelineResult(processed=len(chunks), written=len(chunks))

        return IngestionPipeline(
            FakeChunker(), FakeEmbeddingPipeline(), FakeService(), dlq=dlq
        )

    def test_load_failure_is_recorded(self, tmp_path: Path) -> None:
        dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
        result = self._pipeline(dlq, loads_ok=False).ingest("a.txt")

        assert result.failed == 1
        entries = dlq.entries()
        assert len(entries) == 1
        assert entries[0].source == "a.txt"
        assert entries[0].error_type == "Load failed"
        assert entries[0].content_hash == ""

    def test_chunk_failure_is_recorded_with_hash(self, tmp_path: Path) -> None:
        dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
        result = self._pipeline(dlq, chunker_fails=True).ingest("a.txt")

        assert result.failed == 1
        entry = dlq.entries()[0]
        assert entry.error == "ValueError: boom"
        assert entry.error_type == "ValueError"
        assert entry.content_hash

    def test_successful_run_records_nothing(self, tmp_path: Path) -> None:
        dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
        result = self._pipeline(dlq).ingest("a.txt")

        assert result.succeeded == 1
        assert dlq.entries() == []

    def test_retry_clears_successful_source(self, tmp_path: Path) -> None:
        dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
        dlq.append(make_entry())
        pipeline = self._pipeline(dlq)

        result = pipeline.retry_dlq()
        assert result.succeeded == 1
        assert dlq.entries() == []

    def test_failed_retry_is_deduplicated(self, tmp_path: Path) -> None:
        dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
        dlq.append(DLQEntry(source="a.txt", error="Load failed: bad"))
        pipeline = self._pipeline(dlq, loads_ok=False)

        result = pipeline.retry_dlq()
        assert result.failed == 1
        assert len(dlq.entries()) == 1

    def test_retry_without_dlq_is_empty(self) -> None:
        pipeline = IngestionPipeline(object(), object(), object())
        result = pipeline.retry_dlq()
        assert result.files == 0
        assert result.succeeded == 0


class TestConfigAndCli:
    def test_config_accepts_dlq_path(self) -> None:
        config = IngestionConfig(dlq_path="queue.jsonl")
        assert config.to_dict()["dlq_path"] == "queue.jsonl"
        assert IngestionConfig().dlq_path is None
        with pytest.raises(ValidationError, match="dlq_path"):
            IngestionConfig(dlq_path=123)  # type: ignore[arg-type]

    def test_parser_flags(self) -> None:
        args = create_parser().parse_args(
            ["ingest", "--dlq", "queue.jsonl", "--retry-dlq"]
        )
        assert args.dlq == Path("queue.jsonl")
        assert args.retry_dlq is True
        assert args.path is None

    def test_cli_writes_dlq_for_unsupported_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad_file = tmp_path / "unknown.bin"
        bad_file.write_bytes(b"binary")
        dlq_path = tmp_path / "dlq.jsonl"

        code = main(["ingest", str(bad_file), "--dlq", str(dlq_path), "-v"])
        assert code == 1

        entries = dlq_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(entries) == 1
        assert json.loads(entries[0])["source"] == str(bad_file)

        payload = json.loads(capsys.readouterr().out)
        assert payload["dlq"]["pending"] == 1
        assert dlq_module.DEFAULT_DLQ_FILENAME == ".airag_dlq.jsonl"

    def test_cli_retry_requires_dlq(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["ingest", "--retry-dlq"]) == 1
        assert "requires --dlq" in capsys.readouterr().err
