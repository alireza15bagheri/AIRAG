"""
Unit tests for in-process ingestion metrics (step 2.6.3).

Covers counters, histograms, timers, snapshot/reset, thread safety and
instrumentation of the two pipeline stages plus CLI output.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from src.airag.ingestion import metrics as metric_module
from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.cli import main
from src.airag.ingestion.metrics import (
    CHUNK_SECONDS,
    CHUNKS_CREATED,
    CHUNKS_SKIPPED,
    CHUNKS_UPDATED,
    CHUNKS_WRITTEN,
    EMBED_SECONDS,
    FILES_FAILED,
    FILES_PROCESSED,
    FILES_SUCCEEDED,
    LOAD_SECONDS,
    STORE_SECONDS,
    HistogramStats,
    MetricsRegistry,
)
from src.airag.ingestion.models import ExtractedDocument, IngestionResult, LoadResult
from src.airag.ingestion.pipeline import (
    EmbeddingPipeline,
    IngestionPipeline,
    PipelineResult,
)
from src.airag.ingestion.vector_store import WriteResult


class TestMetricsRegistry:
    def test_counters_start_zero_and_increment(self) -> None:
        registry = MetricsRegistry()
        assert registry.get_counter(FILES_PROCESSED) == 0

        registry.increment(FILES_PROCESSED)
        registry.increment(FILES_PROCESSED, 2)
        assert registry.get_counter(FILES_PROCESSED) == 3

    def test_negative_increment_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MetricsRegistry().increment(FILES_PROCESSED, -1)

    def test_observe_validates_latency(self) -> None:
        registry = MetricsRegistry()
        for invalid in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite non-negative"):
                registry.observe(LOAD_SECONDS, invalid)

    def test_histogram_stats_and_percentiles(self) -> None:
        registry = MetricsRegistry()
        for value in (0.4, 0.1, 0.3, 0.2):
            registry.observe(LOAD_SECONDS, value)

        stats = registry.histogram(LOAD_SECONDS)
        assert isinstance(stats, HistogramStats)
        assert stats.count == 4
        assert stats.total == pytest.approx(1.0)
        assert stats.minimum == pytest.approx(0.1)
        assert stats.maximum == pytest.approx(0.4)
        assert stats.average == pytest.approx(0.25)
        assert stats.p50 == pytest.approx(0.2)
        assert stats.p95 == pytest.approx(0.4)

    def test_empty_histogram_is_zeroed(self) -> None:
        stats = MetricsRegistry().histogram(LOAD_SECONDS)
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.p95 == 0.0

    def test_timer_records_even_when_block_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([1.0, 2.5])
        monkeypatch.setattr(metric_module.time, "perf_counter", lambda: next(clock))
        registry = MetricsRegistry()

        with pytest.raises(RuntimeError, match="boom"):
            with registry.timer(CHUNK_SECONDS):
                raise RuntimeError("boom")

        assert registry.histogram(CHUNK_SECONDS).total == pytest.approx(1.5)

    def test_snapshot_and_reset(self) -> None:
        registry = MetricsRegistry()
        registry.increment(FILES_PROCESSED)
        registry.increment("custom_total", 5)
        registry.observe(LOAD_SECONDS, 0.25)

        snapshot = registry.snapshot()
        assert snapshot["counters"][FILES_PROCESSED] == 1
        assert snapshot["counters"]["custom_total"] == 5
        assert snapshot["histograms"][LOAD_SECONDS]["count"] == 1
        assert set(snapshot["histograms"]) >= {
            LOAD_SECONDS,
            CHUNK_SECONDS,
            EMBED_SECONDS,
            STORE_SECONDS,
        }
        json.dumps(snapshot)

        registry.reset()
        assert registry.get_counter(FILES_PROCESSED) == 0
        assert registry.get_counter("custom_total") == 0
        assert registry.histogram(LOAD_SECONDS).count == 0

    def test_thread_safe_increments(self) -> None:
        registry = MetricsRegistry()

        def increment_many() -> None:
            for _ in range(100):
                registry.increment(FILES_PROCESSED)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: increment_many(), range(4)))
        assert registry.get_counter(FILES_PROCESSED) == 400

    def test_package_reexports(self) -> None:
        import src.airag.ingestion as ingestion

        assert ingestion.MetricsRegistry is MetricsRegistry
        assert ingestion.HistogramStats is HistogramStats


class TestPipelineInstrumentation:
    def test_embedding_pipeline_records_stage_metrics(self) -> None:
        chunks = [Chunk(text="one"), Chunk(text="two")]
        metrics = MetricsRegistry()

        class FakeEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

        class FakeWriter:
            def document_id(self, chunk: Chunk) -> str:
                return "doc"

            def chunk_id(self, chunk: Chunk) -> str:
                return f"doc:{chunk.chunk_index}"

            def write(self, chunks: list[Chunk], vectors: Any) -> WriteResult:
                return WriteResult(written=len(chunks))

        EmbeddingPipeline(FakeEmbedder(), FakeWriter(), metrics).run(chunks)

        assert metrics.get_counter(CHUNKS_CREATED) == 2
        assert metrics.get_counter(CHUNKS_WRITTEN) == 2
        assert metrics.get_counter(CHUNKS_UPDATED) == 0
        assert metrics.get_counter(CHUNKS_SKIPPED) == 0
        assert metrics.histogram(EMBED_SECONDS).count == 1
        assert metrics.histogram(STORE_SECONDS).count == 1

    def test_ingestion_pipeline_records_file_and_stage_metrics(self) -> None:
        document = ExtractedDocument(text="hello", metadata={"source": "a.txt"})
        load = LoadResult(
            success=True, document=document, error=None, source_path=Path("a.txt")
        )
        metrics = MetricsRegistry()

        class FakeService:
            def ingest_files(self, paths: list[str | Path]) -> IngestionResult:
                return IngestionResult(1, 0, 0, [load], "", "")

            def ingest_directory(self, path: str | Path) -> IngestionResult:
                return self.ingest_files([path])

        class FakeChunker:
            def chunk(self, document: ExtractedDocument) -> list[Chunk]:
                return [Chunk(text=document.text)]

        class FakeEmbeddingPipeline:
            def run(self, chunks: list[Chunk]) -> PipelineResult:
                return PipelineResult(processed=1, written=1)

        embedding_pipeline = FakeEmbeddingPipeline()
        embedding_pipeline.metrics = metrics

        IngestionPipeline(
            FakeChunker(), embedding_pipeline, FakeService(), metrics=metrics
        ).ingest("a.txt")

        assert metrics.get_counter(FILES_PROCESSED) == 1
        assert metrics.get_counter(FILES_SUCCEEDED) == 1
        assert metrics.get_counter(FILES_FAILED) == 0
        assert metrics.histogram(LOAD_SECONDS).count == 1
        assert metrics.histogram(CHUNK_SECONDS).count == 1

    def test_ingestion_pipeline_shares_embedding_registry(self) -> None:
        metrics = MetricsRegistry()

        class FakeEmbeddingPipeline:
            pass

        embedding_pipeline = FakeEmbeddingPipeline()
        embedding_pipeline.metrics = metrics

        pipeline = IngestionPipeline(
            object(), embedding_pipeline, object(), metrics=None
        )
        assert pipeline.metrics is metrics


class TestCliMetrics:
    def test_verbose_payload_and_human_output_include_metrics(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "note.txt").write_text("hello world", encoding="utf-8")

        class FakeChunker:
            def chunk(self, document: ExtractedDocument) -> list[Chunk]:
                return [Chunk(text=document.text)]

        class FakeEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

        class FakeWriter:
            def document_id(self, chunk: Chunk) -> str:
                return "doc"

            def chunk_id(self, chunk: Chunk) -> str:
                return f"doc:{chunk.chunk_index}"

            def write(self, chunks: list[Chunk], vectors: Any) -> WriteResult:
                return WriteResult(written=len(chunks))

        registry = MetricsRegistry()

        def fake_components(config: Any) -> tuple[Any, Any]:
            return (
                FakeChunker(),
                EmbeddingPipeline(FakeEmbedder(), FakeWriter(), registry),
            )

        monkeypatch.setattr("src.airag.ingestion.cli.create_components", fake_components)

        assert main(["ingest", str(tmp_path), "-v"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["metrics"]["counters"][FILES_PROCESSED] == 1
        assert payload["metrics"]["counters"][CHUNKS_CREATED] == 1

        assert main(["ingest", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "Metrics:" in out
