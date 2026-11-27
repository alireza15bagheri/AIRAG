"""
Unit tests for the chunk → embedding → ID pipeline (step 2.4.3).

Covers grouping by document ID, delegation to embedder/writer, aggregation,
error propagation and an end-to-end idempotent run with real 2.4 components.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.airag.ingestion.chunking import Chunk, ChunkConfig, SemanticChunker
from src.airag.ingestion.chunking.fixed import FixedSizeChunker
from src.airag.ingestion.embedding import EmbeddingService
from src.airag.ingestion.exceptions import EmbeddingError, IngestionError
from src.airag.ingestion.models import (
    ExtractedDocument,
    IngestionResult,
    LoadResult,
)
from src.airag.ingestion.pipeline import (
    CheckpointManager,
    EmbeddingPipeline,
    FileResult,
    IngestionPipeline,
    NullProgress,
    PipelineResult,
    PipelineRunResult,
    TqdmProgress,
)
from src.airag.ingestion.service import DocumentIngestionService
from src.airag.ingestion.vector_store import VectorStoreWriter, WriteResult


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("embed boom")
        return [[1.0, 0.0] for _ in texts]


class FakeWriter:
    def __init__(
        self, fail: bool = False, result: WriteResult | None = None
    ) -> None:
        self.calls: list[tuple[list[Chunk], list[list[float]]]] = []
        self.fail = fail
        self.result = result

    @staticmethod
    def document_id(chunk: Chunk) -> str:
        return VectorStoreWriter.document_id(chunk)

    @staticmethod
    def chunk_id(chunk: Chunk) -> str:
        return VectorStoreWriter.chunk_id(chunk)

    def write(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> WriteResult:
        self.calls.append((list(chunks), [list(vector) for vector in embeddings]))
        if self.fail:
            raise IngestionError("write boom")
        if self.result is not None:
            return self.result
        return WriteResult(
            written=len(chunks), ids=[self.chunk_id(chunk) for chunk in chunks]
        )


def make_chunk(
    text: str, index: int = 0, document_hash: str = "doc-1"
) -> Chunk:
    return Chunk(
        text=text,
        metadata={"source": "doc.txt", "hash": document_hash},
        chunk_index=index,
        start_char=0,
        end_char=len(text),
    )


class MemoryCollection:
    """Small ChromaDB-compatible collection for the end-to-end test."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def get(self, ids: list[str], include: list[str] | None = None) -> dict[str, Any]:
        found = [chunk_id for chunk_id in ids if chunk_id in self.items]
        return {
            "ids": found,
            "metadatas": [self.items[chunk_id]["metadatas"] for chunk_id in found],
        }

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for chunk_id, _vector, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.items[chunk_id] = {"metadatas": dict(metadata), "documents": document}

    def count(self) -> int:
        return len(self.items)


class TestPipelineResult:
    def test_defaults(self) -> None:
        a = PipelineResult()
        b = PipelineResult()
        assert a == b
        a.document_ids.append("doc-1")
        assert b.document_ids == []


class TestEmbeddingPipeline:
    def test_empty_run(self) -> None:
        embedder = FakeEmbedder()
        writer = FakeWriter()

        result = EmbeddingPipeline(embedder, writer).run([])

        assert result == PipelineResult()
        assert embedder.calls == []
        assert writer.calls == []

    def test_single_document_grouped(self) -> None:
        embedder = FakeEmbedder()
        writer = FakeWriter()
        chunks = [make_chunk("a", 0), make_chunk("bb", 1), make_chunk("ccc", 2)]

        result = EmbeddingPipeline(embedder, writer).run(chunks)

        assert embedder.calls == [["a", "bb", "ccc"]]
        assert len(writer.calls) == 1
        assert result.processed == 3
        assert result.written == 3
        assert result.document_ids == ["doc-1"]
        assert result.chunk_ids == ["doc-1:0", "doc-1:1", "doc-1:2"]

    def test_multiple_documents_grouped_in_first_seen_order(self) -> None:
        embedder = FakeEmbedder()
        writer = FakeWriter()
        chunks = [
            make_chunk("a", 0, "doc-1"),
            make_chunk("b", 0, "doc-2"),
            make_chunk("c", 1, "doc-1"),
        ]

        result = EmbeddingPipeline(embedder, writer).run(chunks)

        assert embedder.calls == [["a", "c"], ["b"]]
        assert [call[0][0].text for call in writer.calls] == ["a", "b"]
        assert result.document_ids == ["doc-1", "doc-2"]
        assert result.chunk_ids == ["doc-1:0", "doc-1:1", "doc-2:0"]

    def test_document_id_falls_back_to_chunk_hash(self) -> None:
        chunk = Chunk(text="orphan", metadata={})
        expected = hashlib.sha256(b"orphan").hexdigest()

        result = EmbeddingPipeline(FakeEmbedder(), FakeWriter()).run([chunk])

        assert result.document_ids == [expected]
        assert result.chunk_ids == [f"{expected}:0"]

    def test_invalid_chunk_type(self) -> None:
        with pytest.raises(TypeError, match="Expected Chunk instances"):
            EmbeddingPipeline(FakeEmbedder(), FakeWriter()).run(["not-a-chunk"])

    def test_embedding_failure_propagates(self) -> None:
        embedder = FakeEmbedder(fail=True)
        writer = FakeWriter()

        with pytest.raises(EmbeddingError, match="embed boom"):
            EmbeddingPipeline(embedder, writer).run([make_chunk("a")])
        assert writer.calls == []

    def test_writer_failure_propagates(self) -> None:
        with pytest.raises(IngestionError, match="write boom"):
            EmbeddingPipeline(FakeEmbedder(), FakeWriter(fail=True)).run(
                [make_chunk("a")]
            )

    def test_aggregates_write_result(self) -> None:
        writer = FakeWriter(
            result=WriteResult(written=0, updated=1, skipped=2, ids=["doc-1:0"])
        )

        result = EmbeddingPipeline(FakeEmbedder(), writer).run(
            [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        )

        assert result.processed == 3
        assert result.updated == 1
        assert result.skipped == 2
        assert result.written == 0


class TestIntegration:
    def test_chunk_embed_store_twice_is_idempotent(self) -> None:
        def topic_encoder(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts]

        text = "Alpha one. Alpha two. Beta three. Beta four."
        document = ExtractedDocument(text=text, metadata={"source": "doc.txt"}, pages=[text])
        chunks = SemanticChunker(ChunkConfig(chunk_size=100), embedder=topic_encoder).chunk(
            document
        )
        pipeline = EmbeddingPipeline(
            EmbeddingService(encoder=topic_encoder),
            VectorStoreWriter(collection=MemoryCollection()),
        )

        first = pipeline.run(chunks)
        second = pipeline.run(chunks)

        assert first.written == 2
        assert second.skipped == 2
        assert pipeline.writer.count() == 2

    def test_package_reexport(self) -> None:
        from src.airag.ingestion import EmbeddingPipeline as Exported

        assert Exported is EmbeddingPipeline


class FakeService:
    def __init__(self, results: list[LoadResult]) -> None:
        self.results = results
        self.dir_calls: list[Path] = []
        self.file_calls: list[list[Path]] = []

    def _result(self) -> IngestionResult:
        return IngestionResult(
            loaded=len(self.results),
            failed=0,
            skipped=0,
            results=self.results,
            started_at="start",
            completed_at="end",
        )

    def ingest_directory(self, path: Path) -> IngestionResult:
        self.dir_calls.append(path)
        return self._result()

    def ingest_files(self, paths: list[Path]) -> IngestionResult:
        self.file_calls.append(paths)
        return self._result()


class FakeChunker:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[ExtractedDocument] = []
        self.fail_for = fail_for or set()

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        self.calls.append(document)
        if document.metadata.get("source") in self.fail_for:
            raise RuntimeError("chunk boom")
        return [
            Chunk(
                text=document.text,
                metadata=dict(document.metadata),
                chunk_index=0,
                end_char=len(document.text),
            )
        ]


class FakeEmbeddingPipeline:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[Chunk]] = []
        self.fail = fail

    def run(self, chunks: list[Chunk]) -> PipelineResult:
        self.calls.append(list(chunks))
        if self.fail:
            raise RuntimeError("embed pipeline boom")
        return PipelineResult(
            processed=len(chunks), written=len(chunks), chunk_ids=["doc:0"]
        )


def successful_load(path: Path, text: str = "hello") -> LoadResult:
    return LoadResult(
        success=True,
        document=ExtractedDocument(
            text=text, metadata={"source": str(path)}, pages=[text]
        ),
        error=None,
        source_path=path,
    )


class TestRunResultDefaults:
    def test_defaults_are_independent(self) -> None:
        a = PipelineRunResult()
        b = PipelineRunResult()
        a.file_results.append(FileResult(source="x", success=True))
        assert b.file_results == []
        assert a.files == 0


class TestIngestionPipeline:
    def test_file_path_uses_ingest_files(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.txt"
        file_path.write_text("hello", encoding="utf-8")
        service = FakeService([successful_load(file_path)])
        pipeline = IngestionPipeline(FakeChunker(), FakeEmbeddingPipeline(), service)

        result = pipeline.ingest(file_path)

        assert service.file_calls == [[file_path]]
        assert service.dir_calls == []
        assert result.files == 1

    def test_directory_path_uses_ingest_directory(self, tmp_path: Path) -> None:
        service = FakeService([])
        pipeline = IngestionPipeline(FakeChunker(), FakeEmbeddingPipeline(), service)

        pipeline.ingest(tmp_path)

        assert service.dir_calls == [tmp_path]
        assert service.file_calls == []

    def test_aggregates_successes(self, tmp_path: Path) -> None:
        first = successful_load(tmp_path / "a.txt", "alpha")
        second = successful_load(tmp_path / "b.txt", "beta")
        chunker = FakeChunker()
        embedder = FakeEmbeddingPipeline()
        pipeline = IngestionPipeline(chunker, embedder, FakeService([first, second]))

        result = pipeline.ingest(tmp_path / "a.txt")

        assert result.files == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert result.chunks == 2
        assert result.written == 2
        assert len(chunker.calls) == 2
        assert len(embedder.calls) == 2
        assert all(item.success for item in result.file_results)

    def test_failed_load_is_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        failed = LoadResult(
            success=False, document=None, error="load failed", source_path=path
        )
        chunker = FakeChunker()
        pipeline = IngestionPipeline(chunker, FakeEmbeddingPipeline(), FakeService([failed]))

        result = pipeline.ingest(path)

        assert result.failed == 1
        assert result.succeeded == 0
        assert result.file_results[0].error == "load failed"
        assert chunker.calls == []

    def test_chunking_failure_is_isolated(self, tmp_path: Path) -> None:
        bad = successful_load(tmp_path / "bad.txt")
        good = successful_load(tmp_path / "good.txt")
        chunker = FakeChunker(fail_for={str(tmp_path / "bad.txt")})
        pipeline = IngestionPipeline(chunker, FakeEmbeddingPipeline(), FakeService([bad, good]))

        result = pipeline.ingest(tmp_path)

        assert result.succeeded == 1
        assert result.failed == 1
        assert "chunk boom" in (result.file_results[0].error or "")

    def test_embedding_failure_is_isolated(self, tmp_path: Path) -> None:
        first = successful_load(tmp_path / "a.txt")
        second = successful_load(tmp_path / "b.txt")
        pipeline = IngestionPipeline(
            FakeChunker(), FakeEmbeddingPipeline(fail=True), FakeService([first, second])
        )

        result = pipeline.ingest(tmp_path)

        assert result.succeeded == 0
        assert result.failed == 2
        assert all("embed pipeline boom" in (item.error or "") for item in result.file_results)

    def test_empty_ingestion(self, tmp_path: Path) -> None:
        pipeline = IngestionPipeline(FakeChunker(), FakeEmbeddingPipeline(), FakeService([]))
        assert pipeline.ingest(tmp_path) == PipelineRunResult()

    def test_default_service(self) -> None:
        pipeline = IngestionPipeline(FakeChunker(), FakeEmbeddingPipeline())
        assert isinstance(pipeline.document_service, DocumentIngestionService)


class TestOrchestrationIntegration:
    def test_end_to_end_twice_is_idempotent(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha alpha alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("# Head\n\nbeta beta beta", encoding="utf-8")

        def encoder(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        embedding_pipeline = EmbeddingPipeline(
            EmbeddingService(encoder=encoder),
            VectorStoreWriter(collection=MemoryCollection()),
        )
        pipeline = IngestionPipeline(
            FixedSizeChunker(ChunkConfig(chunk_size=50, overlap=10)),
            embedding_pipeline,
        )

        first = pipeline.ingest(tmp_path)
        second = pipeline.ingest(tmp_path)

        assert first.files == 2
        assert first.succeeded == 2
        assert first.chunks > 0
        assert first.written > 0
        assert second.skipped == first.written
        assert pipeline.embedding_pipeline.writer.count() == first.written

    def test_package_reexport(self) -> None:
        from src.airag.ingestion import IngestionPipeline as Exported

        assert Exported is IngestionPipeline


class FakeProgress:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def start(self, total: int) -> None:
        self.events.append(("start", total))

    def advance(self, label: str = "") -> None:
        self.events.append(("advance", label))

    def finish(self) -> None:
        self.events.append(("finish",))


class BrokenService(FakeService):
    def ingest_files(self, paths: list[Path]) -> IngestionResult:
        raise RuntimeError("service boom")


class TestCheckpointManager:
    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path / "checkpoint.json")
        assert not manager.is_completed("hash")

    def test_mark_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "checkpoint.json"
        manager = CheckpointManager(path)
        manager.mark_completed("hash-1", "doc.txt")

        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["completed"]["hash-1"]["source"] == "doc.txt"
        assert CheckpointManager(path).is_completed("hash-1")

    def test_corrupt_file_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.json"
        path.write_text("{not json", encoding="utf-8")
        assert not CheckpointManager(path).is_completed("hash")

    def test_clear(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(path)
        manager.mark_completed("hash", "doc.txt")
        manager.clear()
        assert not path.exists()
        assert not manager.is_completed("hash")


class TestProgressAndResume:
    def test_progress_events(self, tmp_path: Path) -> None:
        service = FakeService([
            successful_load(tmp_path / "a.txt"),
            successful_load(tmp_path / "b.txt"),
        ])
        progress = FakeProgress()
        pipeline = IngestionPipeline(
            FakeChunker(), FakeEmbeddingPipeline(), service, progress=progress
        )

        pipeline.ingest(tmp_path)

        assert progress.events[0] == ("start", 2)
        assert [event[0] for event in progress.events].count("advance") == 2
        assert progress.events[-1] == ("finish",)

    def test_progress_finishes_on_service_error(self, tmp_path: Path) -> None:
        progress = FakeProgress()
        pipeline = IngestionPipeline(
            FakeChunker(), FakeEmbeddingPipeline(),
            BrokenService([]), progress=progress,
        )

        with pytest.raises(RuntimeError, match="service boom"):
            pipeline.ingest(tmp_path / "a.txt")
        assert progress.events[-1] == ("finish",)

    def test_null_progress_is_default(self) -> None:
        pipeline = IngestionPipeline(FakeChunker(), FakeEmbeddingPipeline())
        assert isinstance(pipeline.progress, NullProgress)

    def test_checkpoint_skips_completed_document(self, tmp_path: Path) -> None:
        load = successful_load(tmp_path / "a.txt")
        checkpoint = CheckpointManager(tmp_path / "checkpoint.json")
        checkpoint.mark_completed(load.document.content_hash, str(load.source_path))
        chunker = FakeChunker()
        pipeline = IngestionPipeline(
            chunker, FakeEmbeddingPipeline(), FakeService([load]), checkpoint=checkpoint
        )

        result = pipeline.ingest(tmp_path / "a.txt")

        assert result.resumed == 1
        assert result.succeeded == 0
        assert result.file_results[0].resumed is True
        assert chunker.calls == []

    def test_successful_document_is_checkpointed(self, tmp_path: Path) -> None:
        load = successful_load(tmp_path / "a.txt")
        checkpoint = CheckpointManager(tmp_path / "checkpoint.json")
        pipeline = IngestionPipeline(
            FakeChunker(), FakeEmbeddingPipeline(), FakeService([load]),
            checkpoint=checkpoint,
        )

        result = pipeline.ingest(tmp_path / "a.txt")

        assert result.succeeded == 1
        assert result.resumed == 0
        assert checkpoint.is_completed(load.document.content_hash)

    def test_tqdm_progress_uses_injected_bar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[tuple[Any, ...]] = []

        class FakeBar:
            def __init__(self, **kwargs: Any) -> None:
                events.append(("create", kwargs.get("total")))

            def update(self, count: int) -> None:
                events.append(("update", count))

            def set_postfix_str(self, label: str) -> None:
                events.append(("label", label))

            def close(self) -> None:
                events.append(("close",))

        monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeBar))
        progress = TqdmProgress()
        progress.start(2)
        progress.advance("a.txt")
        progress.finish()

        assert events == [
            ("create", 2), ("update", 1), ("label", "a.txt"), ("close",)
        ]

