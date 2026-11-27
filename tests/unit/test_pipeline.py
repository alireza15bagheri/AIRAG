"""
Unit tests for the chunk → embedding → ID pipeline (step 2.4.3).

Covers grouping by document ID, delegation to embedder/writer, aggregation,
error propagation and an end-to-end idempotent run with real 2.4 components.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from src.airag.ingestion.chunking import Chunk, ChunkConfig, SemanticChunker
from src.airag.ingestion.embedding import EmbeddingService
from src.airag.ingestion.exceptions import EmbeddingError, IngestionError
from src.airag.ingestion.models import ExtractedDocument
from src.airag.ingestion.pipeline import EmbeddingPipeline, PipelineResult
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
