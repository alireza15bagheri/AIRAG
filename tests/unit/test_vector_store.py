"""
Unit tests for the vector store writer (step 2.4.2).

Covers batching, deterministic IDs, idempotent skip/update, metadata
sanitization, client/collection injection, default ChromaDB creation paths,
error wrapping and an end-to-end chunk → embed → store smoke test.
"""

from __future__ import annotations

import builtins
import hashlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.airag.ingestion.chunking import Chunk, ChunkConfig, SemanticChunker
from src.airag.ingestion.embedding import EmbeddingService
from src.airag.ingestion.exceptions import IngestionError
from src.airag.ingestion.models import ExtractedDocument
from src.airag.ingestion.vector_store import VectorStoreWriter, WriteResult


class FakeCollection:
    """Minimal ChromaDB-compatible in-memory collection."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.get_calls: list[list[str]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.fail_get = False
        self.fail_upsert = False

    def get(self, ids: list[str], include: list[str] | None = None) -> dict[str, Any]:
        self.get_calls.append(list(ids))
        if self.fail_get:
            raise RuntimeError("get exploded")
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
        self.upsert_calls.append(
            {
                "ids": list(ids),
                "embeddings": [list(vector) for vector in embeddings],
                "documents": list(documents),
                "metadatas": [dict(metadata) for metadata in metadatas],
            }
        )
        if self.fail_upsert:
            raise RuntimeError("upsert exploded")
        for chunk_id, vector, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.items[chunk_id] = {
                "embeddings": list(vector),
                "documents": document,
                "metadatas": dict(metadata),
            }

    def count(self) -> int:
        return len(self.items)


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.names: list[str] = []

    def get_or_create_collection(self, name: str) -> FakeCollection:
        self.names.append(name)
        return self.collection


def make_chunk(
    text: str, index: int = 0, metadata: dict[str, Any] | None = None
) -> Chunk:
    values: dict[str, Any] = {
        "source": "doc.txt",
        "hash": "doc-hash",
        "format": "text",
        "page": 1,
    }
    if metadata:
        values.update(metadata)
    return Chunk(
        text=text,
        metadata=values,
        chunk_index=index,
        start_char=0,
        end_char=len(text),
    )


def vectors(count: int) -> list[list[float]]:
    return [[1.0, 0.0] for _ in range(count)]


class TestContract:
    def test_invalid_batch_size(self) -> None:
        with pytest.raises(IngestionError, match="batch_size must be greater"):
            VectorStoreWriter(batch_size=0)

    def test_write_result_defaults(self) -> None:
        assert WriteResult() == WriteResult()
        a = WriteResult()
        b = WriteResult()
        a.ids.append("x")
        assert b.ids == []

    def test_empty_write(self) -> None:
        collection = FakeCollection()
        result = VectorStoreWriter(collection=collection).write([], [])

        assert result == WriteResult()
        assert collection.get_calls == []
        assert collection.upsert_calls == []

    def test_length_mismatch(self) -> None:
        writer = VectorStoreWriter(collection=FakeCollection())
        with pytest.raises(IngestionError, match="2 chunks but 1 embeddings"):
            writer.write([make_chunk("a"), make_chunk("b")], vectors(1))


class TestWrite:
    def test_ids_and_payload(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection)
        chunks = [make_chunk("alpha", 0), make_chunk("beta", 1)]

        result = writer.write(chunks, vectors(2))

        assert result.written == 2
        assert result.updated == 0
        assert result.skipped == 0
        assert result.ids == ["doc-hash:0", "doc-hash:1"]
        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["documents"] == ["alpha", "beta"]
        assert call["embeddings"] == vectors(2)
        assert call["metadatas"][0]["chunk_hash"] == hashlib.sha256(
            b"alpha"
        ).hexdigest()
        assert call["metadatas"][1]["chunk_index"] == 1

    def test_batching(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection, batch_size=2)
        chunks = [make_chunk(str(i), i) for i in range(5)]

        result = writer.write(chunks, vectors(5))

        assert result.written == 5
        assert [len(call["ids"]) for call in collection.upsert_calls] == [2, 2, 1]

    def test_second_write_is_skipped(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection)
        chunks = [make_chunk("alpha", 0), make_chunk("beta", 1)]

        first = writer.write(chunks, vectors(2))
        second = writer.write(chunks, vectors(2))

        assert first.written == 2
        assert second.skipped == 2
        assert second.written == 0
        assert len(collection.upsert_calls) == 1
        assert writer.count() == 2

    def test_changed_chunk_text_updates(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection)

        writer.write([make_chunk("alpha")], vectors(1))
        result = writer.write([make_chunk("beta")], vectors(1))

        assert result.updated == 1
        assert result.skipped == 0
        assert collection.items["doc-hash:0"]["documents"] == "beta"
        assert writer.count() == 1

    def test_missing_document_hash_uses_chunk_hash(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection)
        chunk = make_chunk("alpha", metadata={"hash": None})

        writer.write([chunk], vectors(1))

        expected = hashlib.sha256(b"alpha").hexdigest()
        assert collection.upsert_calls[0]["ids"] == [f"{expected}:0"]

    def test_missing_chunk_hash_forces_update(self) -> None:
        collection = FakeCollection()
        collection.items["doc-hash:0"] = {
            "embeddings": [1.0, 0.0],
            "documents": "old",
            "metadatas": {"source": "doc.txt"},
        }
        writer = VectorStoreWriter(collection=collection)

        result = writer.write([make_chunk("alpha")], vectors(1))

        assert result.updated == 1

    def test_metadata_sanitization(self) -> None:
        collection = FakeCollection()
        writer = VectorStoreWriter(collection=collection)
        chunk = make_chunk(
            "alpha",
            metadata={
                "title": None,
                "headings": [{"level": 1, "text": "H"}],
                "flag": True,
                "score": 0.5,
            },
        )

        writer.write([chunk], vectors(1))

        metadata = collection.upsert_calls[0]["metadatas"][0]
        assert "title" not in metadata
        assert metadata["flag"] is True
        assert metadata["score"] == 0.5
        assert metadata["headings"] == '[{"level": 1, "text": "H"}]'


class TestFailurePaths:
    def test_lookup_failure(self) -> None:
        collection = FakeCollection()
        collection.fail_get = True
        writer = VectorStoreWriter(collection=collection)

        with pytest.raises(IngestionError, match="lookup failed"):
            writer.write([make_chunk("alpha")], vectors(1))

    def test_upsert_failure(self) -> None:
        collection = FakeCollection()
        collection.fail_upsert = True
        writer = VectorStoreWriter(collection=collection)

        with pytest.raises(IngestionError, match="upsert failed"):
            writer.write([make_chunk("alpha")], vectors(1))

    def test_count_failure(self) -> None:
        class BrokenCollection(FakeCollection):
            def count(self) -> int:
                raise RuntimeError("count exploded")

        writer = VectorStoreWriter(collection=BrokenCollection())
        with pytest.raises(IngestionError, match="count failed"):
            writer.count()

    def test_injected_client(self) -> None:
        collection = FakeCollection()
        client = FakeClient(collection)
        writer = VectorStoreWriter(client=client, collection_name="custom")

        writer.write([make_chunk("alpha")], vectors(1))

        assert client.names == ["custom"]
        assert collection.count() == 1

    def test_injected_client_failure(self) -> None:
        class BrokenClient:
            def get_or_create_collection(self, name: str) -> FakeCollection:
                raise RuntimeError("client exploded")

        writer = VectorStoreWriter(client=BrokenClient())
        with pytest.raises(IngestionError, match="Could not open vector collection"):
            writer.write([make_chunk("alpha")], vectors(1))

    def test_count_propagates_open_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "chromadb":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(IngestionError, match="chromadb is required"):
            VectorStoreWriter().count()


class TestDefaultClient:
    def test_missing_chromadb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "chromadb":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(IngestionError, match="chromadb is required"):
            VectorStoreWriter().write([make_chunk("alpha")], vectors(1))

    def test_default_client_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("client exploded")

        fake_module = SimpleNamespace(PersistentClient=boom, EphemeralClient=boom)
        monkeypatch.setitem(sys.modules, "chromadb", fake_module)

        with pytest.raises(IngestionError, match="Could not open vector collection"):
            VectorStoreWriter().write([make_chunk("alpha")], vectors(1))

    def test_persistent_client_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection = FakeCollection()
        paths: list[str] = []

        def persistent(path: str) -> FakeClient:
            paths.append(path)
            return FakeClient(collection)

        fake_module = SimpleNamespace(
            PersistentClient=persistent,
            EphemeralClient=lambda: FakeClient(collection),
        )
        monkeypatch.setitem(sys.modules, "chromadb", fake_module)

        writer = VectorStoreWriter(persist_directory="/tmp/airag-test")
        writer.write([make_chunk("alpha")], vectors(1))

        assert paths == ["/tmp/airag-test"]
        assert collection.count() == 1

    def test_ephemeral_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection = FakeCollection()
        created = 0

        def ephemeral() -> FakeClient:
            nonlocal created
            created += 1
            return FakeClient(collection)

        fake_module = SimpleNamespace(
            PersistentClient=lambda path: FakeClient(collection),
            EphemeralClient=ephemeral,
        )
        monkeypatch.setitem(sys.modules, "chromadb", fake_module)

        VectorStoreWriter().write([make_chunk("alpha")], vectors(1))

        assert created == 1
        assert collection.count() == 1


class TestIntegration:
    def test_chunk_embed_store(self) -> None:
        def topic_encoder(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts]

        text = "Alpha one. Alpha two. Beta three. Beta four."
        document = ExtractedDocument(
            text=text,
            metadata={"source": "doc.txt", "hash": "doc-hash"},
            pages=[text],
        )
        chunks = SemanticChunker(ChunkConfig(chunk_size=100), embedder=topic_encoder).chunk(
            document
        )
        embeddings = EmbeddingService(encoder=topic_encoder).embed(
            [chunk.text for chunk in chunks]
        )
        collection = FakeCollection()

        result = VectorStoreWriter(collection=collection).write(chunks, embeddings)

        assert result.written == 2
        assert collection.count() == 2
        assert set(collection.items) == {
            f"{document.content_hash}:0",
            f"{document.content_hash}:1",
        }
