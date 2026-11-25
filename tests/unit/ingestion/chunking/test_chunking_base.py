"""
Unit tests for the chunking contract (step 2.3.1).

Covers ChunkConfig validation/defaults, Chunk construction and immutability,
and protocol conformance for concrete chunker implementations.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, Chunker
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument


class TestChunkConfig:
    """Configuration value object and its validation rules."""

    def test_defaults(self) -> None:
        config = ChunkConfig(chunk_size=500)
        assert config.chunk_size == 500
        assert config.overlap == 0
        assert config.length_unit == "char"

    def test_custom_values(self) -> None:
        config = ChunkConfig(chunk_size=200, overlap=20, length_unit="token")
        assert config.chunk_size == 200
        assert config.overlap == 20
        assert config.length_unit == "token"

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"chunk_size": 0}, "greater than zero"),
            ({"chunk_size": -5}, "greater than zero"),
            ({"chunk_size": 100, "overlap": -1}, "non-negative"),
            ({"chunk_size": 100, "overlap": 100}, "less than chunk_size"),
            ({"chunk_size": 100, "overlap": 101}, "less than chunk_size"),
            (
                {"chunk_size": 100, "length_unit": "word"},
                "(?i)unsupported length_unit",
            ),
        ],
    )
    def test_invalid_config_raises_chunk_error(
        self, values: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ChunkError, match=message):
            ChunkConfig(**values)

    def test_frozen(self) -> None:
        config = ChunkConfig(chunk_size=100)
        with pytest.raises(FrozenInstanceError):
            config.chunk_size = 200

    def test_slots(self) -> None:
        assert not hasattr(ChunkConfig(chunk_size=100), "__dict__")

    def test_equality_by_value(self) -> None:
        assert ChunkConfig(chunk_size=100, overlap=10) == ChunkConfig(
            chunk_size=100, overlap=10
        )
        assert ChunkConfig(chunk_size=100) != ChunkConfig(chunk_size=200)


class TestChunk:
    """Chunk value object and metadata contract."""

    def test_minimal(self) -> None:
        chunk = Chunk(text="hello")
        assert chunk.text == "hello"
        assert chunk.metadata == {}
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 0

    def test_full_construction(self) -> None:
        metadata = {
            "source": "doc.txt",
            "hash": "abc123",
            "ingested_at": "2026-01-01T00:00:00+00:00",
            "section": "Introduction",
        }
        chunk = Chunk(
            text="hello",
            metadata=metadata,
            chunk_index=3,
            start_char=10,
            end_char=15,
        )
        assert chunk.metadata == metadata
        assert chunk.chunk_index == 3
        assert (chunk.start_char, chunk.end_char) == (10, 15)

    def test_default_metadata_isolation(self) -> None:
        a = Chunk(text="a")
        b = Chunk(text="b")
        a.metadata["source"] = "a.txt"
        assert b.metadata == {}

    def test_frozen(self) -> None:
        chunk = Chunk(text="hello")
        with pytest.raises(FrozenInstanceError):
            chunk.text = "modified"

    def test_slots(self) -> None:
        assert not hasattr(Chunk(text="hello"), "__dict__")

    def test_equality_by_value(self) -> None:
        assert Chunk(text="x", chunk_index=1) == Chunk(text="x", chunk_index=1)
        assert Chunk(text="x") != Chunk(text="y")


class TestChunkerProtocol:
    """Runtime conformance of concrete chunker implementations."""

    def test_conforming_class_is_instance(self) -> None:
        class DummyChunker:
            def chunk(self, document: ExtractedDocument) -> list[Chunk]:
                return [Chunk(text=document.text)]

        assert isinstance(DummyChunker(), Chunker)

    def test_non_conforming_class_is_not_instance(self) -> None:
        class NotAChunker:
            def split(self, text: str) -> list[str]:
                return [text]

        assert not isinstance(NotAChunker(), Chunker)

    def test_protocol_call_returns_canonical_chunks(self) -> None:
        class DummyChunker:
            def __init__(self, config: ChunkConfig) -> None:
                self.config = config

            def chunk(self, document: ExtractedDocument) -> list[Chunk]:
                return [
                    Chunk(
                        text=document.text[:5],
                        metadata={
                            **document.metadata,
                            "hash": document.content_hash,
                            "chunk_index": 0,
                        },
                        chunk_index=0,
                        start_char=0,
                        end_char=5,
                    )
                ]

        document = ExtractedDocument(
            text="hello world",
            metadata={"source": "doc.txt", "format": "text"},
        )
        chunks = DummyChunker(ChunkConfig(chunk_size=5)).chunk(document)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == "hello"
        assert chunks[0].metadata["source"] == "doc.txt"
        assert chunks[0].metadata["hash"] == document.content_hash
        assert chunks[0].end_char == 5


def test_package_reexports() -> None:
    from src.airag.ingestion import chunking

    assert chunking.Chunk is Chunk
    assert chunking.ChunkConfig is ChunkConfig
    assert chunking.Chunker is Chunker
