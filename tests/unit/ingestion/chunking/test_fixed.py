"""
Unit tests for the fixed-size chunker (step 2.3.2).

Covers character and token windows, overlap, page boundaries, offset mapping,
metadata propagation, tokenizer failure paths and protocol conformance.
"""

from __future__ import annotations

import pytest

from src.airag.ingestion.chunking.base import ChunkConfig, Chunker
from src.airag.ingestion.chunking.fixed import FixedSizeChunker
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument


def make_document(text: str, pages: list[str] | None = None) -> ExtractedDocument:
    return ExtractedDocument(
        text=text,
        metadata={"source": "doc.txt", "format": "text"},
        pages=pages if pages is not None else [text],
    )


class FakeTokenizer:
    """One token per ASCII character; no external encoding needed."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode_tokens_bytes(self, ids: list[int]) -> list[bytes]:
        return [chr(token_id).encode("utf-8") for token_id in ids]


class TestCharMode:
    def test_exact_size_no_overlap(self) -> None:
        document = make_document("abcdefghij")
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=4)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
        assert [(c.start_char, c.end_char) for c in chunks] == [
            (0, 4), (4, 8), (8, 10)
        ]

    def test_overlap(self) -> None:
        document = make_document("abcdefgh")
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=4, overlap=2)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh"]
        assert [(c.start_char, c.end_char) for c in chunks] == [
            (0, 4), (2, 6), (4, 8)
        ]

    def test_exact_multiple(self) -> None:
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=4)).chunk(
            make_document("abcdefgh")
        )
        assert [chunk.text for chunk in chunks] == ["abcd", "efgh"]

    def test_short_document_single_chunk(self) -> None:
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=50)).chunk(
            make_document("abc")
        )
        assert len(chunks) == 1
        assert chunks[0].text == "abc"
        assert (chunks[0].start_char, chunks[0].end_char) == (0, 3)

    def test_empty_document(self) -> None:
        assert FixedSizeChunker(ChunkConfig(chunk_size=4)).chunk(make_document("")) == []

    def test_whitespace_only_document(self) -> None:
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=2)).chunk(
            make_document("   ")
        )
        assert chunks == []

    def test_metadata_propagation(self) -> None:
        document = make_document("abcdef")
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=3)).chunk(document)

        assert len(chunks) == 2
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "doc.txt"
            assert chunk.metadata["format"] == "text"
            assert chunk.metadata["hash"] == document.content_hash
            assert chunk.metadata["ingested_at"] == document.ingested_at
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["chunk_index"] == index
            assert chunk.chunk_index == index

    def test_pages_are_chunked_separately(self) -> None:
        pages = ["first page text", "second page text"]
        document = make_document("\n\n".join(pages), pages=pages)

        chunks = FixedSizeChunker(ChunkConfig(chunk_size=10)).chunk(document)

        assert [chunk.metadata["page"] for chunk in chunks] == [1, 1, 2, 2]
        for chunk in chunks:
            assert document.text[chunk.start_char:chunk.end_char] == chunk.text

    def test_falls_back_to_full_text_when_pages_empty(self) -> None:
        document = make_document("abcdefgh", pages=[])
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=4)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["abcd", "efgh"]
        assert all(chunk.metadata["page"] == 1 for chunk in chunks)

    def test_page_not_found_in_text_uses_zero_offset(self) -> None:
        document = make_document("actual text", pages=["unrelated page"])
        chunks = FixedSizeChunker(ChunkConfig(chunk_size=20)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["unrelated page"]
        assert (chunks[0].start_char, chunks[0].end_char) == (0, 14)


class TestTokenMode:
    def test_token_windows_with_injected_tokenizer(self) -> None:
        document = make_document("abcdefgh")
        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=4, overlap=2, length_unit="token"),
            tokenizer=FakeTokenizer(),
        )

        chunks = chunker.chunk(document)

        assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh"]
        assert [(c.start_char, c.end_char) for c in chunks] == [
            (0, 4), (2, 6), (4, 8)
        ]

    def test_empty_token_stream(self) -> None:
        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=4, length_unit="token"), tokenizer=FakeTokenizer()
        )
        assert chunker.chunk(make_document("")) == []

    def test_encode_failure_raises_chunk_error(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("encode failed")

        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=4, length_unit="token"),
            tokenizer=BrokenTokenizer(),
        )
        with pytest.raises(ChunkError, match="(?i)tokenization failed"):
            chunker.chunk(make_document("abc"))

    def test_offset_mapping_failure_raises_chunk_error(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode_tokens_bytes(self, ids: list[int]) -> list[bytes]:
                raise RuntimeError("decode failed")

        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=2, length_unit="token"),
            tokenizer=BrokenTokenizer(),
        )
        with pytest.raises(ChunkError, match="(?i)offset mapping failed"):
            chunker.chunk(make_document("abc"))

    def test_missing_tiktoken_raises_chunk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "tiktoken":
                raise ImportError("tiktoken unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=4, length_unit="token")
        )
        with pytest.raises(ChunkError, match="tiktoken is required"):
            chunker.chunk(make_document("abc"))

    def test_tokenizer_load_failure_raises_chunk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tiktoken

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("encoding unavailable")

        monkeypatch.setattr(tiktoken, "get_encoding", boom)
        chunker = FixedSizeChunker(
            ChunkConfig(chunk_size=4, length_unit="token")
        )
        with pytest.raises(ChunkError, match="(?i)could not load tokenizer"):
            chunker.chunk(make_document("abc"))

    def test_default_tiktoken_encoding(self) -> None:
        import tiktoken

        document = make_document("alpha beta gamma delta epsilon zeta eta theta")
        chunks = FixedSizeChunker(
            ChunkConfig(chunk_size=3, overlap=1, length_unit="token")
        ).chunk(document)

        encoding = tiktoken.get_encoding("cl100k_base")
        assert len(chunks) > 1
        assert all(0 < len(encoding.encode(chunk.text)) <= 3 for chunk in chunks)
        for chunk in chunks:
            assert document.text[chunk.start_char:chunk.end_char] == chunk.text


class TestConformance:
    def test_protocol_conformance(self) -> None:
        assert isinstance(FixedSizeChunker(ChunkConfig(chunk_size=4)), Chunker)

    def test_package_reexport(self) -> None:
        from src.airag.ingestion import chunking

        assert chunking.FixedSizeChunker is FixedSizeChunker
