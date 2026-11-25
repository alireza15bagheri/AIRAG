"""
Unit tests for the sentence-aware chunker (step 2.3.3).

Covers boundary detection (including abbreviations, initials and multilingual
punctuation), sentence packing with overlap, page/offset/metadata handling,
injected splitters, token-mode lengths and protocol conformance.
"""

from __future__ import annotations

import pytest

from src.airag.ingestion.chunking.base import ChunkConfig, Chunker
from src.airag.ingestion.chunking.sentence import SentenceAwareChunker
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument


def make_document(text: str, pages: list[str] | None = None) -> ExtractedDocument:
    return ExtractedDocument(
        text=text,
        metadata={"source": "doc.txt", "format": "text"},
        pages=pages if pages is not None else [text],
    )


class FakeTokenizer:
    """One token per ASCII character."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class TestDefaultSplitting:
    def test_basic_sentences(self) -> None:
        document = make_document("One. Two. Three?")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=4)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["One.", "Two.", "Three?"]

    def test_abbreviations_are_not_boundaries(self) -> None:
        document = make_document("Dr. Smith arrived. He left.")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=20)).chunk(document)

        assert [chunk.text for chunk in chunks] == [
            "Dr. Smith arrived.",
            "He left.",
        ]

    def test_initials_are_not_boundaries(self) -> None:
        document = make_document("J. Smith went home. Then he slept.")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=25)).chunk(document)

        assert [chunk.text for chunk in chunks] == [
            "J. Smith went home.",
            "Then he slept.",
        ]

    def test_lowercase_following_continues_sentence(self) -> None:
        document = make_document("e.g. this is one. Next.")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=20)).chunk(document)

        assert [chunk.text for chunk in chunks] == ["e.g. this is one.", "Next."]

    def test_multilingual_punctuation(self) -> None:
        document = make_document("سلام دنیا. خوب است؟ بله.")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=10)).chunk(document)

        assert [chunk.text for chunk in chunks] == [
            "سلام دنیا.",
            "خوب است؟",
            "بله.",
        ]

    def test_empty_and_whitespace_documents(self) -> None:
        chunker = SentenceAwareChunker(ChunkConfig(chunk_size=10))
        assert chunker.chunk(make_document("")) == []
        assert chunker.chunk(make_document("   ")) == []

    def test_single_long_sentence_is_not_split(self) -> None:
        text = "A" * 30 + "."
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=10)).chunk(
            make_document(text)
        )
        assert [chunk.text for chunk in chunks] == [text]


class TestPackingAndMetadata:
    def test_overlap_reincludes_tail_sentence(self) -> None:
        document = make_document("Aa. Bb. Cc. Dd.")
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=7, overlap=3)).chunk(
            document
        )

        assert [chunk.text for chunk in chunks] == [
            "Aa. Bb.",
            "Bb. Cc.",
            "Cc. Dd.",
            "Dd.",
        ]
        assert chunks[1].text.startswith("Bb.")

    def test_metadata_and_page_offsets(self) -> None:
        pages = ["First sentence. Second one.", "Third sentence here."]
        document = make_document(" ".join(pages), pages=pages)
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=100)).chunk(document)

        assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "doc.txt"
            assert chunk.metadata["hash"] == document.content_hash
            assert chunk.metadata["ingested_at"] == document.ingested_at
            assert chunk.metadata["chunk_index"] == index
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text

    def test_falls_back_to_full_text_when_pages_empty(self) -> None:
        document = make_document("One. Two.", pages=[])
        chunks = SentenceAwareChunker(ChunkConfig(chunk_size=3)).chunk(document)
        assert [chunk.text for chunk in chunks] == ["One.", "Two."]
        assert all(chunk.metadata["page"] == 1 for chunk in chunks)


class TestInjectedSplitter:
    def test_injected_splitter_offsets(self) -> None:
        document = make_document("Hello world. Bye.")
        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=12),
            sentence_splitter=lambda text: ["Hello world.", "Bye."],
        )
        chunks = chunker.chunk(document)

        assert [chunk.text for chunk in chunks] == ["Hello world.", "Bye."]
        assert document.text[chunks[0].start_char : chunks[0].end_char] == "Hello world."

    def test_unmappable_splitter_output_is_skipped(self) -> None:
        document = make_document("Hello world. Bye.")
        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=100),
            sentence_splitter=lambda text: ["not in source", "Bye."],
        )
        assert [chunk.text for chunk in chunker.chunk(document)] == ["Bye."]

    def test_splitter_failure_raises_chunk_error(self) -> None:
        def boom(text: str) -> list[str]:
            raise RuntimeError("splitter failed")

        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=100), sentence_splitter=boom
        )
        with pytest.raises(ChunkError, match="(?i)sentence splitting failed"):
            chunker.chunk(make_document("Hello."))


class TestTokenMode:
    def test_injected_tokenizer(self) -> None:
        document = make_document("Aa. Bb. Cc.")
        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=5, length_unit="token"),
            tokenizer=FakeTokenizer(),
        )
        assert [chunk.text for chunk in chunker.chunk(document)] == [
            "Aa.",
            "Bb.",
            "Cc.",
        ]

    def test_default_tiktoken_length(self) -> None:
        import tiktoken

        document = make_document("Alpha beta. Gamma delta.")
        chunks = SentenceAwareChunker(
            ChunkConfig(chunk_size=3, length_unit="token")
        ).chunk(document)

        encoding = tiktoken.get_encoding("cl100k_base")
        assert len(chunks) >= 2
        assert all(0 < len(encoding.encode(chunk.text)) <= 3 for chunk in chunks)

    def test_tokenizer_unavailable_raises_chunk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "tiktoken":
                raise ImportError("tiktoken unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=5, length_unit="token")
        )
        with pytest.raises(ChunkError, match="(?i)tokenization unavailable"):
            chunker.chunk(make_document("Hello. World."))

    def test_encode_failure_raises_chunk_error(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("encode failed")

        chunker = SentenceAwareChunker(
            ChunkConfig(chunk_size=5, length_unit="token"),
            tokenizer=BrokenTokenizer(),
        )
        with pytest.raises(ChunkError, match="(?i)tokenization failed"):
            chunker.chunk(make_document("Hello. World."))


class TestConformance:
    def test_protocol_conformance(self) -> None:
        assert isinstance(
            SentenceAwareChunker(ChunkConfig(chunk_size=10)), Chunker
        )

    def test_package_reexport(self) -> None:
        from src.airag.ingestion import chunking

        assert chunking.SentenceAwareChunker is SentenceAwareChunker
