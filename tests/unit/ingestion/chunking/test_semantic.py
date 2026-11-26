"""
Unit tests for the semantic chunker (step 2.3.4).

Covers cosine scoring, similarity-driven boundaries, the chunk_size cap,
threshold validation, embedder/tokenizer failure paths, page/offset/metadata
propagation and protocol conformance. All tests inject embeddings; no model
weights are downloaded.
"""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.airag.ingestion.chunking.base import ChunkConfig, Chunker
from src.airag.ingestion.chunking.semantic import SemanticChunker, cosine_similarity
from src.airag.ingestion.exceptions import ChunkError
from src.airag.ingestion.models import ExtractedDocument


def make_document(text: str, pages: list[str] | None = None) -> ExtractedDocument:
    return ExtractedDocument(
        text=text,
        metadata={"source": "doc.txt", "format": "text"},
        pages=pages if pages is not None else [text],
    )


def topic_embedder(texts: list[str]) -> list[list[float]]:
    """Two topic vectors: Alpha -> [1,0], everything else -> [0,1]."""
    return [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts]


def same_embedder(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


class FakeTokenizer:
    """One token per ASCII character."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(ChunkError, match="dimensions differ"):
            cosine_similarity([1.0], [1.0, 0.0])


class TestBoundaryDetection:
    def test_similarity_drop_creates_boundary(self) -> None:
        document = make_document("Alpha one. Alpha two. Beta three. Beta four.")
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=100), embedder=topic_embedder
        ).chunk(document)

        assert [chunk.text for chunk in chunks] == [
            "Alpha one. Alpha two.",
            "Beta three. Beta four.",
        ]
        assert [chunk.metadata["page"] for chunk in chunks] == [1, 1]

    def test_size_cap_forces_boundary(self) -> None:
        document = make_document("Alpha. Beta. Gamma.")
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=10), embedder=same_embedder
        ).chunk(document)

        assert [chunk.text for chunk in chunks] == ["Alpha.", "Beta.", "Gamma."]

    def test_threshold_zero_keeps_one_chunk(self) -> None:
        document = make_document("Alpha one. Alpha two. Beta three. Beta four.")
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=100), embedder=topic_embedder, threshold=0.0
        ).chunk(document)

        assert len(chunks) == 1
        assert chunks[0].text == document.text

    def test_threshold_validation(self) -> None:
        for bad in (-0.1, 1.1):
            with pytest.raises(ChunkError, match="between 0 and 1"):
                SemanticChunker(ChunkConfig(chunk_size=10), threshold=bad)

    def test_zero_vector_embedder(self) -> None:
        document = make_document("Alpha one. Beta two.")
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=100),
            embedder=lambda texts: [[0.0, 0.0], [1.0, 0.0]],
        ).chunk(document)
        assert len(chunks) == 2

    def test_empty_and_single_sentence(self) -> None:
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=10), embedder=topic_embedder
        )
        assert chunker.chunk(make_document("")) == []
        assert chunker.chunk(make_document("   ")) == []
        chunks = chunker.chunk(make_document("Alpha only."))
        assert [chunk.text for chunk in chunks] == ["Alpha only."]


class TestMetadataAndFailurePaths:
    def test_page_offsets_and_inherited_metadata(self) -> None:
        pages = ["Alpha one. Alpha two.", "Beta three. Beta four."]
        document = make_document(" ".join(pages), pages=pages)
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=100), embedder=topic_embedder
        ).chunk(document)

        assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "doc.txt"
            assert chunk.metadata["hash"] == document.content_hash
            assert chunk.metadata["ingested_at"] == document.ingested_at
            assert chunk.metadata["chunk_index"] == index
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text

    def test_embedding_count_mismatch(self) -> None:
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=10), embedder=lambda texts: []
        )
        with pytest.raises(ChunkError, match="Embedder returned 0 vectors"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_embedder_exception_is_wrapped(self) -> None:
        def boom(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding boom")

        chunker = SemanticChunker(ChunkConfig(chunk_size=10), embedder=boom)
        with pytest.raises(ChunkError, match="(?i)embedding failed: embedding boom"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_invalid_embedding_values(self) -> None:
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=10), embedder=lambda texts: [["not-a-number"]]
        )
        with pytest.raises(ChunkError, match="Invalid embeddings"):
            chunker.chunk(make_document("Alpha only."))

    def test_embedding_dimension_mismatch(self) -> None:
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=100),
            embedder=lambda texts: [[1.0, 0.0], [1.0, 0.0, 0.0]],
        )
        with pytest.raises(ChunkError, match="dimensions differ"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_missing_sentence_transformers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "sentence_transformers":
                raise ImportError("no sentence-transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        chunker = SemanticChunker(ChunkConfig(chunk_size=100))
        with pytest.raises(ChunkError, match="sentence-transformers is required"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_default_embedder_load_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("model unavailable")

        fake_module = SimpleNamespace(SentenceTransformer=boom)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        chunker = SemanticChunker(ChunkConfig(chunk_size=100))
        with pytest.raises(ChunkError, match="(?i)embedding failed: model unavailable"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_default_embedder_success_and_model_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeModel:
            def encode(self, texts: list[str], convert_to_numpy: bool = True) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        fake_module = SimpleNamespace(SentenceTransformer=lambda name: FakeModel())
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        chunker = SemanticChunker(ChunkConfig(chunk_size=100))
        chunks = chunker.chunk(make_document("Alpha one. Alpha two."))

        assert [chunk.text for chunk in chunks] == ["Alpha one. Alpha two."]
        assert chunker._model is not None
        assert chunker._model is chunker._model


class TestTokenMode:
    def test_injected_tokenizer_size_cap(self) -> None:
        document = make_document("Alpha. Beta. Gamma.")
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=5, length_unit="token"),
            embedder=same_embedder,
            tokenizer=FakeTokenizer(),
        )
        assert [chunk.text for chunk in chunker.chunk(document)] == [
            "Alpha.", "Beta.", "Gamma.",
        ]

    def test_default_tiktoken_length(self) -> None:
        import tiktoken

        document = make_document("Alpha beta. Gamma delta.")
        chunks = SemanticChunker(
            ChunkConfig(chunk_size=3, length_unit="token"), embedder=same_embedder
        ).chunk(document)

        encoding = tiktoken.get_encoding("cl100k_base")
        assert len(chunks) >= 2
        assert all(0 < len(encoding.encode(chunk.text)) <= 3 for chunk in chunks)

    def test_tokenizer_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        chunker = SemanticChunker(
            ChunkConfig(chunk_size=5, length_unit="token"), embedder=same_embedder
        )
        with pytest.raises(ChunkError, match="(?i)tokenization unavailable"):
            chunker.chunk(make_document("Alpha one. Beta two."))

    def test_tokenizer_encode_failure(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("encode failed")

        chunker = SemanticChunker(
            ChunkConfig(chunk_size=5, length_unit="token"),
            embedder=same_embedder,
            tokenizer=BrokenTokenizer(),
        )
        with pytest.raises(ChunkError, match="(?i)tokenization failed"):
            chunker.chunk(make_document("Alpha one. Beta two."))


class TestConformance:
    def test_protocol_conformance(self) -> None:
        assert isinstance(
            SemanticChunker(ChunkConfig(chunk_size=10), embedder=same_embedder),
            Chunker,
        )

    def test_package_reexport(self) -> None:
        from src.airag.ingestion import chunking

        assert chunking.SemanticChunker is SemanticChunker
