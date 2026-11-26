"""
Unit tests for the embedding service (step 2.4.1).

Covers validation, batching, LRU caching, retry/backoff, embedding validation,
lazy model loading and use as a SemanticChunker embedder. Tests inject an
encoder/sleep so no model weights are loaded and no real time is spent.
"""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.airag.ingestion.chunking import ChunkConfig, SemanticChunker
from src.airag.ingestion.embedding import EmbeddingService
from src.airag.ingestion.exceptions import EmbeddingError
from src.airag.ingestion.models import ExtractedDocument


class RecordingEncoder:
    """Deterministic encoder that records batches and can fail on demand."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.fail_times = fail_times

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("encoder boom")
        return [[float(len(text)), 1.0] for text in texts]


class TestConfiguration:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size must be greater than zero"),
            ({"cache_size": -1}, "cache_size must be non-negative"),
            ({"max_retries": -1}, "max_retries must be non-negative"),
            ({"retry_backoff": -0.1}, "retry_backoff must be non-negative"),
        ],
    )
    def test_invalid_config(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(EmbeddingError, match=message):
            EmbeddingService(**kwargs)


class TestBatchingAndCache:
    def test_empty_input(self) -> None:
        encoder = RecordingEncoder()
        assert EmbeddingService(encoder=encoder).embed([]) == []
        assert encoder.calls == []

    def test_single_string_input(self) -> None:
        service = EmbeddingService(encoder=RecordingEncoder())
        assert service.embed("hello") == [[5.0, 1.0]]

    def test_batching_preserves_order(self) -> None:
        encoder = RecordingEncoder()
        service = EmbeddingService(batch_size=2, encoder=encoder)

        vectors = service.embed(["a", "bb", "ccc", "dddd", "eeeee"])

        assert encoder.calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
        assert vectors == [
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]
        ]

    def test_duplicates_in_one_batch_encoded_once(self) -> None:
        encoder = RecordingEncoder()
        service = EmbeddingService(encoder=encoder)

        vectors = service.embed(["a", "a", "b"])

        assert encoder.calls == [["a", "b"]]
        assert vectors == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

    def test_cache_hit_across_calls(self) -> None:
        encoder = RecordingEncoder()
        service = EmbeddingService(encoder=encoder)

        assert service.embed(["a"]) == service.embed(["a"])
        assert encoder.calls == [["a"]]

    def test_lru_eviction(self) -> None:
        encoder = RecordingEncoder()
        service = EmbeddingService(cache_size=2, encoder=encoder)

        service.embed(["a", "b"])
        service.embed(["a"])  # moves a to most-recent
        service.embed(["c"])  # evicts b
        service.embed(["b"])

        assert encoder.calls == [["a", "b"], ["c"], ["b"]]

    def test_zero_cache_disables_caching(self) -> None:
        encoder = RecordingEncoder()
        service = EmbeddingService(cache_size=0, encoder=encoder)

        service.embed(["a"])
        service.embed(["a"])

        assert len(encoder.calls) == 2

    def test_clear_cache_resets_dimension(self) -> None:
        calls = 0

        def variable_encoder(texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            width = 2 if calls == 1 else 3
            return [[float(len(text))] * width for text in texts]

        service = EmbeddingService(encoder=variable_encoder)
        service.embed(["a"])
        service.clear_cache()

        assert service.embed(["bb"]) == [[2.0, 2.0, 2.0]]

    def test_callable_interface(self) -> None:
        service = EmbeddingService(encoder=RecordingEncoder())
        assert service(["a"]) == [[1.0, 1.0]]


class TestRetryAndValidation:
    def test_retry_then_success_with_backoff(self) -> None:
        delays: list[float] = []
        encoder = RecordingEncoder(fail_times=2)
        service = EmbeddingService(
            encoder=encoder, max_retries=3, retry_backoff=0.5,
            sleep=delays.append,
        )

        assert service.embed(["a"]) == [[1.0, 1.0]]
        assert len(encoder.calls) == 3
        assert delays == [0.5, 1.0]

    def test_retry_exhaustion(self) -> None:
        delays: list[float] = []
        encoder = RecordingEncoder(fail_times=99)
        service = EmbeddingService(
            encoder=encoder, max_retries=2, retry_backoff=0.5,
            sleep=delays.append,
        )

        with pytest.raises(EmbeddingError, match="after 3 attempts"):
            service.embed(["a"])
        assert len(encoder.calls) == 3
        assert delays == [0.5, 1.0]

    def test_encoder_wrong_vector_count(self) -> None:
        service = EmbeddingService(encoder=lambda texts: [])
        with pytest.raises(EmbeddingError, match="0 vectors for 2 texts"):
            service.embed(["a", "b"])

    def test_invalid_vector_values(self) -> None:
        service = EmbeddingService(encoder=lambda texts: [["not-a-number"]])
        with pytest.raises(EmbeddingError, match="Invalid embedding vector"):
            service.embed(["a"])

    def test_empty_vector(self) -> None:
        service = EmbeddingService(encoder=lambda texts: [[]])
        with pytest.raises(EmbeddingError, match="must not be empty"):
            service.embed(["a"])

    def test_dimension_mismatch(self) -> None:
        encoded = 0

        def variable_encoder(texts: list[str]) -> list[list[float]]:
            nonlocal encoded
            encoded += 1
            width = 2 if encoded == 1 else 3
            return [[1.0] * width for _ in texts]

        service = EmbeddingService(batch_size=1, encoder=variable_encoder)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            service.embed(["a", "b"])


class TestDefaultEncoder:
    def test_missing_sentence_transformers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(EmbeddingError, match="sentence-transformers is required"):
            EmbeddingService().embed(["a"])

    def test_model_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("model unavailable")

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=boom)
        )
        with pytest.raises(EmbeddingError, match="Could not load model"):
            EmbeddingService().embed(["a"])

    def test_lazy_model_loaded_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created = 0

        class FakeModel:
            def encode(
                self, texts: list[str], convert_to_numpy: bool = True
            ) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        def factory(name: str) -> FakeModel:
            nonlocal created
            created += 1
            return FakeModel()

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=factory),
        )
        service = EmbeddingService()
        assert service.embed(["a"]) == [[1.0, 0.0]]
        assert service.embed(["b"]) == [[1.0, 0.0]]
        assert created == 1
        assert service._model is not None


class TestIntegration:
    def test_usable_as_semantic_chunker_embedder(self) -> None:
        def topic_encoder(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts]

        text = "Alpha one. Alpha two. Beta three. Beta four."
        document = ExtractedDocument(text=text, metadata={"source": "doc.txt"}, pages=[text])
        service = EmbeddingService(encoder=topic_encoder)

        chunks = SemanticChunker(ChunkConfig(chunk_size=100), embedder=service).chunk(
            document
        )

        assert [chunk.text for chunk in chunks] == [
            "Alpha one. Alpha two.",
            "Beta three. Beta four.",
        ]
