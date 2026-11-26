"""
Embedding service for Phase 2 step 2.4.1.

Provides batched, cached embedding generation with bounded retry/backoff.
The underlying encoder is injectable for deterministic tests; the default
lazily loads ``SentenceTransformer`` on first use (it may download weights if
the model is not cached locally).

``EmbeddingService.__call__`` makes the service a drop-in ``embedder`` for
``SemanticChunker`` without changing that step's code.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from src.airag.ingestion.exceptions import EmbeddingError

Encoder = Callable[[list[str]], Any]
SleepFunc = Callable[[float], None]


class EmbeddingService:
    """Generate embeddings in batches, with LRU caching and retries."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_size: int = 1024,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        encoder: Encoder | None = None,
        sleep: SleepFunc | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            model_name: SentenceTransformer model used by the default encoder.
            batch_size: Maximum number of unique texts sent per encoder call.
            cache_size: Maximum cached vectors; 0 disables caching.
            max_retries: Retries after the first failed encoder call.
            retry_backoff: Base delay in seconds, doubled after each retry.
            encoder: Optional callable replacing the default model encoder.
            sleep: Optional sleep function, injected by tests.
        """
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be greater than zero")
        if cache_size < 0:
            raise EmbeddingError("cache_size must be non-negative")
        if max_retries < 0:
            raise EmbeddingError("max_retries must be non-negative")
        if retry_backoff < 0:
            raise EmbeddingError("retry_backoff must be non-negative")

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._encoder = encoder
        self._sleep = sleep if sleep is not None else time.sleep
        self._model: Any = None
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._dimension: int | None = None

    def embed(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Embed texts in batches, reusing cached vectors.

        Args:
            texts: One string or a sequence of strings.

        Returns:
            One vector per input text, in the same order.
        """
        input_texts = [texts] if isinstance(texts, str) else list(texts)
        if not input_texts:
            return []

        result: list[list[float]] = [[] for _ in input_texts]
        pending: OrderedDict[str, list[int]] = OrderedDict()
        for index, text in enumerate(input_texts):
            cached = self._get_cached(text)
            if cached is not None:
                result[index] = list(cached)
            else:
                pending.setdefault(text, []).append(index)

        if pending:
            unique = list(pending)
            for offset in range(0, len(unique), self.batch_size):
                batch = unique[offset : offset + self.batch_size]
                vectors = self._encode_with_retry(batch)
                for text, vector in zip(batch, vectors, strict=True):
                    self._store(text, vector)
                    for index in pending[text]:
                        result[index] = list(vector)
        return result

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Allow the service to be used as ``SemanticChunker``'s embedder."""
        return self.embed(texts)

    def clear_cache(self) -> None:
        """Drop all cached vectors and reset the known embedding dimension."""
        self._cache.clear()
        self._dimension = None

    def _get_cached(self, text: str) -> tuple[float, ...] | None:
        if text not in self._cache:
            return None
        self._cache.move_to_end(text)
        return self._cache[text]

    def _store(self, text: str, vector: Sequence[float]) -> None:
        if self.cache_size == 0:
            return
        vector_tuple = self._validate_vector(vector)
        self._cache[text] = vector_tuple
        self._cache.move_to_end(text)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        try:
            validated = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as e:
            raise EmbeddingError(f"Invalid embedding vector: {e}") from e
        if not validated:
            raise EmbeddingError("Embedding vector must not be empty")
        if self._dimension is None:
            self._dimension = len(validated)
        elif len(validated) != self._dimension:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(validated)}"
            )
        return validated

    def _encode_with_retry(self, batch: list[str]) -> list[Sequence[float]]:
        encoder = self._get_encoder()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = encoder(batch)
                vectors = list(raw)
                if len(vectors) != len(batch):
                    raise EmbeddingError(
                        f"Encoder returned {len(vectors)} vectors for "
                        f"{len(batch)} texts"
                    )
                return vectors
            except EmbeddingError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self._sleep(self.retry_backoff * (2**attempt))
        raise EmbeddingError(
            f"Embedding failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _get_encoder(self) -> Encoder:
        """Return the injected encoder or lazily load the default model."""
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers is required for the default encoder"
            ) from e
        try:
            if self._model is None:
                self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise EmbeddingError(f"Could not load model {self.model_name!r}: {e}") from e

        def encode(texts: list[str]) -> Any:
            return self._model.encode(texts, convert_to_numpy=True)

        return encode
