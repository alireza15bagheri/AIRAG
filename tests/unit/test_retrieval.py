"""
Unit tests for the core retrieval contract (step 6.1).

Covers ``RetrievalResult`` construction/metadata mapping, validation helpers,
the ``Retriever`` protocol and package exports. No ChromaDB or model is used.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import src.airag.retrieval as retrieval_package
from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.exceptions import ValidationError
from src.airag.retrieval import (
    RetrievalResult,
    Retriever,
    validate_score_threshold,
    validate_top_k,
)


class TestValidationHelpers:
    @pytest.mark.parametrize("top_k", [1, 3, 100])
    def test_valid_top_k(self, top_k: int) -> None:
        assert validate_top_k(top_k) == top_k

    @pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "3", None])
    def test_invalid_top_k(self, invalid: Any) -> None:
        with pytest.raises(ValidationError, match="positive integer"):
            validate_top_k(invalid)

    def test_score_threshold_defaults_and_coercion(self) -> None:
        assert validate_score_threshold(None) is None
        assert validate_score_threshold(1) == 1.0
        assert validate_score_threshold(0.75) == 0.75

    @pytest.mark.parametrize("invalid", ["high", True, float("nan"), float("inf")])
    def test_invalid_score_threshold(self, invalid: Any) -> None:
        with pytest.raises(ValidationError):
            validate_score_threshold(invalid)


class TestRetrievalResult:
    def test_minimal_construction_and_defaults(self) -> None:
        result = RetrievalResult(chunk_id="doc:0", text="hello", score=0.9)

        assert result.source == ""
        assert result.format == ""
        assert result.page is None
        assert result.section is None
        assert result.metadata == {}

    @pytest.mark.parametrize("score", [-1.0, 0.0, 0.5, 1.0])
    def test_finite_scores_are_allowed(self, score: float) -> None:
        assert RetrievalResult(chunk_id="id", text="text", score=score).score == score

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"chunk_id": "", "text": "x", "score": 1.0},
            {"chunk_id": "id", "text": "", "score": 1.0},
            {"chunk_id": "id", "text": "x", "score": float("nan")},
            {"chunk_id": "id", "text": "x", "score": float("inf")},
        ],
    )
    def test_invalid_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult(**kwargs)

    def test_to_dict_is_json_serializable_and_isolated(self) -> None:
        metadata = {"source": "a.txt", "tags": ["x"]}
        result = RetrievalResult(
            chunk_id="id", text="text", score=0.8, metadata=metadata
        )

        payload = result.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        payload["metadata"]["tags"].append("y")
        assert metadata["tags"] == ["x"]

    def test_from_chunk_maps_provenance_metadata(self) -> None:
        chunk = Chunk(
            text="Retrieved text",
            metadata={
                "source": "docs/guide.pdf",
                "format": "pdf",
                "page": "3",
                "section": "Setup",
                "hash": "abc123",
                "chunk_index": 2,
            },
        )

        result = RetrievalResult.from_chunk(chunk, "abc123:2", 0.87)

        assert result.chunk_id == "abc123:2"
        assert result.text == "Retrieved text"
        assert result.score == pytest.approx(0.87)
        assert result.source == "docs/guide.pdf"
        assert result.format == "pdf"
        assert result.page == 3
        assert result.section == "Setup"
        assert result.metadata["hash"] == "abc123"
        assert result.metadata["chunk_index"] == 2

    @pytest.mark.parametrize("raw_page", [None, "", "not-a-page", [], {}])
    def test_from_chunk_handles_missing_or_invalid_page(
        self, raw_page: Any
    ) -> None:
        chunk = Chunk(text="text", metadata={"page": raw_page})
        result = RetrievalResult.from_chunk(chunk, "id", 0.5)
        assert result.page is None

    def test_from_chunk_copies_metadata(self) -> None:
        metadata: dict[str, Any] = {"source": "a.txt"}
        chunk = Chunk(text="text", metadata=metadata)
        result = RetrievalResult.from_chunk(chunk, "id", 0.5)

        result.metadata["source"] = "changed.txt"
        assert metadata["source"] == "a.txt"


class TestRetrieverProtocol:
    def test_conforming_class_is_instance(self) -> None:
        class FakeRetriever:
            def retrieve(
                self,
                query: str,
                top_k: int = 5,
                *,
                score_threshold: float | None = None,
                where: dict[str, Any] | None = None,
            ) -> list[RetrievalResult]:
                assert query == "question"
                assert top_k == 2
                assert score_threshold == pytest.approx(0.5)
                assert where == {"source": "a.txt"}
                return []

        assert isinstance(FakeRetriever(), Retriever)
        assert FakeRetriever().retrieve(
            "question", 2, score_threshold=0.5, where={"source": "a.txt"}
        ) == []

    def test_non_conforming_class_is_not_instance(self) -> None:
        class NotARetriever:
            def search(self, query: str) -> list[RetrievalResult]:
                return []

        assert not isinstance(NotARetriever(), Retriever)

    def test_empty_results_are_valid(self) -> None:
        class EmptyRetriever:
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return []

        results: list[RetrievalResult] = EmptyRetriever().retrieve("nothing")
        assert results == []


class TestPackageExports:
    def test_exports(self) -> None:
        assert retrieval_package.RetrievalResult is RetrievalResult
        assert retrieval_package.Retriever is Retriever
        assert retrieval_package.validate_top_k is validate_top_k
        assert retrieval_package.validate_score_threshold is validate_score_threshold
        assert set(retrieval_package.__all__) == {
            "RetrievalResult",
            "Retriever",
            "validate_score_threshold",
            "validate_top_k",
        }
