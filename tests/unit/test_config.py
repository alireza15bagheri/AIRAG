"""
Unit tests for ingestion configuration (step 2.5.2).

Covers defaults, validation, YAML/JSON loading and component construction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.airag.ingestion.config import IngestionConfig, create_components, load_config
from src.airag.ingestion.exceptions import ValidationError
from src.airag.ingestion.pipeline import EmbeddingPipeline


class TestIngestionConfig:
    def test_defaults(self) -> None:
        config = IngestionConfig()

        assert config.chunker == "fixed"
        assert config.chunk_size == 500
        assert config.overlap == 50
        assert config.length_unit == "char"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_batch_size == 32
        assert config.cache_size == 1024
        assert config.collection_name == "airag_chunks"
        assert config.persist_directory is None
        assert config.recursive is True
        assert config.loader_options == {}

    def test_to_dict_round_trip(self) -> None:
        data = IngestionConfig(chunk_size=100).to_dict()
        assert IngestionConfig(**data).chunk_size == 100

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"chunker": "wrong"}, "chunker must be one of"),
            ({"chunk_size": 0}, "chunk_size must be greater than zero"),
            ({"overlap": -1}, "overlap must be non-negative"),
            ({"chunk_size": 10, "overlap": 10}, "smaller than chunk_size"),
            ({"length_unit": "word"}, "length_unit must be"),
            ({"embedding_batch_size": 0}, "embedding_batch_size must be greater"),
            ({"cache_size": -1}, "cache_size must be non-negative"),
            ({"loader_options": []}, "loader_options must be a mapping"),
            ({"loader_options": {1: "x"}}, "keys must be strings"),
        ],
    )
    def test_validation(self, values: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            IngestionConfig(**values)


class TestLoadConfig:
    def test_none_returns_defaults(self) -> None:
        assert load_config(None) == IngestionConfig()

    def test_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "ingestion.yaml"
        path.write_text(
            "chunker: semantic\nchunk_size: 120\noverlap: 12\n"
            "length_unit: token\nrecursive: false\n"
            "loader_options:\n  password: secret\n",
            encoding="utf-8",
        )

        config = load_config(path)

        assert config.chunker == "semantic"
        assert config.chunk_size == 120
        assert config.overlap == 12
        assert config.length_unit == "token"
        assert config.recursive is False
        assert config.loader_options == {"password": "secret"}

    def test_json(self, tmp_path: Path) -> None:
        path = tmp_path / "ingestion.json"
        path.write_text('{"chunker": "sentence", "cache_size": 0}', encoding="utf-8")

        config = load_config(path)

        assert config.chunker == "sentence"
        assert config.cache_size == 0

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path) == IngestionConfig()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="Config file not found"):
            load_config(tmp_path / "missing.yaml")

    def test_non_mapping_root(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="must be a mapping"):
            load_config(path)

    def test_unknown_key(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown.yaml"
        path.write_text("unknown_key: 1\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="Unknown config key"):
            load_config(path)

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValidationError, match="Invalid JSON config"):
            load_config(path)

    def test_yaml_shim_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class Shim:
            def safe_load(self, text: str) -> Any:
                raise AttributeError("no safe_load")

            def YAML(self, **kwargs: Any) -> Shim:
                return self

            def load(self, text: str) -> dict[str, Any]:
                return {"chunker": "semantic"}

        monkeypatch.setitem(sys.modules, "yaml", Shim())
        path = tmp_path / "shim.yaml"
        path.write_text("chunker: semantic\n", encoding="utf-8")

        assert load_config(path).chunker == "semantic"


    def test_unreadable_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "unreadable.yaml"
        path.write_text("chunker: fixed\n", encoding="utf-8")

        def boom(*args: object, **kwargs: object) -> str:
            raise OSError("disk error")

        monkeypatch.setattr(Path, "read_text", boom)
        with pytest.raises(ValidationError, match="Could not read config file"):
            load_config(path)

    def test_missing_yaml_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "yaml", None)
        path = tmp_path / "no_yaml.yaml"
        path.write_text("chunker: fixed\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="PyYAML is required"):
            load_config(path)

    def test_invalid_yaml_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        def boom(text: str) -> None:
            raise ValueError("bad yaml")

        monkeypatch.setitem(
            sys.modules,
            "yaml",
            SimpleNamespace(safe_load=boom, YAML=lambda **kwargs: None),
        )
        path = tmp_path / "bad.yaml"
        path.write_text("chunker: fixed\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="Invalid YAML config"):
            load_config(path)


class TestCreateComponents:
    @pytest.mark.parametrize(
        ("chunker", "expected"),
        [("fixed", "FixedSizeChunker"), ("sentence", "SentenceAwareChunker"),
         ("semantic", "SemanticChunker")],
    )
    def test_strategy_selection(self, chunker: str, expected: str) -> None:
        config = IngestionConfig(chunker=chunker, chunk_size=100, overlap=10)

        built_chunker, pipeline = create_components(config)

        assert type(built_chunker).__name__ == expected
        assert isinstance(pipeline, EmbeddingPipeline)
        assert pipeline.embedder is not None
        assert pipeline.writer is not None
