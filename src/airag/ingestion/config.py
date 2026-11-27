"""
YAML/JSON configuration for the ingestion CLI (step 2.5.2).

``IngestionConfig`` holds settings for chunking, embedding, vector storage and
loader options. ``load_config`` reads a YAML or JSON file, and
``create_components`` builds the chunker and ``EmbeddingPipeline`` from it.

The plan also names ``config/ingestion.yaml``; a documented sample lives there.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from src.airag.ingestion.exceptions import ValidationError

VALID_CHUNKERS = ("fixed", "sentence", "semantic")
VALID_LENGTH_UNITS = ("char", "token")


@dataclass(slots=True)
class IngestionConfig:
    """Validated ingestion settings.

    Attributes:
        chunker: One of ``fixed``, ``sentence`` or ``semantic``.
        chunk_size: Maximum chunk length in ``length_unit`` units.
        overlap: Overlap between adjacent chunks.
        length_unit: ``char`` or ``token``.
        embedding_model: SentenceTransformer model name.
        embedding_batch_size: Texts per embedding encoder call.
        cache_size: Embedding LRU cache size; 0 disables caching.
        collection_name: ChromaDB collection name.
        persist_directory: Optional directory for persistent ChromaDB storage.
        recursive: Whether directory ingestion traverses subdirectories.
        dlq_path: Optional JSONL dead letter queue path; ``None`` disables it.
        loader_options: Extra keyword arguments for loader constructors.
    """

    chunker: str = "fixed"
    chunk_size: int = 500
    overlap: int = 50
    length_unit: str = "char"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    cache_size: int = 1024
    collection_name: str = "airag_chunks"
    persist_directory: str | None = None
    recursive: bool = True
    dlq_path: str | None = None
    loader_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValidationError`` when any setting is invalid."""
        if self.chunker not in VALID_CHUNKERS:
            raise ValidationError(
                f"chunker must be one of {', '.join(VALID_CHUNKERS)}, got {self.chunker!r}"
            )
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size must be greater than zero")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValidationError(
                "overlap must be non-negative and smaller than chunk_size"
            )
        if self.length_unit not in VALID_LENGTH_UNITS:
            raise ValidationError(
                f"length_unit must be 'char' or 'token', got {self.length_unit!r}"
            )
        if self.embedding_batch_size <= 0:
            raise ValidationError("embedding_batch_size must be greater than zero")
        if self.cache_size < 0:
            raise ValidationError("cache_size must be non-negative")
        if self.dlq_path is not None and not isinstance(self.dlq_path, str):
            raise ValidationError("dlq_path must be a string or null")
        if not isinstance(self.loader_options, dict):
            raise ValidationError("loader_options must be a mapping")
        if any(not isinstance(key, str) for key in self.loader_options):
            raise ValidationError("loader_options keys must be strings")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict, e.g. for JSON output or round-trip tests."""
        return asdict(self)


def load_config(path: str | Path | None = None) -> IngestionConfig:
    """Load configuration from YAML/JSON, or return defaults for ``None``."""
    if path is None:
        return IngestionConfig()

    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"Config file not found: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as e:
        raise ValidationError(f"Could not read config file: {e}") from e

    if not text.strip():
        data: Any = {}
    elif source.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON config: {e}") from e
    else:
        try:
            import yaml
        except ImportError as e:
            raise ValidationError("PyYAML is required to read YAML config") from e
        try:
            try:
                data = yaml.safe_load(text) or {}
            except AttributeError:
                # ruamel-backed shim without safe_load (see markdown loader).
                data = yaml.YAML(typ="safe", pure=True).load(text) or {}
        except Exception as e:
            raise ValidationError(f"Invalid YAML config: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError("Config root must be a mapping")
    try:
        return IngestionConfig(**data)
    except TypeError as e:
        raise ValidationError(f"Unknown config key: {e}") from e


def create_components(
    config: IngestionConfig, metrics: Any | None = None
) -> tuple[Any, Any]:
    """Build ``(chunker, embedding_pipeline)`` from validated settings.

    ``metrics`` is optional and defaults to a fresh registry inside the
    embedding pipeline.
    """
    from src.airag.ingestion.chunking.base import ChunkConfig
    from src.airag.ingestion.chunking.fixed import FixedSizeChunker
    from src.airag.ingestion.chunking.semantic import SemanticChunker
    from src.airag.ingestion.chunking.sentence import SentenceAwareChunker
    from src.airag.ingestion.embedding import EmbeddingService
    from src.airag.ingestion.metrics import MetricsRegistry
    from src.airag.ingestion.pipeline import EmbeddingPipeline
    from src.airag.ingestion.vector_store import VectorStoreWriter

    chunk_config = ChunkConfig(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        length_unit=cast(Any, config.length_unit),
    )
    embedder = EmbeddingService(
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size,
        cache_size=config.cache_size,
    )
    if config.chunker == "fixed":
        chunker: Any = FixedSizeChunker(chunk_config)
    elif config.chunker == "sentence":
        chunker = SentenceAwareChunker(chunk_config)
    else:
        chunker = SemanticChunker(chunk_config, embedder=embedder)

    writer = VectorStoreWriter(
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
    )
    return chunker, EmbeddingPipeline(
        embedder, writer, metrics if metrics is not None else MetricsRegistry()
    )
