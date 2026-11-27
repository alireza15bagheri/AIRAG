"""
AIRAG Ingestion Package.

This package provides document loading, ingestion pipeline, and CLI.
"""

from __future__ import annotations

from src.airag.ingestion.embedding import EmbeddingService
from src.airag.ingestion.exceptions import (
    ChunkError,
    EmbeddingError,
    IngestionError,
    LoadError,
    UnsupportedFormatError,
    ValidationError,
)
from src.airag.ingestion.factory import (
    get_loader,
    register_loader,
    supported_extensions,
)
from src.airag.ingestion.loaders import (
    DocumentLoader,
    DocxLoader,
    ExtractedDocument,
    LoadResult,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)
from src.airag.ingestion.models import (
    ExtractedDocument as ModelExtractedDocument,
)
from src.airag.ingestion.models import (
    IngestionResult,
)
from src.airag.ingestion.models import (
    LoadResult as ModelLoadResult,
)
from src.airag.ingestion.pipeline import (
    EmbeddingPipeline,
    IngestionPipeline,
    PipelineRunResult,
)
from src.airag.ingestion.service import (
    DocumentIngestionService,
    create_ingestion_service,
)
from src.airag.ingestion.vector_store import VectorStoreWriter

__all__ = [
    # Exceptions
    "IngestionError",
    "LoadError",
    "UnsupportedFormatError",
    "ChunkError",
    "EmbeddingError",
    "ValidationError",
    # Factory
    "get_loader",
    "register_loader",
    "supported_extensions",
    # Loaders
    "DocumentLoader",
    "PDFLoader",
    "DocxLoader",
    "MarkdownLoader",
    "TextLoader",
    "ExtractedDocument",
    "LoadResult",
    # Models
    "ModelExtractedDocument",
    "ModelLoadResult",
    "IngestionResult",
    # Service
    "DocumentIngestionService",
    "create_ingestion_service",
    # Embedding
    "EmbeddingService",
    # Vector store
    "VectorStoreWriter",
    # Pipeline
    "EmbeddingPipeline",
    "IngestionPipeline",
    "PipelineRunResult",
]
