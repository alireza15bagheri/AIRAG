"""
Document loaders: protocol, concrete implementations, and the loader factory.

This package is the single public import surface for ingestion step 2.1:

    from src.airag.ingestion.loaders import (
        DocumentLoader, PDFLoader, DocxLoader, MarkdownLoader, TextLoader,
        get_loader, register_loader, supported_extensions,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)
from src.airag.ingestion.loaders.docx import DocxLoader
from src.airag.ingestion.loaders.markdown import MarkdownLoader
from src.airag.ingestion.loaders.pdf import PDFLoader
from src.airag.ingestion.loaders.text import TextLoader

if TYPE_CHECKING:
    # The factory imports this package lazily inside _register_defaults(), so
    # these re-exports must stay out of the runtime import path.
    from src.airag.ingestion.factory import (
        get_loader as get_loader,
    )
    from src.airag.ingestion.factory import (
        register_loader as register_loader,
    )
    from src.airag.ingestion.factory import (
        supported_extensions as supported_extensions,
    )

_FACTORY_NAMES = frozenset({"get_loader", "register_loader", "supported_extensions"})


def __getattr__(name: str) -> Any:
    """Lazily expose the factory API without a circular import."""
    if name in _FACTORY_NAMES:
        from src.airag.ingestion import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DocumentLoader",
    "ExtractedDocument",
    "LoadResult",
    "PDFLoader",
    "DocxLoader",
    "MarkdownLoader",
    "TextLoader",
    "get_loader",
    "register_loader",
    "supported_extensions",
]
