"""
Document loader registry and resolution API.

``_registry`` maps normalized extensions to DocumentLoader classes;
``get_loader`` returns a fresh instance per call. Directory walking, CLI,
and pipeline orchestration belong to steps 2.5.x-2.6.x.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.airag.ingestion.exceptions import UnsupportedFormatError
from src.airag.ingestion.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)

_registry: dict[str, type[DocumentLoader]] = {}


def _normalize_extension(extension: str) -> str:
    """Normalize an extension to lowercase with one leading dot."""
    normalized = extension.strip().lower()
    return normalized if normalized.startswith(".") else f".{normalized}"


def _validate_loader_class(loader_cls: Any, extension: str) -> None:
    """Reject symbols that cannot serve as a document loader."""
    is_subclass = isinstance(loader_cls, type) and issubclass(loader_cls, DocumentLoader)
    if not is_subclass and not callable(getattr(loader_cls, "load", None)):
        raise TypeError(
            f"Cannot register {loader_cls!r} for '{extension}': expected a "
            "DocumentLoader subclass with a callable load()"
        )


def _register(
    loader_cls: type[DocumentLoader], extensions: tuple[str, ...], override: bool = False
) -> None:
    """Validate and register ``loader_cls`` for each extension."""
    if not extensions:
        raise ValueError("At least one extension is required")

    for raw_extension in extensions:
        extension = _normalize_extension(raw_extension)
        _validate_loader_class(loader_cls, extension)
        existing = _registry.get(extension)
        if existing is not None and existing is not loader_cls:
            if not override:
                raise ValueError(
                    f"Extension '{extension}' is already registered to "
                    f"{existing.__name__}; pass override=True to replace it"
                )
            logger.warning(
                "Overriding loader for '%s': %s -> %s", extension,
                existing.__name__, loader_cls.__name__,
            )
        _registry[extension] = loader_cls


def register_loader(
    *extensions: str, override: bool = False
) -> Callable[[type[DocumentLoader]], type[DocumentLoader]]:
    """Register a loader class for one or more extensions.

    Usage::

        @register_loader(".csv")
        class CsvLoader(DocumentLoader): ...

    ``override=True`` replaces an existing registration instead of raising.
    """

    def decorator(loader_cls: type[DocumentLoader]) -> type[DocumentLoader]:
        _register(loader_cls, extensions, override=override)
        return loader_cls

    return decorator


def supported_extensions() -> list[str]:
    """Return registered extensions: lowercase, dot-prefixed, sorted."""
    return sorted(_registry)


def get_loader(path: str | Path, **loader_options: Any) -> DocumentLoader:
    """Resolve ``path`` to a new loader instance; options go to the constructor.

    Raises ``UnsupportedFormatError`` for unknown/missing extensions and for
    directory paths (recursive traversal belongs to step 2.5.1).
    """
    file_path = Path(path)

    if file_path.is_dir():
        raise UnsupportedFormatError(
            "Path is a directory; recursive directory traversal belongs to "
            "step 2.5.1",
            source_path=str(file_path),
        )

    extension = file_path.suffix.lower()
    loader_cls = _registry.get(extension)
    if loader_cls is None:
        raise UnsupportedFormatError(
            f"No loader registered for extension '{extension or '(none)'}' "
            f"(supported: {', '.join(supported_extensions())})",
            source_path=str(file_path),
        )

    logger.debug(
        "Resolved '%s' -> %s", extension, loader_cls.__name__,
        extra={"source_path": str(file_path), "extension": extension},
    )
    return loader_cls(**loader_options)


def _register_defaults() -> None:
    """Import the built-in loaders once and populate the registry."""
    from src.airag.ingestion.loaders.docx import DocxLoader
    from src.airag.ingestion.loaders.markdown import MarkdownLoader
    from src.airag.ingestion.loaders.pdf import PDFLoader
    from src.airag.ingestion.loaders.text import TextLoader

    _register(DocxLoader, (".docx",))
    _register(MarkdownLoader, (".md", ".markdown"))
    _register(PDFLoader, (".pdf",))
    _register(TextLoader, (".txt", ".text"))


_register_defaults()