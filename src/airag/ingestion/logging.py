"""
Structured logging for the ingestion pipeline (step 2.6.2).

``configure_logging`` attaches one JSON formatter to a logger (root by default)
and adds a correlation ID from a ``ContextVar`` to every record. Module code
keeps using ``logging.getLogger(__name__)``; configuration happens once at the
CLI boundary (and later at the API boundary in Phase 8). Logs go to stderr so
CLI JSON output on stdout stays machine-readable.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from typing import IO

from pythonjsonlogger.json import JsonFormatter

DEFAULT_SENSITIVE_KEYS = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "credential",
)

_HANDLER_NAME = "airag_json"
_correlation_id: ContextVar[str | None] = ContextVar(
    "airag_correlation_id", default=None
)


def new_correlation_id() -> str:
    """Return a new hex correlation ID for one CLI/API operation."""
    return uuid.uuid4().hex


def set_correlation_id(value: str | None = None) -> str:
    """Set the current correlation ID, generating one when omitted."""
    correlation_id = value or new_correlation_id()
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str | None:
    """Return the current correlation ID, if any."""
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID (mainly for tests)."""
    _correlation_id.set(None)


class CorrelationFilter(logging.Filter):
    """Attach the current correlation ID to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


class RedactionFilter(logging.Filter):
    """Mask sensitive ``extra`` fields on each log record."""

    def __init__(self, sensitive_keys: Sequence[str] = DEFAULT_SENSITIVE_KEYS) -> None:
        super().__init__()
        self.sensitive_keys = tuple(key.lower() for key in sensitive_keys)

    def _is_sensitive(self, name: str) -> bool:
        lowered = name.lower()
        return any(key in lowered for key in self.sensitive_keys)

    def filter(self, record: logging.LogRecord) -> bool:
        for name in list(record.__dict__):
            if self._is_sensitive(name):
                setattr(record, name, "***")
        if isinstance(record.msg, dict):
            record.msg = {
                key: "***" if self._is_sensitive(str(key)) else value
                for key, value in record.msg.items()
            }
        return True


def configure_logging(
    level: str = "INFO",
    *,
    stream: IO[str] | None = None,
    logger_name: str | None = None,
    redact_keys: Sequence[str] = DEFAULT_SENSITIVE_KEYS,
    force: bool = False,
) -> logging.Logger:
    """Configure JSON logging once for a logger.

    Args:
        level: Minimum level name (e.g. ``"INFO"``).
        stream: Output stream; defaults to ``sys.stderr``.
        logger_name: Named logger to configure; ``None`` configures root.
        redact_keys: Substrings marking sensitive ``extra`` fields.
        force: Remove other handlers on the chosen logger (use for tests).

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level.upper())

    for handler in list(logger.handlers):
        if handler.name == _HANDLER_NAME:
            logger.removeHandler(handler)
            handler.close()
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
                "correlation_id": "correlation_id",
            },
        )
    )
    handler.addFilter(CorrelationFilter())
    handler.addFilter(RedactionFilter(redact_keys))
    logger.addHandler(handler)
    if logger_name:
        logger.propagate = False
    return logger


__all__ = [
    "DEFAULT_SENSITIVE_KEYS",
    "CorrelationFilter",
    "RedactionFilter",
    "clear_correlation_id",
    "configure_logging",
    "get_correlation_id",
    "new_correlation_id",
    "set_correlation_id",
]
