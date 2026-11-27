"""
Dead letter queue for failed ingestion files (step 2.6.4).

A small JSONL-backed queue records failed sources so they can be reviewed or
retried later. Entries are deduplicated by source, error type, content hash and
message, which keeps repeated failed retries from multiplying records.
"""

from __future__ import annotations

import json
import threading
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DLQ_FILENAME = ".airag_dlq.jsonl"


@dataclass(frozen=True, slots=True)
class DLQEntry:
    """One failed file recorded for review or retry.

    Attributes:
        source: Path of the source file that failed.
        error: Human-readable error message.
        error_type: Exception class name or coarse category. Derived from the
            error message when left at the default.
        timestamp: ISO-8601 UTC time of the failure.
        content_hash: Document hash when available, otherwise empty.
        chunks: Number of chunks produced before the failure, if any.
    """

    source: str
    error: str
    error_type: str = "Error"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str = ""
    chunks: int = 0

    def __post_init__(self) -> None:
        if self.error_type == "Error" and ":" in self.error:
            candidate = self.error.split(":", 1)[0].strip()
            if candidate:
                object.__setattr__(self, "error_type", candidate)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DLQEntry | None:
        """Build an entry from a stored record, or ``None`` when malformed."""
        source = data.get("source")
        error = data.get("error", "")
        if not isinstance(source, str) or not isinstance(error, str):
            return None
        return cls(
            source=source,
            error=error,
            error_type=str(data.get("error_type") or "Error"),
            timestamp=str(
                data.get("timestamp") or datetime.now(timezone.utc).isoformat()
            ),
            content_hash=str(data.get("content_hash") or ""),
            chunks=int(data.get("chunks") or 0),
        )


class DeadLetterQueue:
    """Append-only JSONL queue with deduplication and retry support."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(DEFAULT_DLQ_FILENAME)
        self._lock = threading.Lock()

    def _read_unlocked(self) -> list[DLQEntry]:
        if not self.path.is_file():
            return []
        entries: list[DLQEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict):
                entry = DLQEntry.from_dict(data)
                if entry is not None:
                    entries.append(entry)
        return entries

    @staticmethod
    def _key(entry: DLQEntry) -> tuple[str, str, str, str]:
        return (entry.source, entry.error_type, entry.content_hash, entry.error)

    def append(self, entry: DLQEntry) -> bool:
        """Append an entry unless an identical failure is already recorded."""
        with self._lock:
            keys = {self._key(existing) for existing in self._read_unlocked()}
            if self._key(entry) in keys:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict()) + "\n")
            return True

    def entries(self) -> list[DLQEntry]:
        """Return all recorded failures in insertion order."""
        with self._lock:
            return self._read_unlocked()

    def pending_sources(self) -> list[str]:
        """Return unique failed source paths in first-seen order."""
        seen: set[str] = set()
        sources: list[str] = []
        for entry in self.entries():
            if entry.source not in seen:
                seen.add(entry.source)
                sources.append(entry.source)
        return sources

    def remove(self, source: str) -> int:
        """Remove every entry for a source; return the number removed."""
        with self._lock:
            entries = self._read_unlocked()
            kept = [entry for entry in entries if entry.source != source]
            removed = len(entries) - len(kept)
            if removed:
                self._write_unlocked(kept)
            return removed

    def clear(self) -> None:
        """Delete the queue file entirely."""
        with self._lock:
            with suppress(FileNotFoundError):
                self.path.unlink()

    def _write_unlocked(self, entries: list[DLQEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(
            "".join(json.dumps(entry.to_dict()) + "\n" for entry in entries),
            encoding="utf-8",
        )
        temp.replace(self.path)


__all__ = ["DEFAULT_DLQ_FILENAME", "DLQEntry", "DeadLetterQueue"]
