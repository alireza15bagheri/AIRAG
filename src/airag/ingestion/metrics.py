"""
In-process metrics for ingestion runs (step 2.6.3).

The registry is dependency-free and injectable: counters and latency histograms
are held in memory and exposed through ``snapshot()`` for a future FastAPI
endpoint. Pipelines own a registry instance, so each CLI run starts clean.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

FILES_PROCESSED = "files_processed_total"
FILES_SUCCEEDED = "files_succeeded_total"
FILES_FAILED = "files_failed_total"
FILES_SKIPPED = "files_skipped_total"
CHUNKS_CREATED = "chunks_created_total"
CHUNKS_WRITTEN = "chunks_written_total"
CHUNKS_UPDATED = "chunks_updated_total"
CHUNKS_SKIPPED = "chunks_skipped_total"
LOAD_SECONDS = "load_latency_seconds"
CHUNK_SECONDS = "chunk_latency_seconds"
EMBED_SECONDS = "embed_latency_seconds"
STORE_SECONDS = "store_latency_seconds"

COUNTER_NAMES = (
    FILES_PROCESSED,
    FILES_SUCCEEDED,
    FILES_FAILED,
    FILES_SKIPPED,
    CHUNKS_CREATED,
    CHUNKS_WRITTEN,
    CHUNKS_UPDATED,
    CHUNKS_SKIPPED,
)
HISTOGRAM_NAMES = (
    LOAD_SECONDS,
    CHUNK_SECONDS,
    EMBED_SECONDS,
    STORE_SECONDS,
)


@dataclass(frozen=True, slots=True)
class HistogramStats:
    """Summary of observed latency values, in seconds."""

    count: int
    total: float
    minimum: float
    maximum: float
    average: float
    p50: float
    p95: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable view for the future HTTP endpoint."""
        return {
            "count": self.count,
            "total": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "avg": self.average,
            "p50": self.p50,
            "p95": self.p95,
        }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


class MetricsRegistry:
    """Thread-safe in-memory counters and latency histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = dict.fromkeys(COUNTER_NAMES, 0)
        self._histograms: dict[str, list[float]] = {
            name: [] for name in HISTOGRAM_NAMES
        }

    def increment(self, name: str, value: int = 1) -> None:
        """Add a non-negative amount to a counter."""
        if value < 0:
            raise ValueError("counter increment must be non-negative")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_counter(self, name: str) -> int:
        """Return the current value of a counter."""
        with self._lock:
            return self._counters.get(name, 0)

    def observe(self, name: str, seconds: float) -> None:
        """Record one latency observation in seconds."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("latency observation must be a finite non-negative number")
        with self._lock:
            self._histograms.setdefault(name, []).append(seconds)

    def histogram(self, name: str) -> HistogramStats:
        """Return summary statistics for one histogram."""
        with self._lock:
            return self._stats(name)

    def _stats(self, name: str) -> HistogramStats:
        values = sorted(self._histograms.get(name, []))
        count = len(values)
        total = sum(values)
        return HistogramStats(
            count=count,
            total=total,
            minimum=values[0] if values else 0.0,
            maximum=values[-1] if values else 0.0,
            average=total / count if count else 0.0,
            p50=_percentile(values, 0.50),
            p95=_percentile(values, 0.95),
        )

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Time a block and record its duration even when it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable view of every known metric."""
        with self._lock:
            counters: dict[str, int] = {
                name: self._counters.get(name, 0) for name in COUNTER_NAMES
            }
            for name, value in self._counters.items():
                counters.setdefault(name, value)
            histograms: dict[str, dict[str, int | float]] = {
                name: self._stats(name).to_dict() for name in HISTOGRAM_NAMES
            }
            for name in self._histograms:
                histograms.setdefault(name, self._stats(name).to_dict())
        return {"counters": counters, "histograms": histograms}

    def reset(self) -> None:
        """Clear all observations so a new run starts from zero."""
        with self._lock:
            self._counters = dict.fromkeys(COUNTER_NAMES, 0)
            self._histograms = {name: [] for name in HISTOGRAM_NAMES}


__all__ = [
    "CHUNK_SECONDS",
    "CHUNKS_CREATED",
    "CHUNKS_SKIPPED",
    "CHUNKS_UPDATED",
    "CHUNKS_WRITTEN",
    "COUNTER_NAMES",
    "EMBED_SECONDS",
    "FILES_FAILED",
    "FILES_PROCESSED",
    "FILES_SKIPPED",
    "FILES_SUCCEEDED",
    "HISTOGRAM_NAMES",
    "LOAD_SECONDS",
    "STORE_SECONDS",
    "HistogramStats",
    "MetricsRegistry",
]
