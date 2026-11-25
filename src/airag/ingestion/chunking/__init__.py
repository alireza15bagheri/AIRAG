"""
Chunking package for Phase 2 step 2.3.

The public contract (``Chunker``, ``ChunkConfig``, ``Chunk``) is defined in
``base.py``. Concrete strategies will be added in steps 2.3.2-2.3.4.
"""

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, Chunker

__all__ = ["Chunk", "ChunkConfig", "Chunker"]
