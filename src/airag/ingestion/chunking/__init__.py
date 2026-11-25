"""
Chunking package for Phase 2 step 2.3.

The public contract (``Chunker``, ``ChunkConfig``, ``Chunk``) is defined in
``base.py``; ``fixed.py`` implements the fixed-size strategy (2.3.2).
Sentence-aware and semantic strategies (2.3.3-2.3.4) are still pending.
"""

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, Chunker
from src.airag.ingestion.chunking.fixed import FixedSizeChunker

__all__ = ["Chunk", "ChunkConfig", "Chunker", "FixedSizeChunker"]
