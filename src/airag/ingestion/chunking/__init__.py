"""
Chunking package for Phase 2 step 2.3.

The public contract (``Chunker``, ``ChunkConfig``, ``Chunk``) is defined in
``base.py``; ``fixed.py`` (2.3.2) and ``sentence.py`` (2.3.3) implement
concrete strategies. The semantic chunker (2.3.4) is still pending.
"""

from src.airag.ingestion.chunking.base import Chunk, ChunkConfig, Chunker
from src.airag.ingestion.chunking.fixed import FixedSizeChunker
from src.airag.ingestion.chunking.sentence import SentenceAwareChunker

__all__ = [
    "Chunk",
    "ChunkConfig",
    "Chunker",
    "FixedSizeChunker",
    "SentenceAwareChunker",
]
