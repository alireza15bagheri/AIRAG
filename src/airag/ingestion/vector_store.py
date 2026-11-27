"""
Vector store writer for Phase 2 step 2.4.2.

Writes pre-computed embeddings and chunk text to a ChromaDB-compatible
collection in batches. Upsert identity is ``<document hash>:<chunk index>``;
a ``chunk_hash`` of the chunk text is stored in metadata so re-ingesting
unchanged content is a no-op (idempotent write, step 2.4.4).

The ChromaDB client/collection is injectable, so unit tests use a fake
in-memory collection and never open a database. Embeddings are supplied by
the caller (e.g. ``EmbeddingService``); this module does not load a model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.exceptions import VectorStoreError

DEFAULT_COLLECTION = "airag_chunks"


@dataclass(slots=True)
class WriteResult:
    """Outcome of a vector store write.

    Attributes:
        written: New chunk IDs upserted.
        updated: Existing IDs upserted because their chunk text changed.
        skipped: Existing IDs whose chunk hash was unchanged.
        ids: IDs actually written or updated, in processing order.
    """

    written: int = 0
    updated: int = 0
    skipped: int = 0
    ids: list[str] = field(default_factory=list)


class VectorStoreWriter:
    """Batch-upsert chunks into a vector store, skipping unchanged content."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        batch_size: int = 64,
        persist_directory: str | Path | None = None,
        collection: Any | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the writer.

        Args:
            collection_name: ChromaDB collection to create or reuse.
            batch_size: Maximum number of chunks upserted per call.
            persist_directory: Directory for a persistent ChromaDB client;
                when omitted, an ephemeral in-memory client is used.
            collection: Pre-built ChromaDB-compatible collection (test hook).
            client: Pre-built ChromaDB-compatible client (test hook).
        """
        if batch_size <= 0:
            raise VectorStoreError("batch_size must be greater than zero")
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.persist_directory = persist_directory
        self._collection = collection
        self._client = client

    def write(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> WriteResult:
        """Upsert chunks with their embeddings, skipping unchanged content."""
        chunk_list = list(chunks)
        embedding_list = [list(vector) for vector in embeddings]
        if len(chunk_list) != len(embedding_list):
            raise VectorStoreError(
                f"Got {len(chunk_list)} chunks but {len(embedding_list)} embeddings"
            )
        if not chunk_list:
            return WriteResult()

        collection = self._get_collection()
        result = WriteResult()
        for offset in range(0, len(chunk_list), self.batch_size):
            batch = chunk_list[offset : offset + self.batch_size]
            batch_embeddings = embedding_list[offset : offset + self.batch_size]
            ids = [self.chunk_id(chunk) for chunk in batch]
            existing = self._existing_hashes(collection, ids)

            ids_to_write: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            batch_vectors: list[list[float]] = []
            new_count = 0
            updated_count = 0

            for chunk, vector, chunk_id in zip(batch, batch_embeddings, ids, strict=True):
                chunk_hash = self._chunk_hash(chunk.text)
                if existing.get(chunk_id) == chunk_hash:
                    result.skipped += 1
                    continue
                if chunk_id in existing:
                    updated_count += 1
                else:
                    new_count += 1
                ids_to_write.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(self._metadata(chunk, chunk_hash))
                batch_vectors.append(vector)

            if ids_to_write:
                try:
                    collection.upsert(
                        ids=ids_to_write,
                        embeddings=batch_vectors,
                        documents=documents,
                        metadatas=metadatas,
                    )
                except Exception as e:
                    raise VectorStoreError(f"Vector store upsert failed: {e}") from e
                result.written += new_count
                result.updated += updated_count
                result.ids.extend(ids_to_write)
        return result

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        try:
            return int(self._get_collection().count())
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Vector store count failed: {e}") from e

    @staticmethod
    def _chunk_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def document_id(chunk: Chunk) -> str:
        """Stable document identifier derived from chunk metadata or text."""
        return chunk.metadata.get("hash") or VectorStoreWriter._chunk_hash(chunk.text)

    @classmethod
    def chunk_id(cls, chunk: Chunk) -> str:
        """Stable vector-store ID for a chunk of a document."""
        return f"{cls.document_id(chunk)}:{chunk.chunk_index}"

    @classmethod
    def _metadata(cls, chunk: Chunk, chunk_hash: str) -> dict[str, Any]:
        raw = dict(chunk.metadata)
        raw.update(
            {
                "chunk_hash": chunk_hash,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "page": raw.get("page", 1),
            }
        )
        clean: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = json.dumps(value, default=str)
        return clean

    @staticmethod
    def _existing_hashes(collection: Any, ids: list[str]) -> dict[str, str | None]:
        try:
            response = collection.get(ids=ids, include=["metadatas"])
        except Exception as e:
            raise VectorStoreError(f"Vector store lookup failed: {e}") from e
        existing_ids = response.get("ids") or []
        metadatas = response.get("metadatas") or []
        return {
            chunk_id: (metadata or {}).get("chunk_hash")
            for chunk_id, metadata in zip(existing_ids, metadatas, strict=True)
        }

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client is not None:
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            except Exception as e:
                raise VectorStoreError(f"Could not open vector collection: {e}") from e
            return self._collection
        try:
            import chromadb
        except ImportError as e:
            raise VectorStoreError("chromadb is required for the default writer") from e
        try:
            if self.persist_directory is not None:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            raise VectorStoreError(f"Could not open vector collection: {e}") from e
        return self._collection
