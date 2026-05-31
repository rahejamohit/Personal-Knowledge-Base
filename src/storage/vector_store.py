"""Chroma vector store abstraction.

Phase 1 storage layer. The interface is intentionally small and provider-
agnostic so Phase 2's migration to Pinecone is a drop-in replacement:

* `upsert(chunks, embeddings)` — add or replace by `chunk_id`
* `search(query_embedding, top_k)` — nearest-neighbor lookup
* `delete(chunk_id)` — remove a single chunk
* `get_stats()` — collection-level telemetry

Design choices
--------------
* **Persistence is on by default** via `chromadb.PersistentClient(path=...)`,
  so closing and reopening the process keeps your index. The default
  path is read from settings (`pka_chroma_dir`) so the location is one
  config flip away from being moved to e.g. an EBS volume.
* **Embeddings are always supplied by the caller.** Chroma will auto-
  download a `DefaultEmbeddingFunction` (a ~80 MB ONNX model) if we let
  it — so we pin `embedding_function=None` on the collection and refuse
  to embed text ourselves. The provider layer (`src/providers/`) owns
  that responsibility.
* **Async wrappers.** Chroma's SDK is synchronous, but we expose `async
  def` methods so the interface matches what Phase 2 needs (Pinecone +
  Qdrant both have async clients). The methods are sync-internally
  today; switching to `asyncio.to_thread` if Chroma calls ever become a
  hot loop is a one-line change.
* **Distance → similarity.** Chroma returns L2/cosine *distance*; we
  convert to a `[0, 1]` similarity for `RetrievedDoc.score` so the rest
  of the system reads "higher = better" consistently.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.models.conversation import RetrievedDoc
from src.models.document import DocumentChunk
from src.utils import get_logger

logger = get_logger(__name__)

# Soft cap on a single Chroma `add` call. Large inserts work but allocate
# linearly in memory; chunking keeps peak RSS predictable on big ingests.
_BATCH_SIZE = 256


class ChromaVectorStore:
    """Abstraction over a Chroma collection.

    Construction is cheap (no network). Heavy work happens on first
    `add`/`query`, when Chroma loads / builds its HNSW index.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str | Path = ".chroma",
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # `anonymized_telemetry=False` keeps Chroma from phoning home — we
        # don't want unsolicited network traffic from a "local" store.
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # We always supply embeddings ourselves — see module docstring.
            embedding_function=None,
        )
        logger.info(
            "ChromaVectorStore ready (collection=%r, persist_dir=%s, count=%d)",
            collection_name,
            self._persist_dir,
            self._collection.count(),
        )

    # ─── Write path ─────────────────────────────────────────────────

    async def upsert(
        self,
        doc_chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add or replace chunks in the index.

        Uses `collection.upsert` so re-ingesting a document overwrites the
        previous version of each chunk (idempotent) rather than producing
        duplicate rows.

        Raises
        ------
        ValueError
            * `doc_chunks` and `embeddings` have different lengths.
            * embedding vectors have inconsistent dimensions.
        """
        if len(doc_chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(doc_chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        if not doc_chunks:
            logger.debug("upsert called with 0 chunks — no-op")
            return

        # Sanity check: all embedding vectors must be the same length, or
        # Chroma will silently break similarity scoring.
        dim = len(embeddings[0])
        for i, vec in enumerate(embeddings):
            if len(vec) != dim:
                raise ValueError(
                    f"inconsistent embedding dim at index {i}: got {len(vec)}, expected {dim}"
                )

        # Batch the upsert so memory stays predictable on big ingests.
        for start in range(0, len(doc_chunks), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            batch_chunks = doc_chunks[start:end]
            batch_embeddings = embeddings[start:end]
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[c.chunk_id for c in batch_chunks],
                documents=[c.text for c in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[self._chunk_metadata(c) for c in batch_chunks],
            )
        logger.info(
            "Upserted %d chunks (dim=%d) into collection=%r",
            len(doc_chunks),
            dim,
            self._collection_name,
        )

    # ─── Read path ──────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """Return the top-`top_k` chunks most similar to `query_embedding`.

        Results are ordered by descending similarity. If the collection is
        empty, returns an empty list rather than raising — callers can
        treat that as "no docs ingested yet".
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns each field as a list-of-lists (one inner list per
        # query). We always send one query, so we always read index 0.
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedDoc] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True)
        ):
            metadata = metadata or {}
            retrieved.append(
                RetrievedDoc(
                    chunk_id=chunk_id,
                    # We persist `doc_id` in metadata at upsert time so we
                    # can recover the parent doc on read.
                    doc_id=str(metadata.get("doc_id", "")),
                    source=str(metadata.get("source", "unknown")),
                    text=text,
                    score=_distance_to_similarity(distance),
                    rank=rank,
                    metadata=dict(metadata),
                )
            )
        logger.info(
            "Searched collection=%r top_k=%d returned=%d",
            self._collection_name,
            top_k,
            len(retrieved),
        )
        return retrieved

    # ─── Mutation / telemetry ───────────────────────────────────────

    async def delete(self, chunk_id: str) -> None:
        """Delete a single chunk by ID. Silent no-op if it doesn't exist."""
        await asyncio.to_thread(self._collection.delete, ids=[chunk_id])
        logger.info("Deleted chunk_id=%r from collection=%r", chunk_id, self._collection_name)

    async def get_stats(self) -> dict[str, Any]:
        """Collection-level counters. Cheap; safe to poll for health checks."""
        return {
            "total_chunks": self._collection.count(),
            "collection_name": self._collection_name,
            "persist_dir": str(self._persist_dir),
        }

    # ─── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
        """Flatten a chunk's metadata into something Chroma will accept.

        Chroma's metadata column only stores scalars (str/int/float/bool),
        so we ride on `DocumentMetadata.to_chroma_metadata()` and tack on
        the fields the search step needs (`doc_id`, `chunk_index`).
        """
        meta = chunk.metadata.to_chroma_metadata()
        meta["doc_id"] = chunk.doc_id
        meta["chunk_index"] = chunk.chunk_index
        return meta


def _distance_to_similarity(distance: float) -> float:
    """Map Chroma's cosine distance to a `[0, 1]` similarity score.

    For normalized embeddings, `cosine_distance ∈ [0, 2]` and
    `similarity = 1 - distance` lands in `[-1, 1]`. We clamp to `[0, 1]`
    so consumers can treat the score as a confidence percentage without
    worrying about negative numbers for very-dissimilar pairs.
    """
    similarity = 1.0 - float(distance)
    return max(0.0, min(1.0, similarity))
