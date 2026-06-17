"""Embed document chunks and persist them to the vector store.

Phase 1.2 connective tissue. The Phase 1.1 loader + chunker produce
`DocumentChunk`s *without* embeddings; the Phase 1 storage layer
(`ChromaVectorStore`) stores vectors that the caller supplies. This module
bridges the two: run the configured embedding provider over the chunk
texts, then upsert the (chunk, vector) pairs.

Design
------
* **The embedder is injectable.** `ingest_chunks` takes an optional
  `embedder`; when omitted it resolves the configured provider via
  `get_embedding_provider()`. Injection keeps the function testable
  without a real API key (tests pass a fake provider).
* **The store owns batching + metadata.** `ChromaVectorStore.upsert`
  already batches large inserts and flattens chunk metadata to Chroma's
  scalar-only column, so this layer stays thin: embed, then hand
  `(chunks, embeddings)` to the store.
* **Embeddings are returned** alongside the count so callers (e.g. the
  debug ingest endpoint) can echo them back for inspection without a
  second round-trip through the store.
"""

from __future__ import annotations

from src.models.document import DocumentChunk
from src.providers.base import EmbeddingProvider
from src.providers.factory import get_embedding_provider
from src.storage.schema import ChunkSchema
from src.storage.vector_store import ChromaVectorStore
from src.utils import get_logger

logger = get_logger(__name__)


async def ingest_chunks(
    chunks: list[DocumentChunk],
    vector_store: ChromaVectorStore,
    embedder: EmbeddingProvider | None = None,
) -> tuple[int, list[list[float]]]:
    """Embed `chunks` and upsert them into `vector_store`.

    Args:
        chunks: Chunks from the Phase 1.1 loader + chunker. An empty list is
            allowed and is a no-op (returns ``(0, [])``).
        vector_store: Destination store. Owns persistence, batching, and
            metadata flattening.
        embedder: Embedding provider to use. Defaults to the configured
            provider via :func:`get_embedding_provider`. Injectable so tests
            (and callers wanting a specific vendor) can supply their own.

    Returns:
        ``(stored_count, embeddings)`` — the number of chunks upserted and
        the embedding vector for each chunk, in input order.

    Raises:
        ValueError: if the embedder returns a vector count that doesn't match
            the chunk count. `ChromaVectorStore.upsert` raises the same error
            type for dimension/length mismatches, so callers can treat
            ``ValueError`` as "the ingest contract was violated".
    """
    if not chunks:
        logger.debug("ingest_chunks called with 0 chunks — no-op")
        return 0, []

    # Validate metadata up-front, before paying for embeddings. The store
    # validates again at upsert time (same code path), but doing it here
    # turns a bad chunk into a fast, chunk-scoped error instead of an
    # embed-then-fail. Non-scalar metadata surfaces as a `ValueError`.
    for chunk in chunks:
        try:
            ChunkSchema.chunk_to_chroma_format(chunk)
        except TypeError as e:
            raise ValueError(f"invalid metadata in chunk {chunk.chunk_id}: {e}") from e

    embedder = embedder or get_embedding_provider()
    model = getattr(embedder, "model_name", type(embedder).__name__)
    logger.info("Embedding %d chunks with %s ...", len(chunks), model)

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"embedding count mismatch: embedder returned {len(embeddings)} "
            f"vectors for {len(chunks)} chunks"
        )

    # The store flattens metadata and tacks on doc_id/chunk_index itself, so
    # we just hand it the chunks and their vectors in parallel order.
    await vector_store.upsert(chunks, embeddings)

    logger.info(
        "Ingested %d chunks (dim=%d) into the vector store",
        len(chunks),
        embedder.embedding_dimension,
    )
    return len(chunks), embeddings
