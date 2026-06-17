"""Integration tests for the embed + store pipeline (`ingest_chunks`).

`tests/unit/test_vector_store.py` covers `ChromaVectorStore` in isolation.
This module tests the layer above it — `src.storage.ingest.ingest_chunks` —
which embeds Phase 1.1 chunks and upserts them into the store.

Hermetic by default: the bulk of these inject the `FakeEmbedder` from
`conftest` (a deterministic lexical bag-of-words vectorizer), so they run
anywhere with no API key and still exercise real ranking. A couple of
tests at the bottom hit the *configured* embedding provider and are marked
`integration`; they self-skip when no provider is reachable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.models.document import DocumentChunk, DocumentMetadata
from src.providers.base import EmbeddingProvider
from src.providers.factory import get_embedding_provider
from src.storage.ingest import ingest_chunks
from src.storage.vector_store import ChromaVectorStore
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.asyncio


# ─── Helpers / fixtures ───────────────────────────────────────────────────


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc_test",
    metadata: DocumentMetadata | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=int(chunk_id.split("_")[-1]) if chunk_id.split("_")[-1].isdigit() else 0,
        text=text,
        metadata=metadata or DocumentMetadata(source=f"{doc_id}.md"),
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name="ingest_test", persist_dir=tmp_path / "chroma")


# ─── Pipeline basics ──────────────────────────────────────────────────────


class TestIngestChunks:
    async def test_single_chunk(self, store: ChromaVectorStore, fake_embedder: FakeEmbedder) -> None:
        chunk = _chunk("c_0", "Semantic search uses vector embeddings for similarity")
        count, embeddings = await ingest_chunks([chunk], store, embedder=fake_embedder)

        assert count == 1
        assert len(embeddings) == 1
        assert len(embeddings[0]) == fake_embedder.embedding_dimension
        assert (await store.get_stats())["total_chunks"] == 1

        # And it's retrievable.
        results = await store.search(fake_embedder.embed_query(chunk.text), top_k=1)
        assert results[0].chunk_id == "c_0"

    async def test_batch(self, store: ChromaVectorStore, fake_embedder: FakeEmbedder) -> None:
        chunks = [
            _chunk(f"c_{i}", f"Chunk {i} about RAG, embeddings, and vector databases")
            for i in range(10)
        ]
        count, embeddings = await ingest_chunks(chunks, store, embedder=fake_embedder)

        assert count == 10
        assert len(embeddings) == 10
        assert (await store.get_stats())["total_chunks"] == 10

    async def test_empty_list_is_noop(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        count, embeddings = await ingest_chunks([], store, embedder=fake_embedder)
        assert count == 0
        assert embeddings == []
        assert (await store.get_stats())["total_chunks"] == 0

    async def test_reingest_same_id_is_idempotent(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        await ingest_chunks([_chunk("c_0", "Original text")], store, embedder=fake_embedder)
        await ingest_chunks([_chunk("c_0", "Updated text")], store, embedder=fake_embedder)
        # Upsert by chunk_id → still one row, not two.
        assert (await store.get_stats())["total_chunks"] == 1

    async def test_embedding_count_mismatch_raises(self, store: ChromaVectorStore) -> None:
        class _BadCountEmbedder(EmbeddingProvider):
            def __init__(self) -> None:
                pass

            def embed_query(self, text: str) -> list[float]:
                return [0.0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return []  # contract violation: 0 vectors for N chunks

            @property
            def embedding_dimension(self) -> int:
                return 1

        with pytest.raises(ValueError, match="embedding count mismatch"):
            await ingest_chunks([_chunk("c_0", "x")], store, embedder=_BadCountEmbedder())


# ─── Metadata round-trip ──────────────────────────────────────────────────


class TestMetadata:
    async def test_metadata_preserved(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        chunk = _chunk(
            "meta_0",
            "Testing metadata preservation",
            metadata=DocumentMetadata(
                source="metadata_test.md",
                title="Test Document",
                page=5,
                section="Testing",
            ),
        )
        await ingest_chunks([chunk], store, embedder=fake_embedder)

        results = await store.search(fake_embedder.embed_query("metadata"), top_k=1)
        meta = results[0].metadata
        assert meta["source"] == "metadata_test.md"
        assert meta["title"] == "Test Document"
        assert meta["page"] == 5  # int round-trips as int (Chroma allows scalars)
        assert meta["section"] == "Testing"
        # The store tacks doc_id / chunk_index on for retrieval.
        assert meta["doc_id"] == "doc_test"
        assert meta["chunk_index"] == 0

    async def test_datetime_serialized_to_iso_string(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # `ingested_at` defaults to a tz-aware datetime; it must reach Chroma
        # as an ISO string, since Chroma only stores scalars.
        chunk = _chunk("dt_0", "Testing datetime conversion")
        await ingest_chunks([chunk], store, embedder=fake_embedder)

        results = await store.search(fake_embedder.embed_query("datetime"), top_k=1)
        ingested_at = results[0].metadata["ingested_at"]
        assert isinstance(ingested_at, str)
        assert "T" in ingested_at  # ISO 8601


# ─── Schema validation through the pipeline ───────────────────────────────


class TestSchemaValidation:
    async def test_datetime_metadata_ingests_as_iso(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # A tz-aware datetime in metadata must not raise — ChunkSchema coerces
        # it to an ISO string on the way into Chroma.
        meta = DocumentMetadata(
            source="schema_test.md",
            ingested_at=datetime(2026, 6, 7, tzinfo=UTC),
        )
        chunk = _chunk("schema_0", "Testing schema validation in ingestion", metadata=meta)

        count, _ = await ingest_chunks([chunk], store, embedder=fake_embedder)
        assert count == 1

        results = await store.search(fake_embedder.embed_query("schema"), top_k=1)
        ingested_at = results[0].metadata["ingested_at"]
        assert isinstance(ingested_at, str)
        assert "T" in ingested_at

    async def test_non_scalar_metadata_fails_fast(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # An extra non-scalar field (list) is rejected as a ValueError naming
        # the chunk — and nothing is embedded or stored.
        meta = DocumentMetadata(source="bad.md", tags=["a", "b"])
        chunk = _chunk("bad_0", "chunk with a non-scalar metadata field", metadata=meta)

        with pytest.raises(ValueError, match="invalid metadata in chunk bad_0"):
            await ingest_chunks([chunk], store, embedder=fake_embedder)

        # Fail-fast: the store stays empty (validation ran before upsert).
        assert (await store.get_stats())["total_chunks"] == 0


# ─── Ranking + persistence ────────────────────────────────────────────────


class TestRankingAndPersistence:
    async def test_relevant_chunk_ranks_first(
        self, store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # The FakeEmbedder is lexical, so shared vocabulary → higher cosine.
        chunks = [
            _chunk(
                "rag_0",
                "RAG retrieval augmented generation uses vector embeddings for semantic search",
                doc_id="rag_doc",
            ),
            _chunk(
                "weather_0",
                "The weather today is sunny with temperatures reaching warm levels",
                doc_id="weather_doc",
            ),
        ]
        await ingest_chunks(chunks, store, embedder=fake_embedder)

        query = fake_embedder.embed_query("retrieval augmented generation vector embeddings")
        results = await store.search(query, top_k=2)
        assert results[0].chunk_id == "rag_0", "expected the RAG chunk to rank first"
        assert results[0].score >= results[1].score

    async def test_data_persists_across_store_instances(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        persist = tmp_path / "persist"
        store1 = ChromaVectorStore(collection_name="ingest_test", persist_dir=persist)
        await ingest_chunks(
            [_chunk("c_0", "persisted chunk"), _chunk("c_1", "second chunk")],
            store1,
            embedder=fake_embedder,
        )
        del store1

        # Re-open from disk — the ingested chunks survive.
        store2 = ChromaVectorStore(collection_name="ingest_test", persist_dir=persist)
        assert (await store2.get_stats())["total_chunks"] == 2
        results = await store2.search(fake_embedder.embed_query("persisted chunk"), top_k=2)
        assert {r.chunk_id for r in results} == {"c_0", "c_1"}


# ─── Real embedding provider (configured backend) ─────────────────────────


def _real_embedder_or_skip() -> EmbeddingProvider:
    """Return the configured embedding provider, or skip if unreachable."""
    try:
        embedder = get_embedding_provider()
        embedder.embed_query("warmup")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no embedding provider available: {e}")
    return embedder


@pytest.mark.integration
class TestRealEmbeddingProvider:
    async def test_ingest_and_semantic_search_with_real_embeddings(
        self, store: ChromaVectorStore
    ) -> None:
        embedder = _real_embedder_or_skip()
        chunks = [
            _chunk(
                "rag_0",
                "Retrieval-augmented generation grounds an LLM in retrieved documents.",
                doc_id="rag_doc",
            ),
            _chunk(
                "cooking_0",
                "To bake sourdough you need flour, water, salt, and a starter culture.",
                doc_id="cooking_doc",
            ),
        ]
        count, embeddings = await ingest_chunks(chunks, store, embedder=embedder)
        assert count == 2
        assert len(embeddings[0]) == embedder.embedding_dimension

        query = embedder.embed_query("How does RAG use retrieval to ground a model?")
        results = await store.search(query, top_k=2)
        assert results[0].chunk_id == "rag_0"
