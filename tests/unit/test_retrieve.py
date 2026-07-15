"""Unit tests for `src.agent.tools.retrieve()` — the Phase 1.3 RAG retriever.

Hermetic by construction: every test injects the `FakeEmbedder` (the
deterministic lexical bag-of-words vectorizer from `conftest`) and a
throwaway Chroma store under `tmp_path`, so no network call or real
on-disk index is ever touched. `FakeEmbedder` is lexically meaningful —
texts that share words land closer in cosine space — which is exactly
enough signal to test that semantic + BM25 reranking surfaces the right
chunk without a real embedding model.

`asyncio_mode = "auto"` (see `pyproject.toml`) auto-detects the `async
def` tests and async fixtures here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.tools import retrieve
from src.models.conversation import RetrievedDoc
from src.models.document import DocumentChunk, DocumentMetadata
from src.providers.base import EmbeddingProvider
from src.storage.ingest import ingest_chunks
from src.storage.vector_store import ChromaVectorStore
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.asyncio


# ─── Helpers / fixtures ──────────────────────────────────────────────────


def _chunk(chunk_id: str, text: str, *, doc_id: str = "doc", source: str = "doc.md") -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        metadata=DocumentMetadata(source=source),
    )


@pytest.fixture
async def populated_store(tmp_path: Path, fake_embedder: FakeEmbedder) -> ChromaVectorStore:
    """A throwaway store seeded with two RAG chunks + one unrelated chunk."""
    store = ChromaVectorStore(collection_name="retrieve_test", persist_dir=tmp_path / "chroma")
    chunks = [
        _chunk(
            "rag_0",
            "RAG retrieval augmented generation combines retrieval with generation",
            doc_id="rag",
            source="rag.md",
        ),
        _chunk(
            "emb_0",
            "Embeddings are dense vector representations that capture semantic meaning",
            doc_id="rag",
            source="rag.md",
        ),
        _chunk(
            "weather_0",
            "The weather today is sunny with temperatures reaching 25 degrees",
            doc_id="weather",
            source="weather.md",
        ),
    ]
    await ingest_chunks(chunks, store, embedder=fake_embedder)
    return store


# ─── Input validation (no store / embedder touched) ──────────────────────


class TestRetrieveValidation:
    async def test_empty_query_returns_empty(self) -> None:
        # Returns before constructing any embedder/store, so no injection
        # is needed — and crucially, no network call is attempted.
        assert await retrieve("") == []

    async def test_whitespace_query_returns_empty(self) -> None:
        assert await retrieve("   \n\t ") == []

    async def test_out_of_range_top_k_is_clamped_not_raised(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # top_k=99 clamps to 5; the store only holds 3 chunks, so we get 3.
        results = await retrieve(
            "RAG", top_k=99, vector_store=populated_store, embedder=fake_embedder
        )
        assert len(results) <= 5
        assert isinstance(results, list)

    async def test_zero_top_k_is_clamped(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "RAG", top_k=0, vector_store=populated_store, embedder=fake_embedder
        )
        # Clamped to the default of 5, so it doesn't return an empty list
        # the way `top_k=0` literally would.
        assert len(results) > 0


# ─── Result shape & ordering ──────────────────────────────────────────────


class TestRetrieveResults:
    async def test_returns_list(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "What is RAG?", vector_store=populated_store, embedder=fake_embedder
        )
        assert isinstance(results, list)
        assert len(results) > 0

    async def test_respects_top_k(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "RAG embeddings", top_k=2, vector_store=populated_store, embedder=fake_embedder
        )
        assert len(results) <= 2

    async def test_results_are_retrieved_docs(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "retrieval augmented generation",
            vector_store=populated_store,
            embedder=fake_embedder,
        )
        assert results
        for doc in results:
            assert isinstance(doc, RetrievedDoc)
            assert doc.chunk_id and doc.text and doc.source
            assert isinstance(doc.score, float)
            assert 0.0 <= doc.score <= 1.0, f"score {doc.score} out of [0, 1]"

    async def test_results_sorted_by_score_descending(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "retrieval generation embeddings",
            vector_store=populated_store,
            embedder=fake_embedder,
        )
        scores = [d.score for d in results]
        assert scores == sorted(scores, reverse=True)

    async def test_rank_reflects_final_order(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # `.rank` must be reassigned post-rerank, not left as Chroma's order.
        results = await retrieve(
            "retrieval generation embeddings",
            vector_store=populated_store,
            embedder=fake_embedder,
        )
        assert [d.rank for d in results] == list(range(len(results)))


# ─── Relevance (reranking actually helps) ─────────────────────────────────


class TestRetrieveRelevance:
    async def test_finds_relevant_chunk_over_irrelevant(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        results = await retrieve(
            "retrieval augmented generation",
            vector_store=populated_store,
            embedder=fake_embedder,
        )
        # The RAG chunk should outrank the unrelated weather chunk.
        top = results[0]
        assert "weather" not in top.text.lower()
        assert "retrieval" in top.text.lower() or "generation" in top.text.lower()

    async def test_keyword_match_surfaces(
        self, populated_store: ChromaVectorStore, fake_embedder: FakeEmbedder
    ) -> None:
        # A query dominated by "embeddings ... semantic" keywords should put
        # the embeddings chunk first thanks to the BM25 component.
        results = await retrieve(
            "embeddings semantic vector representations",
            top_k=1,
            vector_store=populated_store,
            embedder=fake_embedder,
        )
        assert results[0].chunk_id == "emb_0"


# ─── Graceful failure ─────────────────────────────────────────────────────


class TestRetrieveFailureModes:
    async def test_empty_index_returns_empty(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        empty = ChromaVectorStore(collection_name="empty", persist_dir=tmp_path / "chroma")
        assert await retrieve("anything", vector_store=empty, embedder=fake_embedder) == []

    async def test_embedder_failure_returns_empty_not_raises(
        self, populated_store: ChromaVectorStore
    ) -> None:
        class BoomEmbedder(EmbeddingProvider):
            def __init__(self) -> None: ...

            def embed_query(self, text: str) -> list[float]:
                raise RuntimeError("provider unreachable")

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("provider unreachable")

            @property
            def embedding_dimension(self) -> int:
                return 64

        # The agent loop must survive a dead embedding provider.
        results = await retrieve("RAG", vector_store=populated_store, embedder=BoomEmbedder())
        assert results == []
