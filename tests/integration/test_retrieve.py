"""Integration tests for `src.agent.tools.retrieve()` with a real embedder.

`tests/unit/test_retrieve.py` covers `retrieve` hermetically with the
`FakeEmbedder`. This module exercises the same path against the *configured*
embedding provider (Ollama / OpenAI / Gemini) and a real on-disk Chroma
store, so it proves the wiring works with a genuine embedding model — not
just the lexical fake.

These are marked `integration` and self-skip when no provider is reachable
(no Ollama server, no API key), mirroring `test_vector_store.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.tools import retrieve
from src.models.document import DocumentChunk, DocumentMetadata
from src.providers.base import EmbeddingProvider
from src.providers.factory import get_embedding_provider
from src.storage.ingest import ingest_chunks
from src.storage.vector_store import ChromaVectorStore

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _real_embedder_or_skip() -> EmbeddingProvider:
    """Return the configured embedding provider, or skip if unreachable."""
    try:
        embedder = get_embedding_provider()
        embedder.embed_query("warmup")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no embedding provider available: {e}")
    return embedder


def _chunk(chunk_id: str, text: str, *, doc_id: str = "doc", source: str = "doc.md") -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        metadata=DocumentMetadata(source=source),
    )


class TestRetrieveWithRealEmbeddings:
    async def test_finds_semantically_relevant_chunk(self, tmp_path: Path) -> None:
        embedder = _real_embedder_or_skip()
        store = ChromaVectorStore(collection_name="retrieve_int", persist_dir=tmp_path / "chroma")
        await ingest_chunks(
            [
                _chunk(
                    "rag_0",
                    "Retrieval-augmented generation grounds an LLM's answer in retrieved documents.",
                    doc_id="rag",
                    source="rag.md",
                ),
                _chunk(
                    "cook_0",
                    "Preheat the oven to 200 degrees and bake the bread for thirty minutes.",
                    doc_id="cook",
                    source="cooking.md",
                ),
            ],
            store,
            embedder=embedder,
        )

        results = await retrieve(
            "How does RAG use retrieved documents?",
            top_k=2,
            vector_store=store,
            embedder=embedder,
        )

        assert results, "real embeddings returned no results"
        # Semantic match should beat the unrelated cooking chunk.
        assert results[0].chunk_id == "rag_0"
        assert all(0.0 <= r.score <= 1.0 for r in results)

    async def test_respects_top_k(self, tmp_path: Path) -> None:
        embedder = _real_embedder_or_skip()
        store = ChromaVectorStore(collection_name="retrieve_int", persist_dir=tmp_path / "chroma")
        await ingest_chunks(
            [_chunk(f"c_{i}", f"Document number {i} about embeddings and retrieval") for i in range(5)],
            store,
            embedder=embedder,
        )

        results = await retrieve("embeddings retrieval", top_k=3, vector_store=store, embedder=embedder)
        assert len(results) <= 3
