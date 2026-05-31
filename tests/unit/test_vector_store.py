"""Unit tests for `src.storage.vector_store.ChromaVectorStore`.

These hit a real (local, file-backed) Chroma — chromadb is a project
dependency so it's always installable. Each test uses its own `tmp_path`
so they're hermetic and parallel-safe.

Embedding vectors are tiny (8-d) hand-rolled fixtures, not real
embeddings — we're testing the *store*, not the embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.document import DocumentChunk, DocumentMetadata
from src.storage.vector_store import ChromaVectorStore, _distance_to_similarity

pytestmark = pytest.mark.asyncio


# ─── Helpers ──────────────────────────────────────────────────────────────


def _chunk(chunk_id: str, text: str, *, doc_id: str = "doc_test", page: int = 0) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=int(chunk_id.split("_")[-1]) if "_" in chunk_id else 0,
        text=text,
        metadata=DocumentMetadata(source=f"{doc_id}.md", page=page),
    )


def _vec(*values: float) -> list[float]:
    """Pad/truncate to 8 dims so all test embeddings share a dimension."""
    target = 8
    if len(values) >= target:
        return list(values[:target])
    return list(values) + [0.0] * (target - len(values))


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name="test", persist_dir=tmp_path / "chroma")


# ─── Distance → similarity conversion ────────────────────────────────────


class TestDistanceToSimilarity:
    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (0.0, 1.0),       # identical
            (1.0, 0.0),       # orthogonal (normalized vectors)
            (0.5, 0.5),
            (-0.1, 1.0),      # clamped — Chroma occasionally returns tiny <0
            (2.0, 0.0),       # clamped — fully opposite
            (10.0, 0.0),      # clamped — out-of-range guard
        ],
    )
    def test_clamps_to_unit_interval(self, distance: float, expected: float) -> None:
        assert _distance_to_similarity(distance) == pytest.approx(expected)


# ─── Initialization ──────────────────────────────────────────────────────


class TestInit:
    async def test_creates_collection_and_persist_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "ch"
        s = ChromaVectorStore(collection_name="docs", persist_dir=path)
        assert path.is_dir()
        stats = await s.get_stats()
        assert stats == {
            "total_chunks": 0,
            "collection_name": "docs",
            "persist_dir": str(path),
        }

    async def test_idempotent_open(self, tmp_path: Path) -> None:
        # Two stores pointing at the same persist dir should share the
        # collection without complaining.
        s1 = ChromaVectorStore(persist_dir=tmp_path)
        await s1.upsert([_chunk("c_0", "hello")], [_vec(1.0)])
        s2 = ChromaVectorStore(persist_dir=tmp_path)
        stats = await s2.get_stats()
        assert stats["total_chunks"] == 1


# ─── Upsert ──────────────────────────────────────────────────────────────


class TestUpsert:
    async def test_basic_upsert(self, store: ChromaVectorStore) -> None:
        await store.upsert(
            [_chunk("c_0", "alpha"), _chunk("c_1", "beta")],
            [_vec(1.0), _vec(0.0, 1.0)],
        )
        assert (await store.get_stats())["total_chunks"] == 2

    async def test_count_mismatch_raises(self, store: ChromaVectorStore) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            await store.upsert([_chunk("c_0", "alpha")], [_vec(1.0), _vec(0.0, 1.0)])

    async def test_inconsistent_dim_raises(self, store: ChromaVectorStore) -> None:
        with pytest.raises(ValueError, match="inconsistent embedding dim"):
            await store.upsert(
                [_chunk("c_0", "a"), _chunk("c_1", "b")],
                [[1.0, 0.0, 0.0], [1.0, 0.0]],  # different lengths
            )

    async def test_empty_input_is_noop(self, store: ChromaVectorStore) -> None:
        await store.upsert([], [])  # should not raise
        assert (await store.get_stats())["total_chunks"] == 0

    async def test_upsert_is_idempotent(self, store: ChromaVectorStore) -> None:
        # Same chunk_id, different text → the second wins, count stays 1.
        await store.upsert([_chunk("c_0", "v1")], [_vec(1.0)])
        await store.upsert([_chunk("c_0", "v2")], [_vec(1.0)])
        stats = await store.get_stats()
        assert stats["total_chunks"] == 1

    async def test_batched_upsert_handles_300_chunks(self, store: ChromaVectorStore) -> None:
        # 300 > the internal _BATCH_SIZE of 256, so this exercises the loop.
        n = 300
        chunks = [_chunk(f"c_{i}", f"text {i}") for i in range(n)]
        embeddings = [_vec(float(i % 7), float(i % 5)) for i in range(n)]
        await store.upsert(chunks, embeddings)
        assert (await store.get_stats())["total_chunks"] == n


# ─── Search ──────────────────────────────────────────────────────────────


class TestSearch:
    async def test_top_k_returns_at_most_k(self, store: ChromaVectorStore) -> None:
        for i in range(5):
            await store.upsert(
                [_chunk(f"c_{i}", f"text {i}")],
                [_vec(float(i))],
            )
        results = await store.search(_vec(2.0), top_k=3)
        assert len(results) == 3

    async def test_empty_collection_returns_empty_list(self, store: ChromaVectorStore) -> None:
        # Searching an empty collection should be a clean no-op, not raise.
        results = await store.search(_vec(1.0, 2.0, 3.0), top_k=5)
        assert results == []

    async def test_results_have_required_fields(self, store: ChromaVectorStore) -> None:
        await store.upsert(
            [_chunk("c_0", "alpha"), _chunk("c_1", "beta")],
            [_vec(1.0, 0.0), _vec(0.0, 1.0)],
        )
        results = await store.search(_vec(1.0, 0.0), top_k=2)
        assert results, "expected at least one result"
        first = results[0]
        assert first.chunk_id in {"c_0", "c_1"}
        assert first.doc_id == "doc_test"
        assert first.source.endswith(".md")
        assert 0.0 <= first.score <= 1.0
        assert first.rank == 0
        # `doc_id` and chunk metadata land in the metadata dict, too.
        assert "doc_id" in first.metadata

    async def test_ranks_are_zero_indexed_and_sorted(self, store: ChromaVectorStore) -> None:
        await store.upsert(
            [_chunk(f"c_{i}", f"x{i}") for i in range(4)],
            [_vec(1.0, 0.0), _vec(0.9, 0.1), _vec(0.5, 0.5), _vec(0.0, 1.0)],
        )
        results = await store.search(_vec(1.0, 0.0), top_k=4)
        assert [r.rank for r in results] == [0, 1, 2, 3]
        # Closer vectors should have higher scores than farther ones.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"not sorted desc: {scores}"

    async def test_top_k_zero_rejected(self, store: ChromaVectorStore) -> None:
        with pytest.raises(ValueError):
            await store.search(_vec(1.0), top_k=0)


# ─── Delete ──────────────────────────────────────────────────────────────


class TestDelete:
    async def test_delete_removes_chunk(self, store: ChromaVectorStore) -> None:
        await store.upsert(
            [_chunk("c_0", "a"), _chunk("c_1", "b")],
            [_vec(1.0), _vec(0.0, 1.0)],
        )
        await store.delete("c_0")
        assert (await store.get_stats())["total_chunks"] == 1
        # Confirm `c_1` survived.
        results = await store.search(_vec(0.0, 1.0), top_k=5)
        assert {r.chunk_id for r in results} == {"c_1"}

    async def test_delete_unknown_id_does_not_raise(self, store: ChromaVectorStore) -> None:
        await store.delete("does-not-exist")  # silent no-op


# ─── Persistence ─────────────────────────────────────────────────────────


class TestPersistence:
    async def test_data_survives_store_recreation(self, tmp_path: Path) -> None:
        path = tmp_path / "persist"
        store1 = ChromaVectorStore(persist_dir=path)
        await store1.upsert(
            [_chunk("c_0", "persisted"), _chunk("c_1", "still here")],
            [_vec(1.0), _vec(0.0, 1.0)],
        )

        # Throw away the first handle and re-open from disk.
        del store1
        store2 = ChromaVectorStore(persist_dir=path)
        assert (await store2.get_stats())["total_chunks"] == 2
        results = await store2.search(_vec(1.0), top_k=2)
        assert {r.chunk_id for r in results} == {"c_0", "c_1"}
