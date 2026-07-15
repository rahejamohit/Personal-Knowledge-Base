"""Agent tools — Phase 1 RAG stubs.

Two stateless tool functions the CrewAI agents call:

* `retrieve(query, top_k)` — search the vector store (Phase 1.3 fills in).
* `cite(chunk_id, excerpt)` — format a citation (Phase 1.4 fills in).

Each tool has TWO public surfaces:

1. The pure async function — what scripts, evals, REST handlers, and
   Phase 1+ code import directly.
2. The `@tool(...)`-decorated sync wrapper — what CrewAI's `Agent` sees.

Keeping the async function as the canonical implementation means Phase 1.3
(real retriever) can wire up I/O without touching the wrapper. The wrapper
exists purely to bridge between CrewAI's sync tool-calling convention and
our async core, and to shape the JSON the LLM ultimately reads.
"""

from __future__ import annotations

import asyncio
import atexit
import json
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, TypeVar

from crewai.tools import BaseTool, tool
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.models.document import DocumentChunk, RetrievedDoc  # noqa: F401 (re-exported)
from src.providers.base import EmbeddingProvider
from src.providers.factory import get_embedding_provider
from src.storage.vector_store import ChromaVectorStore
from src.utils import get_logger

logger = get_logger(__name__)
_async_bridge_executor = ThreadPoolExecutor(max_workers=1)
atexit.register(_async_bridge_executor.shutdown, wait=False, cancel_futures=True)
_T = TypeVar("_T")

# ─── Retrieval tuning knobs ──────────────────────────────────────────────
# Collection the production index lives in. MUST match the name ingestion
# writes to (`src.api.ingestion._DEBUG_COLLECTION`), or `retrieve` would
# search an empty collection.
_RETRIEVE_COLLECTION = "documents"
# Hard ceiling on the candidate pool we pull from Chroma before reranking.
# Over-fetching gives BM25 a wider set to reorder; capping it keeps the
# per-query reranking cost (and latency) bounded.
_MAX_CANDIDATES = 10
# Final score = semantic_weight * cosine_similarity + (1 - w) * bm25_keyword.
# 0.6/0.4 favors semantic relevance while letting exact keyword hits break
# ties — tunable; see PHASE 1.3 notes if results need adjusting.
_SEMANTIC_WEIGHT = 0.6


@lru_cache(maxsize=1)
def _default_vector_store() -> ChromaVectorStore:
    """The production Chroma store, memoized like `get_embedding_provider`.

    Built lazily (and cached) so importing this module stays cheap and the
    CLI/agent path doesn't open the on-disk index until the first real
    `retrieve` call. Tests bypass this entirely by passing `vector_store=`.
    """
    return ChromaVectorStore(
        collection_name=_RETRIEVE_COLLECTION,
        persist_dir=get_settings().pka_chroma_dir,
    )


def _rerank(query: str, candidates: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Blend each candidate's semantic score with a BM25 keyword score.

    Semantic search surfaces chunks that are *similar* to the query; BM25
    rewards chunks that share the query's actual *words*. Combining them
    lifts exact-term matches (names, error codes, jargon) that pure cosine
    similarity can bury.

    Mutates `candidates` in place — overwrites `.score` with the blended
    value and returns the list sorted by descending score. Callers reassign
    `.rank` afterward to reflect the new order.
    """
    corpus = [doc.text.lower().split() for doc in candidates]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Normalize keyword scores into [0, 1] so they're commensurate with the
    # cosine similarity. If nothing matched (all-zero), keyword weight is 0
    # and the ranking falls back to pure semantic order.
    max_score = max(bm25_scores) if len(bm25_scores) else 0.0
    keyword_weight = 1.0 - _SEMANTIC_WEIGHT
    for doc, raw in zip(candidates, bm25_scores, strict=True):
        keyword = (float(raw) / max_score) if max_score > 0 else 0.0
        doc.score = _SEMANTIC_WEIGHT * doc.score + keyword_weight * keyword

    candidates.sort(key=lambda d: d.score, reverse=True)
    return candidates


# ─── Pure async tool implementations ─────────────────────────────────────


async def retrieve(
    query: str,
    top_k: int = 5,
    *,
    vector_store: ChromaVectorStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[RetrievedDoc]:
    """Search the knowledge base for documents relevant to `query`.

    The main retrieval entry point the CrewAI agent calls each turn:

    1. Embed `query` with the configured embedding provider.
    2. Pull a candidate pool (up to `_MAX_CANDIDATES`) from Chroma by
       cosine similarity.
    3. Rerank the pool with BM25 keyword scoring blended into the
       similarity (`_rerank`), so exact-term matches surface.
    4. Return the top-`top_k`, with `.rank` reflecting final order.

    Failure is non-fatal: an empty/blank query, an unreachable embedding
    provider, or an empty index all yield `[]` rather than raising, so the
    agent loop degrades to "I couldn't find anything" instead of crashing.

    Args:
        query: Natural-language question to search for.
        top_k: Number of results to return. Clamped to ``[1, 10]``.
        vector_store: Store to search. Defaults to the production index;
            injectable so tests (and future callers) can target a specific
            collection without touching the real one.
        embedder: Embedding provider. Defaults to the configured provider;
            injectable for the same reason.
    """
    query = query.strip() if query else ""
    if not query:
        logger.warning("retrieve() called with empty/blank query — returning []")
        return []

    if not 1 <= top_k <= 10:
        logger.warning("retrieve() top_k=%d out of range [1, 10]; clamping to 5", top_k)
        top_k = 5

    logger.info("retrieve(query=%r, top_k=%d)", query[:80], top_k)

    try:
        embedder = embedder or get_embedding_provider()
        store = vector_store or _default_vector_store()

        # `embed_query` is synchronous and may hit the network (Ollama /
        # OpenAI / Gemini), so offload it rather than block the event loop.
        query_embedding = await asyncio.to_thread(embedder.embed_query, query)

        # Over-fetch so BM25 has a pool to reorder, then trim to top_k.
        candidate_k = min(_MAX_CANDIDATES, top_k * 2)
        candidates = await store.search(query_embedding, top_k=candidate_k)
        if not candidates:
            logger.info("retrieve(): index returned no candidates for %r", query[:80])
            return []

        results = _rerank(query, candidates)[:top_k]
        # `.rank` must reflect the post-rerank order, not Chroma's.
        for rank, doc in enumerate(results):
            doc.rank = rank

        logger.info(
            "retrieve(query=%r) -> %d results (reranked from %d candidates)",
            query[:80],
            len(results),
            len(candidates),
        )
        return results

    except Exception:
        # Defensive fence: the agent loop must keep going even if retrieval
        # fails (provider down, index missing, etc.). Full traceback logged.
        logger.exception("retrieve() failed for query=%r — returning []", query[:80])
        return []


async def cite(chunk_id: str, excerpt: str | None = None) -> str:
    """Format a citation for `chunk_id`.

    Phase 0 stub — returns a `[citation:<chunk_id>]` placeholder so the
    agent's response still parses end-to-end. Phase 1.4 will look up the
    chunk's metadata (source, page, section) and format it as
    `"[N] source"` or `"[N] source | excerpt"`.

    Args:
        chunk_id: ID of the chunk to cite (from `retrieve` results).
        excerpt: Optional text to include in the citation. Trimmed to
            80 chars in the stub.
    """
    logger.info("cite(chunk_id=%r, has_excerpt=%s) [Phase 0 stub]", chunk_id, bool(excerpt))
    if excerpt:
        return f"[citation:{chunk_id} | {excerpt[:80]}]"
    return f"[citation:{chunk_id}]"


# ─── CrewAI-facing sync wrappers ─────────────────────────────────────────


def _retrieved_for_agent(doc: RetrievedDoc) -> dict[str, Any]:
    """Reshape a `RetrievedDoc` for the agent-facing JSON.

    The agent sees the score under the `similarity_score` key the spec
    documents. The internal model uses `score` so the FastAPI
    `TurnResponse` JSON contract stays stable; the wrapper is the one
    place that translates.
    """
    return {
        "chunk_id": doc.chunk_id,
        "text": doc.text,
        "source": doc.source,
        "similarity_score": doc.score,
        "metadata": doc.metadata,
    }


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run coroutine from sync code, including when an event loop already exists."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _async_bridge_executor.submit(asyncio.run, coro).result()


@tool("retrieve")
def tool_retrieve(query: str, top_k: int = 5) -> str:
    """Search the user's knowledge base for documents relevant to a question.

    Use this whenever the user asks something that might be answered by
    their personal documents. The result is a JSON array of objects with
    `chunk_id`, `text`, `source`, `similarity_score`, and `metadata`.
    Quote the `text` directly when you reference a result and pass the
    `chunk_id` to the `cite` tool.

    Args:
        query: A focused, keyword-rich search query (not a full sentence).
        top_k: How many results to return (1-10, default 5).

    Returns:
        JSON-encoded list of retrieved documents.
    """
    docs = _run_sync(retrieve(query=query, top_k=top_k))
    return json.dumps([_retrieved_for_agent(d) for d in docs])


@tool("cite")
def tool_cite(chunk_id: str, excerpt: str = "") -> str:
    """Format a citation for a retrieved document chunk.

    Call once per source you reference. Embed the returned string inline
    in your answer, e.g. `"... improves recall [1] ..."`.

    Args:
        chunk_id: The chunk ID from a `retrieve` result.
        excerpt: Optional excerpt to highlight. Pass `""` to omit.

    Returns:
        Citation string ready to embed in the final answer.
    """
    # `excerpt=""` means "no excerpt"; we normalize to `None` before
    # calling the async core so the stub's branch is exercised correctly.
    return _run_sync(cite(chunk_id=chunk_id, excerpt=excerpt or None))


# ─── Tool registry ────────────────────────────────────────────────────────


def build_default_tools() -> dict[str, BaseTool]:
    """Tool registry consumed by `KnowledgeAgents`.

    Returns CrewAI tool objects keyed by name. `KnowledgeAgents` picks
    them up via `tools.get("retrieve")` / `tools.get("cite")`, so adding
    a new tool means: add a `@tool` definition above, add it to the
    return dict here, and reference its key in `KnowledgeAgents`.
    """
    return {
        "retrieve": tool_retrieve,
        "cite": tool_cite,
    }
