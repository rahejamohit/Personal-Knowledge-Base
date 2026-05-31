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
import json
from typing import Any

from crewai.tools import BaseTool, tool

from src.models.document import DocumentChunk, RetrievedDoc  # noqa: F401 (re-exported)
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Pure async tool implementations ─────────────────────────────────────


async def retrieve(query: str, top_k: int = 5) -> list[RetrievedDoc]:
    """Search the knowledge base for documents relevant to `query`.

    Phase 0 stub — returns `[]`. Phase 1.3 wires this to the embedding
    provider + `ChromaVectorStore.search()`. The signature is the
    long-term contract; only the body changes.

    Args:
        query: Natural-language question to search for.
        top_k: Number of results to return (1-10). Validation lives in
            the `@tool` wrapper / Phase 1.3 implementation, not here, so
            callers can experiment freely.
    """
    logger.info("retrieve(query=%r, top_k=%d) [Phase 0 stub]", query[:80], top_k)
    # Phase 1.3 implementation:
    #   embedding = await get_embedding_provider().embed_query(query)
    #   return await get_vector_store().search(embedding, top_k=top_k)
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
    # `asyncio.run` is safe because CrewAI's `crew.kickoff()` calls tools
    # from a synchronous context. If a future caller invokes tools from
    # within a running event loop, switch this to `asyncio.to_thread` or
    # a `run_until_complete` on a dedicated loop.
    docs = asyncio.run(retrieve(query=query, top_k=top_k))
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
    return asyncio.run(cite(chunk_id=chunk_id, excerpt=excerpt or None))


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
