"""Unit tests for the Phase 1 RAG tools (`retrieve`, `cite`).

Two layers are exercised here:

* **Async core** — the pure async stubs (`retrieve`, `cite`) that Phase
  1.3 / 1.4 will replace. We verify the Phase 0 stub behavior so we'll
  catch any accidental regression when the real implementation lands
  (the tests in this file should be UPDATED, not deleted, at that point).
* **CrewAI wrappers** — the `@tool`-decorated sync wrappers, including
  the JSON-encoded output shape the LLM ultimately reads.

Note: `pyproject.toml` sets `asyncio_mode = "auto"`, so `async def
test_*` functions are auto-detected as asyncio tests without needing a
per-function marker. The explicit markers below match the spec exactly.
"""

from __future__ import annotations

import json

import pytest

from src.agent.tools import (
    build_default_tools,
    cite,
    retrieve,
    tool_cite,
    tool_retrieve,
)
from src.models.conversation import RetrievedDoc

# `retrieve` is `async def`, so `mocker.patch` auto-creates an AsyncMock —
# `return_value=<list>` is what the awaited coroutine resolves to, which is
# exactly what `_run_sync(retrieve(...))` hands the wrapper.


# ─── Async core: retrieve ────────────────────────────────────────────────
#
# The Phase 1.3 implementation is exercised in depth (reranking, relevance,
# failure modes) by `tests/unit/test_retrieve.py`, which injects a fake
# embedder + throwaway store. Here we only pin the contract that the CrewAI
# wrapper relies on: a blank query short-circuits to `[]` without ever
# touching an embedding provider or the index (so it's network-free).


class TestRetrieveContract:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        results = await retrieve("")
        assert results == []
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self) -> None:
        results = await retrieve("   \t\n ")
        assert results == []


# ─── Async core: cite ────────────────────────────────────────────────────


class TestCiteAsyncStub:
    @pytest.mark.asyncio
    async def test_returns_placeholder_with_chunk_id(self) -> None:
        result = await cite("doc_001_chunk_02")
        assert isinstance(result, str)
        assert "citation:" in result
        # The chunk_id should be embedded in the placeholder so the agent
        # can correlate citations back to retrieved chunks during the
        # Phase 0 stub phase.
        assert "doc_001_chunk_02" in result

    @pytest.mark.asyncio
    async def test_accepts_optional_excerpt(self) -> None:
        result = await cite("doc_001", excerpt="some excerpt text")
        assert isinstance(result, str)
        # When an excerpt is provided, it should appear in the citation
        # (truncated, but visible).
        assert "some excerpt" in result

    @pytest.mark.asyncio
    async def test_truncates_long_excerpt(self) -> None:
        long_excerpt = "x" * 500
        result = await cite("doc_001", excerpt=long_excerpt)
        # Truncated to 80 chars by the stub.
        assert len(result) < 200


# ─── CrewAI wrappers ─────────────────────────────────────────────────────


class TestToolRetrieveWrapper:
    """The CrewAI sync wrapper bridges to the async core and shapes JSON.

    We patch the async `retrieve` so these assert the *wrapper's* behavior
    (sync bridging + JSON shape) deterministically, independent of whether
    an embedding provider or index is reachable.
    """

    def test_returns_valid_json_array(self, mocker) -> None:
        mocker.patch("src.agent.tools.retrieve", return_value=[])
        output = tool_retrieve.run(query="hi", top_k=3)
        assert json.loads(output) == []

    def test_shapes_results_with_similarity_score_key(self, mocker) -> None:
        doc = RetrievedDoc(
            chunk_id="c_1",
            doc_id="d_1",
            source="notes.md",
            text="some retrieved text",
            score=0.83,
            rank=0,
            metadata={"page": 2},
        )
        mocker.patch("src.agent.tools.retrieve", return_value=[doc])
        parsed = json.loads(tool_retrieve.run(query="hi", top_k=3))
        assert len(parsed) == 1
        # The agent sees `score` under the `similarity_score` key per the spec.
        assert parsed[0]["similarity_score"] == 0.83
        assert parsed[0]["chunk_id"] == "c_1"
        assert parsed[0]["metadata"] == {"page": 2}

    def test_tool_has_correct_name(self) -> None:
        assert tool_retrieve.name == "retrieve"

    @pytest.mark.asyncio
    async def test_works_inside_running_event_loop(self, mocker) -> None:
        mocker.patch("src.agent.tools.retrieve", return_value=[])
        output = tool_retrieve.run(query="hi", top_k=3)
        assert json.loads(output) == []


class TestToolCiteWrapper:
    def test_returns_placeholder_string(self) -> None:
        out = tool_cite.run(chunk_id="doc_X")
        assert isinstance(out, str)
        assert "citation:" in out and "doc_X" in out

    def test_tool_has_correct_name(self) -> None:
        assert tool_cite.name == "cite"

    @pytest.mark.asyncio
    async def test_works_inside_running_event_loop(self) -> None:
        out = tool_cite.run(chunk_id="doc_X")
        assert isinstance(out, str)
        assert "citation:" in out and "doc_X" in out


# ─── Registry ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_default_tools_includes_both() -> None:
    """Tool registry exposes both tools under stable string keys.

    `KnowledgeAgents` does `tools.get("retrieve")` / `.get("cite")`, so
    changing these keys is a breaking change for the agent loop.
    """
    tools = build_default_tools()
    assert "retrieve" in tools
    assert "cite" in tools
    assert tools["retrieve"] is tool_retrieve
    assert tools["cite"] is tool_cite
