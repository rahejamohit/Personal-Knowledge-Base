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


# ─── Async core: retrieve ────────────────────────────────────────────────


class TestRetrieveAsyncStub:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self) -> None:
        results = await retrieve("What is RAG?", top_k=5)
        assert results == []
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_default_top_k_is_5(self) -> None:
        # Stub returns [] regardless, but the signature should accept the
        # one-positional-arg form without raising.
        results = await retrieve("hi")
        assert results == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 5, 10])
    async def test_accepts_valid_top_k_range(self, top_k: int) -> None:
        # Phase 0 stub doesn't validate; this just confirms the signature
        # accepts the documented range without surprise.
        results = await retrieve("test", top_k=top_k)
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
    def test_returns_valid_json(self) -> None:
        # `tool_retrieve` is a CrewAI tool object; invoke its `.run` method
        # the same way CrewAI would when the agent picks the tool.
        output = tool_retrieve.run(query="hi", top_k=3)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        # Phase 0 stub: empty list → empty JSON array.
        assert parsed == []

    def test_tool_has_correct_name(self) -> None:
        assert tool_retrieve.name == "retrieve"


class TestToolCiteWrapper:
    def test_returns_placeholder_string(self) -> None:
        out = tool_cite.run(chunk_id="doc_X")
        assert isinstance(out, str)
        assert "citation:" in out and "doc_X" in out

    def test_tool_has_correct_name(self) -> None:
        assert tool_cite.name == "cite"


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
