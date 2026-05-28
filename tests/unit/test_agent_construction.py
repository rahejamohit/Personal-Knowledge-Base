"""Unit tests for agent + tool construction (no LLM calls).

These verify the *structure* of the crew: that agents pick up the right
tools, that tool input validation works, and that the factory returns the
expected registry. The actual LLM-driven behavior is exercised by the
integration test.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.agents import KnowledgeAgents
from src.agent.tools import (
    CiteInput,
    CiteTool,
    RetrieveInput,
    RetrieveTool,
    build_default_tools,
)
from src.config import Settings


def _test_settings() -> Settings:
    """Settings with a placeholder Gemini key (no calls actually made)."""
    return Settings(_env_file=None, google_api_key="test-key")  # type: ignore[call-arg]


class TestRetrieveTool:
    def test_input_schema_validates_top_k_range(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveInput(query="hi", top_k=0)
        with pytest.raises(ValidationError):
            RetrieveInput(query="hi", top_k=100)

    def test_stub_returns_helpful_message(self) -> None:
        tool = RetrieveTool()
        out = tool._run(query="What is RAG?", top_k=3)
        assert "phase 0" in out.lower() or "not yet" in out.lower()


class TestCiteTool:
    def test_formats_source_and_excerpt(self) -> None:
        tool = CiteTool()
        out = tool._run(source="paper.pdf", excerpt="A retrieval-augmented model uses ...")
        assert "paper.pdf" in out
        assert "retrieval-augmented" in out

    def test_truncates_long_excerpts(self) -> None:
        tool = CiteTool()
        long = "x" * 500
        out = tool._run(source="s", excerpt=long)
        # Truncated to <100 chars of excerpt content
        assert len(out) < 200

    def test_input_requires_both_fields(self) -> None:
        with pytest.raises(ValidationError):
            CiteInput(source="x")  # type: ignore[call-arg]


class TestToolRegistry:
    def test_default_registry_has_expected_tools(self) -> None:
        tools = build_default_tools()
        assert set(tools.keys()) == {"retrieve", "cite"}
        assert isinstance(tools["retrieve"], RetrieveTool)
        assert isinstance(tools["cite"], CiteTool)


class TestKnowledgeAgents:
    def test_constructs_both_agents_with_tools(self) -> None:
        tools = build_default_tools()
        agents = KnowledgeAgents(tools, settings=_test_settings())
        assert agents.retrieve_agent.role == "Document Retrieval Specialist"
        assert agents.answer_agent.role == "Knowledge Base Analyst"
        # Retrieve agent has the retrieve tool, answer agent has cite tool.
        retrieve_tool_names = {t.name for t in agents.retrieve_agent.tools}
        answer_tool_names = {t.name for t in agents.answer_agent.tools}
        assert "retrieve" in retrieve_tool_names
        assert "cite" in answer_tool_names

    def test_missing_required_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieve.*cite"):
            KnowledgeAgents({}, settings=_test_settings())
