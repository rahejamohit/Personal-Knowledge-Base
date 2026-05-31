"""Unit tests for agent + tool construction (no LLM calls).

These verify the *structure* of the crew: that agents pick up the right
tools and that the factory returns the expected registry. The actual
LLM-driven behavior is exercised by the integration test, and tool
behavior is covered in `test_rag_tools.py`.
"""

from __future__ import annotations

import pytest

from src.agent.agents import KnowledgeAgents
from src.agent.tools import build_default_tools
from src.config import Settings


def _test_settings() -> Settings:
    """Settings with a placeholder Gemini key (no calls actually made)."""
    return Settings(_env_file=None, google_api_key="test-key")  # type: ignore[call-arg]


class TestToolRegistry:
    def test_default_registry_has_expected_tools(self) -> None:
        tools = build_default_tools()
        assert set(tools.keys()) == {"retrieve", "cite"}
        # `@tool("retrieve")` sets the `.name` attribute on the resulting
        # CrewAI tool object. That's all `KnowledgeAgents` cares about.
        assert tools["retrieve"].name == "retrieve"
        assert tools["cite"].name == "cite"

    def test_each_tool_has_a_description(self) -> None:
        # CrewAI feeds `description` into the LLM-facing tool catalog.
        # An empty description would make the LLM blind to the tool's
        # purpose — assert we keep them populated.
        tools = build_default_tools()
        for name, t in tools.items():
            assert t.description, f"tool {name!r} has empty description"


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
