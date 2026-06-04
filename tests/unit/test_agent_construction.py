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
    """Settings used by the agent-construction tests.

    We pin `llm_provider="gemini"` explicitly. Why: the project's default
    `llm_provider` is `"ollama"`, which causes CrewAI/LiteLLM to probe for
    a local Ollama binary at `LLM(...)` construction time — that probe
    fails in CI (no Ollama installed), surfacing as a cryptic
    "ImportError: Error importing native provider". Pinning to `"gemini"`
    keeps construction inert until first .invoke(), which we don't reach
    in these wiring-only tests.
    """
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider="gemini",
        google_api_key="test-key",
    )


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


class _StubAgent:
    """Records constructor kwargs as attributes — substitute for CrewAI's
    `Agent` in wiring-only tests.

    Why we stub `Agent` itself rather than `LLM`: CrewAI's `LLM` and
    `Agent` are Pydantic models with provider-probing in their init
    machinery. Patching just `LLM.__init__` doesn't reliably bypass the
    Pydantic-driven validation chain. Replacing `Agent` outright gives
    us a deterministic, no-import-side-effect path through
    `KnowledgeAgents.__init__`, and the tests' assertions
    (`.role`, `.tools`) still work because we mirror the kwargs onto
    `self`.
    """

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestKnowledgeAgents:
    @pytest.fixture(autouse=True)
    def _stub_crewai_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch `src.agent.agents` so `KnowledgeAgents.__init__` never
        reaches CrewAI's eager provider-probing code paths.

        Two replacements:

        * `_build_llm` → sentinel-returning lambda. We don't construct
          CrewAI's real `LLM`, so its `errno 2` probe never fires.
        * `Agent` → `_StubAgent`. Construction becomes a pure
          kwargs-record, but the tests can still read `.role` and
          `.tools` exactly the way they would on a real agent.
        """
        from src.agent import agents as agents_mod

        monkeypatch.setattr(agents_mod, "_build_llm", lambda _settings: object())
        monkeypatch.setattr(agents_mod, "Agent", _StubAgent)

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
