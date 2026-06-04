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


class TestKnowledgeAgents:
    @pytest.fixture(autouse=True)
    def _stub_llm_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bypass CrewAI's real `LLM.__init__` for these wiring-only tests.

        Why a method patch rather than replacing the whole class:
        CrewAI's `Agent` is a Pydantic model that validates `llm` is a
        true `crewai.LLM` instance. Replacing the class would fail that
        validation. Patching only `__init__` keeps the class identity
        (and thus Pydantic's validator) happy while skipping the eager
        provider probing that fails with `errno 2` in some envs.

        The replacement records the args on `self` so the test can still
        inspect what `_build_llm` was constructed with if needed.
        """

        def _noop_init(
            self: object,
            model: str = "",
            api_key: str = "",
            temperature: float = 0.0,
            **_: object,
        ) -> None:
            object.__setattr__(self, "model", model)
            object.__setattr__(self, "api_key", api_key)
            object.__setattr__(self, "temperature", temperature)

        # Patch via the same import path agents.py uses, so we hit the
        # right class object regardless of how CrewAI internally re-
        # exports `LLM`.
        from src.agent import agents as agents_mod

        monkeypatch.setattr(agents_mod.LLM, "__init__", _noop_init)

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
