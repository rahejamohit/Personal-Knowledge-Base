"""End-to-end integration test: 3+ turn conversation via the real crew.

Marked `integration` so the default `pytest` run skips it. Requires
`GOOGLE_API_KEY` to be set. Run with:

    uv run pytest -m integration

Cost note: each turn is ~2 Gemini calls, each ~300-500 tokens. With
`gemini-2.0-flash` at $0.075/1M input, the whole test costs <$0.001.
"""

from __future__ import annotations

import os

import pytest

from src.agent.conversation_manager import ConversationManager
from src.agent.orchestrator import KnowledgeAgent
from src.agent.tools import build_default_tools
from src.config import get_settings


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def manager() -> ConversationManager:
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set; skipping live Gemini test.")
    settings = get_settings()
    agent = KnowledgeAgent(build_default_tools(), settings=settings, verbose=False)
    return ConversationManager(agent=agent, settings=settings)


def test_three_turn_conversation_maintains_context(manager: ConversationManager) -> None:
    """The agent should remember an earlier turn within the same session."""
    t1 = manager.process_turn("My favorite color is teal. Remember that.")
    assert t1.agent_response  # non-empty

    t2 = manager.process_turn(
        "Pick a programming language and explain why it's good for beginners. One sentence."
    )
    assert t2.agent_response

    t3 = manager.process_turn("What did I say my favorite color was?")
    # The model has access to the history block; "teal" should appear in the answer.
    assert "teal" in t3.agent_response.lower(), (
        f"Expected the agent to recall 'teal' from turn 1, got: {t3.agent_response!r}"
    )
    assert len(manager.turns) == 3
    assert [t.turn_number for t in manager.turns] == [1, 2, 3]
