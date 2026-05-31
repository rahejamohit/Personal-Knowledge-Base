"""Unit tests for the ConversationManager state machine and history logic.

We mock out the orchestrator so these tests don't hit any external API.
The integration test in `tests/integration/` is the one that actually
exercises CrewAI + Gemini.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agent.conversation_manager import ConversationManager, ConversationState
from src.config import Settings
from src.models.conversation import TokenUsage


@dataclass
class _FakeRun:
    answer: str
    token_usage: TokenUsage


class _FakeAgent:
    """In-memory stand-in for `KnowledgeAgent`. Records calls for assertion."""

    def __init__(self, response: str = "fake answer", *, raise_on_call: bool = False) -> None:
        self.response = response
        self.raise_on_call = raise_on_call
        self.calls: list[tuple[str, str]] = []

    def answer_query(self, query: str, history: str = "") -> _FakeRun:
        self.calls.append((query, history))
        if self.raise_on_call:
            raise RuntimeError("simulated crew failure")
        return _FakeRun(
            answer=self.response,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )


def _make_manager(**kwargs) -> ConversationManager:
    agent = kwargs.pop("agent", _FakeAgent())
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    return ConversationManager(agent=agent, settings=settings, **kwargs)


class TestSingleTurn:
    def test_first_turn_produces_one_record(self) -> None:
        mgr = _make_manager()
        turn = mgr.process_turn("What is RAG?")
        assert turn.turn_number == 1
        assert turn.user_message == "What is RAG?"
        assert turn.agent_response == "fake answer"
        assert turn.token_usage.total_tokens == 15
        assert len(mgr.turns) == 1

    def test_history_is_empty_on_first_turn(self) -> None:
        agent = _FakeAgent()
        mgr = _make_manager(agent=agent)
        mgr.process_turn("hi")
        assert agent.calls == [("hi", "")]

    def test_state_returns_to_waiting(self) -> None:
        mgr = _make_manager()
        mgr.process_turn("hi")
        assert mgr.state == ConversationState.WAITING_INPUT

    def test_empty_input_rejected(self) -> None:
        mgr = _make_manager()
        with pytest.raises(ValueError):
            mgr.process_turn("   ")


class TestMultiTurnContext:
    def test_history_grows_with_turns(self) -> None:
        agent = _FakeAgent()
        mgr = _make_manager(agent=agent)
        mgr.process_turn("What is RAG?")
        mgr.process_turn("And what about embeddings?")
        mgr.process_turn("Summarize so far.")

        assert len(mgr.turns) == 3
        # The 3rd call receives history containing the first two turns
        third_history = agent.calls[2][1]
        assert "What is RAG?" in third_history
        assert "And what about embeddings?" in third_history
        assert "Turn 1" in third_history
        assert "Turn 2" in third_history

    def test_history_respects_history_turns_setting(self, monkeypatch) -> None:
        monkeypatch.setenv("PKA_HISTORY_TURNS", "2")
        agent = _FakeAgent()
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        mgr = ConversationManager(agent=agent, settings=settings)

        # Six calls so the final one (i=5) has five prior turns to choose
        # from, and PKA_HISTORY_TURNS=2 truncates that to turns 4 and 5
        # (messages "msg 3" and "msg 4").
        for i in range(6):
            mgr.process_turn(f"msg {i}")

        # The 6th call's history should only include the last 2 prior turns.
        last_history = agent.calls[-1][1]
        assert "msg 0" not in last_history
        assert "msg 1" not in last_history
        assert "msg 2" not in last_history
        assert "msg 3" in last_history
        assert "msg 4" in last_history

    def test_turn_numbers_are_sequential(self) -> None:
        mgr = _make_manager()
        for _ in range(3):
            mgr.process_turn("hi")
        assert [t.turn_number for t in mgr.turns] == [1, 2, 3]


class TestErrorRecovery:
    def test_crew_failure_yields_graceful_turn(self) -> None:
        agent = _FakeAgent(raise_on_call=True)
        mgr = _make_manager(agent=agent)
        turn = mgr.process_turn("anything")
        # We still get a turn record back — the error isn't propagated to the CLI.
        assert "internal error" in turn.agent_response.lower()
        assert turn.metadata.get("errored") is False or "latency_ms" in turn.metadata
        assert mgr.state == ConversationState.WAITING_INPUT

    def test_session_continues_after_error(self) -> None:
        agent = _FakeAgent(raise_on_call=True)
        mgr = _make_manager(agent=agent)
        mgr.process_turn("first")
        # Switch to a healthy agent and keep going.
        mgr.agent = _FakeAgent(response="recovered")
        turn2 = mgr.process_turn("second")
        assert turn2.agent_response == "recovered"
        assert turn2.turn_number == 2
