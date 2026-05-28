"""Conversation manager: state machine, context assembly, turn persistence.

This is the layer between the CLI and the CrewAI orchestrator. It owns:

* The conversation **state machine** (WAITING_INPUT → PROCESSING →
  GENERATING → PERSISTING → WAITING_INPUT).
* Turn history (Phase 0: in-memory only; Phase 1 swaps in SQLite).
* Context assembly with a token budget.
* Error recovery — if the crew fails, the manager produces a graceful error
  turn instead of crashing the CLI.

Why a state machine?
--------------------
The states map 1:1 to the architecture doc's diagram and make it obvious
when concurrency rules apply (e.g. "don't accept new input while
PROCESSING"). They also give us natural log breakpoints for Phase 3
observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Protocol

from src.agent.orchestrator import KnowledgeAgent
from src.config import Settings, get_settings
from src.models.conversation import (
    ConversationTurn,
    Session,
    TokenUsage,
)
from src.utils import get_logger
from src.utils.ids import new_session_id, new_turn_id

logger = get_logger(__name__)


class ConversationState(str, Enum):
    """States in the per-turn lifecycle."""

    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    GENERATING = "generating"
    PERSISTING = "persisting"
    ERROR = "error"


class _AgentProtocol(Protocol):
    """Structural typing for the orchestrator so tests can pass a fake."""

    def answer_query(self, query: str, history: str = "") -> object: ...


@dataclass
class ConversationManager:
    """One instance per conversation thread.

    Phase 0 keeps `turns` in memory. Phase 1's session storage will replace
    the list with a SQLite-backed buffer behind the same interface.
    """

    agent: _AgentProtocol
    settings: Settings = field(default_factory=get_settings)
    session: Session = field(default_factory=lambda: Session(id=new_session_id()))
    turns: list[ConversationTurn] = field(default_factory=list)
    state: ConversationState = ConversationState.WAITING_INPUT

    # ─── public API ────────────────────────────────────────────

    def process_turn(self, user_input: str) -> ConversationTurn:
        """Run one full turn end-to-end. Returns the persisted turn record."""
        if self.state == ConversationState.PROCESSING:
            # Defensive — the CLI shouldn't ever reach here, but if a future
            # async caller does, we refuse rather than corrupt state.
            raise RuntimeError("Conversation is already processing a turn.")

        user_input = user_input.strip()
        if not user_input:
            raise ValueError("Empty user input.")

        turn_number = len(self.turns) + 1
        logger.info("session=%s turn=%d state→PROCESSING", self.session.id, turn_number)

        # --- PROCESSING: assemble context ---
        self.state = ConversationState.PROCESSING
        history_text = self._assemble_history()

        # --- GENERATING: run the crew ---
        self.state = ConversationState.GENERATING
        start = perf_counter()
        try:
            run = self.agent.answer_query(user_input, history=history_text)
            answer = getattr(run, "answer", str(run))
            usage = getattr(run, "token_usage", TokenUsage())
        except Exception as e:  # noqa: BLE001 — top-level recovery boundary
            logger.exception("Crew failed during turn %d: %s", turn_number, e)
            self.state = ConversationState.ERROR
            answer = (
                "Sorry — I ran into an internal error and couldn't complete "
                "that request. The error has been logged."
            )
            usage = TokenUsage()
        latency_ms = (perf_counter() - start) * 1000

        # --- PERSISTING: build + store the turn record ---
        self.state = ConversationState.PERSISTING
        turn = ConversationTurn(
            id=new_turn_id(),
            session_id=self.session.id,
            turn_number=turn_number,
            user_message=user_input,
            agent_response=answer,
            token_usage=usage,
            metadata={"latency_ms": latency_ms, "errored": self.state == ConversationState.ERROR},
        )
        self._persist_turn(turn)

        # --- back to WAITING_INPUT ---
        self.state = ConversationState.WAITING_INPUT
        logger.info(
            "session=%s turn=%d done latency=%.0fms tokens=%d",
            self.session.id,
            turn_number,
            latency_ms,
            usage.total_tokens,
        )
        return turn

    # ─── helpers ───────────────────────────────────────────────

    def _assemble_history(self) -> str:
        """Render the last N turns as a plain-text block for the LLM prompt.

        Phase 0 uses a naive char-based cap as a stand-in for a real token
        counter. Phase 2's memory manager will replace this with summarized
        older turns + verbatim recent turns under a strict token budget.
        """
        if not self.turns:
            return ""

        recent = self.turns[-self.settings.pka_history_turns :]
        # Rough heuristic: ~4 chars/token. We allot half the context budget
        # to history so retrieved docs + system prompt still fit.
        char_budget = self.settings.pka_max_context_tokens * 4 // 2

        rendered: list[str] = []
        total = 0
        # Walk most-recent-first so we keep the freshest turns when truncating.
        for t in reversed(recent):
            block = (
                f"[Turn {t.turn_number}]\n"
                f"  User:  {t.user_message}\n"
                f"  Agent: {t.agent_response}"
            )
            if total + len(block) > char_budget:
                break
            rendered.append(block)
            total += len(block)

        return "\n".join(reversed(rendered))

    def _persist_turn(self, turn: ConversationTurn) -> None:
        """Phase 0: keep in memory. Phase 1: write through to SQLite."""
        self.turns.append(turn)
