"""Conversation models.

The `ConversationTurn` is the central audit record of the system: every
interaction produces exactly one, and Phase 1's SQLite store persists them.

Design notes
------------
* `RetrievedDoc` is intentionally separate from `DocumentChunk` — chunks live
  in the vector store with full text and embedding, while a retrieved doc is
  a *snapshot* of what the agent actually saw (excerpt + score). This means
  we can re-embed documents later without invalidating historical turns.
* `TokenUsage` carries both prompt/completion counts (for cost reporting) and
  a derived `total` so we never recompute it inconsistently.
* All datetimes are timezone-aware UTC. SQLite stores them as ISO strings via
  Pydantic's JSON mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, computed_field

from src.models.tool_calls import ToolCall, ToolResult


class TokenUsage(BaseModel):
    """Token accounting for one turn (aggregated across all Gemini calls)."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: NonNegativeInt = 0
    completion_tokens: NonNegativeInt = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usages — handy when a turn makes multiple Gemini calls."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class RetrievedDoc(BaseModel):
    """A document surfaced by the retriever for one turn.

    Stored verbatim on the turn so we can later see exactly what the agent
    was shown, even if the underlying chunk is re-embedded or deleted.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source: str = Field(..., description="Origin file/URL — used for citations.")
    text: str = Field(..., description="The excerpt actually shown to the LLM.")
    score: float = Field(..., description="Similarity score from the vector store.")
    rank: NonNegativeInt = Field(..., description="0-indexed rank within this turn's results.")


class ConversationTurn(BaseModel):
    """One round-trip of user → agent within a session.

    This is the persisted unit. The agent loop produces one of these per
    user input. Schema is stable across phases so older turns can be replayed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    turn_number: int = Field(..., ge=1, description="1-indexed turn within the session.")
    user_message: str
    agent_response: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A logical conversation thread. One session may contain many turns."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str = Field(default="local", description="Phase 1 is single-user.")
    title: str | None = Field(default=None, description="Optional human label.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
