"""HTTP response models for the REST API.

Design notes
------------
* We **re-export** `TokenUsage` and `RetrievedDoc` from `src.models.conversation`
  rather than duplicating their shape. Two reasons:
    1. One schema, one source of truth — any future field added to the
       internal `ConversationTurn` automatically flows to the wire format
       (or fails loudly via Pydantic if incompatible).
    2. We don't want to touch `src/models/`, but we *can* import from it.
* These response models intentionally do NOT include internal-only state
  such as the per-turn `tool_calls`/`tool_results` arrays. Phase 0 keeps the
  API surface minimal; Phase 1 can expose them under `/turns/{id}/debug`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Re-export internal models so frontend devs only see one set of types.
from src.models.conversation import RetrievedDoc, TokenUsage  # noqa: F401  (re-export)


class TurnResponse(BaseModel):
    """A single conversation turn returned over the wire.

    Mirrors `ConversationTurn` but renames `id` → `turn_id` for clarity at
    the API boundary (the JSON sits next to `session_id`, so `turn_id` is
    less ambiguous than a bare `id`).
    """

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    session_id: str
    turn_number: int
    user_message: str
    agent_response: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    """Summary view of a session — no turn payloads."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    user_id: str
    title: str | None = None
    turn_count: int = Field(..., ge=0)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnsListResponse(BaseModel):
    """Paginated list of turns for a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_turns: int = Field(..., ge=0)
    turns: list[TurnResponse]
