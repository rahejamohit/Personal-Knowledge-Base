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


class TurnRequest(BaseModel):
    """Request body for `POST /api/turns`.

    `session_id` lives in the body (not the URL path) so it doesn't leak into
    server access logs, browser history, referrer headers, or analytics. The
    same shape extends cleanly to Phase 1's Bearer-token auth.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        ...,
        min_length=1,
        description="Server-generated session ID returned by POST /api/sessions.",
    )
    query: str = Field(..., min_length=1, description="User's question or message.")


class CreateSessionResponse(BaseModel):
    """Response body for `POST /api/sessions`.

    The server, not the client, generates `session_id`. This eliminates the
    collision risk that arises when the same user opens the app on multiple
    devices (phone + tablet + web) and lets the backend control the ID
    format (currently `sess_<12-hex>` from UUID4).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    user_id: str = "local"
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
