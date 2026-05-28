"""Memory models (Phase 2 builds on these).

Phase 0 defines the shapes so Phase 2's memory manager can implement against
a stable contract. The hierarchy is:

* Recent turns           → raw `ConversationTurn` objects (Phase 1)
* Older runs of turns    → `MemorySummary` (Phase 2)
* Long-lived facts       → `ExtractedFact` (Phase 2)
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class MemorySummary(BaseModel):
    """Compressed summary of a contiguous range of turns.

    Used to keep older context available without burning tokens on every
    word. Phase 2 generates these with a cheap Gemini call.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_range_start: int = Field(..., ge=1, description="First turn covered (inclusive).")
    turn_range_end: int = Field(..., ge=1, description="Last turn covered (inclusive).")
    summary: str = Field(..., min_length=1)
    key_facts: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractedFact(BaseModel):
    """A single fact extracted from conversation for long-term recall.

    `confidence` and `source_turn_ids` exist so Phase 2's grounding logic can
    decide whether to surface this fact in a new turn.
    """

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    session_id: str
    statement: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_turn_ids: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
