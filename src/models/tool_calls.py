"""Tool-call data models.

These mirror the ReAct loop: Gemini emits a `ToolCall`, we execute it, and
attach a `ToolResult`. Storing both on the `ConversationTurn` lets us replay
or audit any decision the agent made — critical for debugging hallucinations
in Phase 3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM.

    `arguments` is intentionally `dict[str, Any]` because tool schemas vary
    per tool. Each tool handler validates its own inputs against a per-tool
    Pydantic model at the boundary.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    tool_name: str = Field(..., description="Name of the tool (e.g. 'retrieve').")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable arguments produced by the LLM.",
    )
    called_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Wall-clock time the call was dispatched.",
    )


class ToolResult(BaseModel):
    """Result of a tool invocation. Either `output` OR `error` is populated."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    tool_name: str
    output: dict[str, Any] | None = Field(
        default=None,
        description="Structured tool output. Free-form per tool.",
    )
    error: str | None = Field(
        default=None,
        description="Stringified exception if the tool failed.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Wall-clock latency of the tool call in milliseconds.",
    )

    @property
    def succeeded(self) -> bool:
        return self.error is None
