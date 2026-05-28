"""Unit tests for the Pydantic data models (Task 0.2).

These tests verify the *contracts* the rest of the system relies on:
* JSON round-tripping (sessions persist via JSON-serialized turns)
* Validation rejects nonsense (turn_number=0, empty text, etc.)
* Computed fields stay consistent (TokenUsage.total)
* `DocumentMetadata.to_chroma_metadata()` only emits scalars
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import (
    ConversationTurn,
    DocumentChunk,
    DocumentMetadata,
    MemorySummary,
    RetrievedDoc,
    Session,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from src.models.memory import ExtractedFact


class TestTokenUsage:
    def test_total_is_sum(self) -> None:
        u = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_addition_merges(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5)
        b = TokenUsage(prompt_tokens=20, completion_tokens=8)
        c = a + b
        assert c.prompt_tokens == 30
        assert c.completion_tokens == 13
        assert c.total_tokens == 43

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)


class TestToolCallAndResult:
    def test_tool_call_round_trip(self) -> None:
        call = ToolCall(tool_name="retrieve", arguments={"query": "RAG"})
        restored = ToolCall.model_validate_json(call.model_dump_json())
        assert restored.tool_name == "retrieve"
        assert restored.arguments == {"query": "RAG"}

    def test_result_success_state(self) -> None:
        ok = ToolResult(tool_name="retrieve", output={"docs": []}, latency_ms=12.4)
        assert ok.succeeded
        fail = ToolResult(tool_name="retrieve", error="timeout")
        assert not fail.succeeded


class TestDocumentMetadata:
    def test_chroma_metadata_is_scalar_only(self) -> None:
        meta = DocumentMetadata(
            source="/tmp/a.pdf",
            title="A",
            page=3,
            section="Intro",
            ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        chroma = meta.to_chroma_metadata()
        # No nested values, no None
        for v in chroma.values():
            assert isinstance(v, (str, int, float, bool)), f"non-scalar: {v!r}"
        # datetime got coerced to a string
        assert isinstance(chroma["ingested_at"], str)

    def test_extra_tags_allowed(self) -> None:
        # `extra="allow"` so users can attach custom tags via `metadata={"tag": ...}`
        meta = DocumentMetadata(source="x", tag="personal")
        assert getattr(meta, "tag") == "personal"


class TestDocumentChunk:
    def test_basic(self) -> None:
        chunk = DocumentChunk(
            doc_id="doc_1",
            chunk_id="doc_1::chunk_0000",
            chunk_index=0,
            text="hello world",
            metadata=DocumentMetadata(source="/tmp/a.txt"),
        )
        assert chunk.embedding is None  # not yet embedded

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DocumentChunk(
                doc_id="d",
                chunk_id="c",
                chunk_index=0,
                text="",
                metadata=DocumentMetadata(source="x"),
            )


class TestConversationModels:
    def _make_turn(self) -> ConversationTurn:
        return ConversationTurn(
            id="turn_1",
            session_id="sess_1",
            turn_number=1,
            user_message="What is RAG?",
            agent_response="Retrieval-Augmented Generation is...",
            retrieved_docs=[
                RetrievedDoc(
                    chunk_id="c1",
                    doc_id="d1",
                    source="rag_paper.pdf",
                    text="RAG combines retrieval with generation.",
                    score=0.91,
                    rank=0,
                ),
            ],
            tool_calls=[ToolCall(tool_name="retrieve", arguments={"query": "RAG"})],
            tool_results=[
                ToolResult(tool_name="retrieve", output={"n": 1}, latency_ms=42.0),
            ],
            token_usage=TokenUsage(prompt_tokens=500, completion_tokens=200),
        )

    def test_round_trip_json(self) -> None:
        turn = self._make_turn()
        # Use mode="json" so datetimes serialize as ISO strings
        payload = turn.model_dump(mode="json")
        # Must be JSON-serializable end-to-end (this is how SQLite will store it)
        encoded = json.dumps(payload)
        restored = ConversationTurn.model_validate_json(encoded)
        assert restored == turn

    def test_turn_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(
                id="t",
                session_id="s",
                turn_number=0,  # 1-indexed; 0 is invalid
                user_message="hi",
                agent_response="hi",
            )

    def test_unknown_fields_rejected(self) -> None:
        # extra="forbid" — typos in field names should error loudly, not silently lose data
        with pytest.raises(ValidationError):
            ConversationTurn(
                id="t",
                session_id="s",
                turn_number=1,
                user_message="hi",
                agent_response="hi",
                surprise_field="oops",  # type: ignore[call-arg]
            )

    def test_session_defaults(self) -> None:
        s = Session(id="sess_1")
        assert s.user_id == "local"
        assert s.created_at.tzinfo is not None  # always tz-aware


class TestMemoryModels:
    def test_summary_validates_range(self) -> None:
        s = MemorySummary(
            session_id="s",
            turn_range_start=1,
            turn_range_end=10,
            summary="user discussed RAG",
            topics=["rag", "retrieval"],
        )
        assert s.token_count == 0

    def test_fact_confidence_bounded(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFact(
                fact_id="f",
                session_id="s",
                statement="x",
                confidence=1.5,  # > 1.0 is invalid
            )
