"""Unit tests for `src.storage.schema.ChunkSchema`.

`ChunkSchema` is the single source of truth for the `DocumentChunk` →
Chroma mapping. These tests pin its two guarantees:

* **scalar-only metadata** — `str`/`int`/`float`/`bool` kept *as-is*
  (we do NOT stringify numbers — Chroma stores them natively), `datetime`
  coerced to ISO strings, `None` dropped;
* **fail loud** — any non-scalar value raises `TypeError`, rather than
  being silently dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.models.document import DocumentChunk, DocumentMetadata
from src.storage.schema import ChunkSchema


class TestValidateMetadata:
    def test_strings_pass_through(self) -> None:
        result = ChunkSchema.validate_metadata(
            {"source": "test.md", "content_type": "text/markdown"}
        )
        assert result == {"source": "test.md", "content_type": "text/markdown"}

    def test_int_kept_as_int(self) -> None:
        # Chroma stores ints natively — we must NOT stringify them.
        result = ChunkSchema.validate_metadata({"page": 5})
        assert result["page"] == 5
        assert isinstance(result["page"], int)

    def test_float_kept_as_float(self) -> None:
        result = ChunkSchema.validate_metadata({"score": 0.95})
        assert result["score"] == 0.95
        assert isinstance(result["score"], float)

    def test_bool_preserved(self) -> None:
        # `bool` is an `int` subclass — verify it stays a real bool.
        result = ChunkSchema.validate_metadata({"is_active": True})
        assert result["is_active"] is True

    def test_datetime_to_iso_string(self) -> None:
        now = datetime(2026, 6, 7, 10, 30, 0, tzinfo=UTC)
        result = ChunkSchema.validate_metadata({"ingested_at": now})
        assert isinstance(result["ingested_at"], str)
        assert result["ingested_at"].startswith("2026-06-07")
        assert "T" in result["ingested_at"]  # ISO 8601

    def test_none_dropped(self) -> None:
        result = ChunkSchema.validate_metadata({"field": None, "source": "test.md"})
        assert "field" not in result
        assert result["source"] == "test.md"

    @pytest.mark.parametrize("bad_value", [["a", "b"], {"nested": "dict"}, {1, 2}, object()])
    def test_non_scalar_raises_typeerror(self, bad_value: object) -> None:
        with pytest.raises(TypeError) as exc_info:
            ChunkSchema.validate_metadata({"bad_field": bad_value})
        msg = str(exc_info.value)
        assert "unsupported type" in msg
        assert "bad_field" in msg

    def test_empty_metadata(self) -> None:
        assert ChunkSchema.validate_metadata({}) == {}


class TestChunkToChromaFormat:
    def _chunk(self, **meta_kwargs: object) -> DocumentChunk:
        return DocumentChunk(
            doc_id="test_doc",
            chunk_id="test_doc_0000",
            chunk_index=0,
            text="Sample chunk text",
            metadata=DocumentMetadata(source="test.md", **meta_kwargs),
        )

    def test_basic_shape(self) -> None:
        result = ChunkSchema.chunk_to_chroma_format(self._chunk())
        assert result["chunk_id"] == "test_doc_0000"
        assert result["text"] == "Sample chunk text"
        assert isinstance(result["metadata"], dict)
        # Auto fields added; chunk_id is the Chroma id, not duplicated here.
        assert result["metadata"]["doc_id"] == "test_doc"
        assert result["metadata"]["chunk_index"] == 0
        assert "chunk_id" not in result["metadata"]

    def test_chunk_index_stays_int(self) -> None:
        chunk = DocumentChunk(
            doc_id="d",
            chunk_id="d_0005",
            chunk_index=5,
            text="x",
            metadata=DocumentMetadata(source="f.md"),
        )
        result = ChunkSchema.chunk_to_chroma_format(chunk)
        assert result["metadata"]["chunk_index"] == 5
        assert isinstance(result["metadata"]["chunk_index"], int)

    def test_all_values_are_scalar(self) -> None:
        # `ingested_at` defaults to a datetime; everything in the output must
        # be a Chroma scalar afterwards.
        chunk = self._chunk(title="Title", page=10, section="Section")
        result = ChunkSchema.chunk_to_chroma_format(chunk)
        for key, value in result["metadata"].items():
            assert isinstance(value, (str, int, float, bool)), (
                f"metadata[{key!r}] is {type(value).__name__}, not scalar"
            )
        # Spot-check native typing is preserved through the pipeline.
        assert result["metadata"]["source"] == "test.md"
        assert result["metadata"]["page"] == 10
        assert isinstance(result["metadata"]["page"], int)
        assert isinstance(result["metadata"]["ingested_at"], str)

    def test_non_scalar_extra_field_raises(self) -> None:
        # `DocumentMetadata` allows extra fields; a non-scalar one must be
        # rejected loudly rather than silently dropped.
        chunk = self._chunk(tags=["personal", "rag"])
        with pytest.raises(TypeError, match="tags"):
            ChunkSchema.chunk_to_chroma_format(chunk)
