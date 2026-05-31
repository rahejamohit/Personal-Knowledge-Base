"""Document and chunk models.

Phase 1's ingestion pipeline produces `DocumentChunk` objects which are then
embedded and upserted into Chroma. We separate *content* (`text`) from
*provenance* (`metadata`) so the metadata round-trips cleanly through Chroma's
JSON-only metadata column.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class DocumentMetadata(BaseModel):
    """Metadata attached to every chunk.

    Kept flat (no nested models) so it serializes cleanly to Chroma metadata,
    which only accepts str/int/float/bool values.
    """

    model_config = ConfigDict(extra="allow")  # allow user-supplied tags

    source: str = Field(..., description="Original file path or URL.")
    title: str | None = None
    page: int | None = Field(default=None, ge=0, description="0-indexed page (PDFs).")
    section: str | None = Field(default=None, description="Section header if extractable.")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_type: str = Field(
        default="text/plain",
        description="MIME type of the source file.",
    )

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Flatten to a Chroma-compatible dict (only scalar values).

        Chroma rejects nested dicts and non-scalar values, so we coerce
        datetimes to ISO strings and drop `None`s.
        """
        raw = self.model_dump(mode="json", exclude_none=True)
        # `model_dump(mode="json")` already converts datetimes to ISO strings.
        return {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool))}


class DocumentChunk(BaseModel):
    """A single chunk of a document, ready to be embedded.

    `embedding` is optional because the chunker produces chunks *before* the
    embedding step. The vector store layer fills it in.

    `created_at` is filled in automatically — useful when comparing which
    chunks were live in the index for a given conversation turn (Phase 2
    audit/replay).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="ID of the parent document.")
    chunk_id: str = Field(..., description="Globally unique chunk ID.")
    chunk_index: NonNegativeInt = Field(..., description="0-indexed position within the doc.")
    text: str = Field(..., min_length=1, description="The chunk's textual content.")
    embedding: list[float] | None = Field(
        default=None,
        description="Embedding vector — populated after the embedding step.",
    )
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Re-export so callers can `from src.models.document import RetrievedDoc` —
# the type is defined in conversation.py because `ConversationTurn` references
# it; here we just expose it under the more natural "document" namespace.
from src.models.conversation import RetrievedDoc  # noqa: E402, F401  (re-export)
