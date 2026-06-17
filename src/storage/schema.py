"""Schema for mapping a `DocumentChunk` onto Chroma's data model.

This is the **single source of truth** for how a Phase 1.1 `DocumentChunk`
becomes a Chroma row, and the one place that enforces Chroma's metadata
constraints. `DocumentMetadata.to_chroma_metadata` and
`ChromaVectorStore` both route through here, so validation can't be
bypassed by storing chunks via a different path.

Chroma's contract
-----------------
* The **id** is the chunk's `chunk_id` (Chroma's primary key — not
  duplicated into metadata).
* The **document** is the chunk's `text`.
* **metadata** values must be scalars: ``str | int | float | bool``.
  Anything else (lists, dicts, nested models) is rejected.

Conversions we apply
--------------------
* `datetime` → ISO-8601 string (Chroma stores no datetime type).
* `None` → dropped (Chroma has no null; absence is the signal).
* `str | int | float | bool` → kept **as-is**. We deliberately do *not*
  stringify numbers: Chroma stores `int`/`float` natively and numeric
  metadata stays filterable (`where={"page": {"$gte": 3}}`).
* Anything else → `TypeError`, raised loudly rather than silently
  dropped, so a bad metadata field is caught at ingest time with an
  actionable message instead of vanishing from the index.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a runtime import cycle (document.py imports this module)
    from src.models.document import DocumentChunk

# Scalar types Chroma accepts in a metadata value, plus the conversions
# `validate_metadata` knows how to apply.
SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool)


class ChunkSchema:
    """Convert + validate `DocumentChunk` data for Chroma storage."""

    # Auto-added at storage time for grouping/filtering. `chunk_id` is NOT
    # duplicated here — it's Chroma's primary id, recovered on read.
    AUTO_METADATA_FIELDS = ("doc_id", "chunk_index")

    @staticmethod
    def validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a Chroma-compatible copy of `metadata`.

        Drops ``None``, coerces `datetime` to ISO strings, and keeps scalar
        values (``str``/``int``/``float``/``bool``) unchanged.

        Raises:
            TypeError: if any value is a non-scalar, non-datetime type
                (e.g. a list or dict). Chroma would reject it; we surface
                it early with the offending key, type, and value.
        """
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            # NOTE: `bool` is a subclass of `int`; this single branch keeps
            # all scalars as-is, so there's no risk of stringifying `True`.
            elif isinstance(value, SCALAR_TYPES):
                cleaned[key] = value
            else:
                raise TypeError(
                    f"Metadata field {key!r} has unsupported type "
                    f"{type(value).__name__}. Chroma only accepts scalar "
                    f"metadata (str, int, float, bool) or datetime; "
                    f"got value: {value!r}"
                )
        return cleaned

    @staticmethod
    def chunk_to_chroma_format(chunk: DocumentChunk) -> dict[str, Any]:
        """Map a `DocumentChunk` to ``{chunk_id, text, metadata}`` for Chroma.

        The ``metadata`` is validated (see :meth:`validate_metadata`) and
        augmented with the auto fields (`doc_id`, `chunk_index`). This is
        exactly what `ChromaVectorStore` stores per chunk.

        Raises:
            TypeError: if the chunk's metadata holds a non-scalar value.
        """
        metadata = ChunkSchema.validate_metadata(chunk.metadata.model_dump())
        # Native types — Chroma stores them as-is and keeps them filterable.
        metadata["doc_id"] = chunk.doc_id
        metadata["chunk_index"] = chunk.chunk_index
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": metadata,
        }
