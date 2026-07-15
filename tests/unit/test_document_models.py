"""Unit tests for Phase 1 document/retrieval model additions.

These tests focus on what's *new* in Phase 1:
* `DocumentChunk.created_at` auto-population.
* `RetrievedDoc.metadata` and `RetrievedDoc.similarity_score` property.
* The re-export path `from src.models.document import RetrievedDoc`.

The Phase 0 model tests in `test_models.py` already cover the rest of
the schema, so we don't duplicate them here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.conversation import RetrievedDoc
from src.models.document import DocumentChunk, DocumentMetadata
from src.models.document import RetrievedDoc as ReExportedRetrievedDoc


class TestDocumentChunkPhase1:
    def _meta(self) -> DocumentMetadata:
        return DocumentMetadata(source="/tmp/notes.pdf", title="notes", page=2)

    def test_created_at_auto_populated_and_utc(self) -> None:
        before = datetime.now(timezone.utc)
        chunk = DocumentChunk(
            doc_id="doc_1",
            chunk_id="doc_1::chunk_0000",
            chunk_index=0,
            text="hello world",
            metadata=self._meta(),
        )
        after = datetime.now(timezone.utc)

        assert chunk.created_at.tzinfo is not None
        assert before <= chunk.created_at <= after

    def test_required_fields_validated(self) -> None:
        # `text` is min_length=1
        with pytest.raises(ValidationError):
            DocumentChunk(
                doc_id="d",
                chunk_id="c",
                chunk_index=0,
                text="",
                metadata=self._meta(),
            )

    def test_round_trip_includes_created_at(self) -> None:
        chunk = DocumentChunk(
            doc_id="d1",
            chunk_id="d1::chunk_0001",
            chunk_index=1,
            text="hello",
            metadata=self._meta(),
        )
        encoded = chunk.model_dump_json()
        # `created_at` should appear in the JSON form (datetimes → ISO strings).
        assert "created_at" in json.loads(encoded)
        restored = DocumentChunk.model_validate_json(encoded)
        assert restored == chunk


class TestRetrievedDocPhase1:
    def _doc(self, **overrides) -> RetrievedDoc:
        defaults = dict(
            chunk_id="c1",
            doc_id="d1",
            source="paper.pdf",
            text="RAG combines retrieval with generation.",
            score=0.82,
            rank=0,
        )
        defaults.update(overrides)
        return RetrievedDoc(**defaults)

    def test_metadata_defaults_to_empty_dict(self) -> None:
        doc = self._doc()
        assert doc.metadata == {}

    def test_metadata_round_trip(self) -> None:
        doc = self._doc(metadata={"page": 5, "section": "Intro"})
        restored = RetrievedDoc.model_validate_json(doc.model_dump_json())
        assert restored.metadata == {"page": 5, "section": "Intro"}

    def test_similarity_score_property_mirrors_score(self) -> None:
        doc = self._doc(score=0.73)
        # The new alias should return the same value as `score`.
        assert doc.similarity_score == 0.73
        assert doc.similarity_score == doc.score

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, -0.1, 1.5])
    def test_score_accepts_any_float(self, score: float) -> None:
        # Score isn't bounded — clamp lives in the vector store, not here.
        # The model accepts any float so we can record raw distances during
        # Phase 1 experiments without ValidationError noise.
        doc = self._doc(score=score)
        assert doc.score == score

    def test_reexport_from_document_module_is_same_class(self) -> None:
        # The spec asks callers to import from `src.models.document`, so
        # the re-export must point to the same class object (not a copy).
        assert ReExportedRetrievedDoc is RetrievedDoc
