"""Unit tests for the Phase 1.1 chunker.

Two layers exercised:

* **Private helpers** (`_estimate_tokens`, `_get_last_n_tokens`,
  `_extract_code_blocks`, `_recursive_split`) — we test these directly
  because the chunker's correctness hinges on them and they're the
  units Phase 1.3 will reimplement against `tiktoken`.
* **`chunk_document`** — the public entry point. Confirms metadata,
  IDs, and overlap come out right end-to-end.

We deliberately use small `chunk_size` values in many tests (e.g. 64)
to force the splitter to actually split — the default 512 is too big
to exercise the recursion against test-sized inputs.
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _estimate_tokens,
    _extract_code_blocks,
    _generate_doc_id,
    _get_last_n_tokens,
    _infer_content_type,
    _recursive_split,
    _restore_code_blocks,
    chunk_document,
)
from src.models.document import DocumentChunk


# ─── Token estimation ────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_zero_chars_zero_tokens(self) -> None:
        assert _estimate_tokens("") == 0

    def test_uses_chars_per_token_constant(self) -> None:
        text = "a" * 450  # 450 / 4.5 = 100
        assert _estimate_tokens(text) == 100

    def test_get_last_n_tokens_returns_tail(self) -> None:
        text = "x" * 1000
        tail = _get_last_n_tokens(text, n_tokens=50)
        assert len(tail) == int(50 * CHARS_PER_TOKEN)
        assert tail == "x" * len(tail)

    def test_get_last_n_tokens_caps_at_text_length(self) -> None:
        text = "short"
        # Asking for 100 tokens of a 5-char string returns the whole thing.
        assert _get_last_n_tokens(text, n_tokens=100) == text


# ─── Code block extraction / restoration ─────────────────────────────────


class TestCodeBlocks:
    def test_single_block_replaced_with_marker(self) -> None:
        text = "Intro\n```python\nprint('hi')\n```\nEnd"
        prose, blocks = _extract_code_blocks(text)
        assert prose == "Intro\n[CODE_BLOCK_0]\nEnd"
        assert blocks == ["```python\nprint('hi')\n```"]

    def test_multiple_blocks_numbered_in_order(self) -> None:
        text = "a ```b``` c ```d``` e"
        prose, blocks = _extract_code_blocks(text)
        assert prose == "a [CODE_BLOCK_0] c [CODE_BLOCK_1] e"
        assert blocks == ["```b```", "```d```"]

    def test_no_blocks_returns_input_unchanged(self) -> None:
        text = "no code here"
        prose, blocks = _extract_code_blocks(text)
        assert prose == text
        assert blocks == []

    def test_restore_round_trips(self) -> None:
        text = "before ```a\nb\n``` after"
        prose, blocks = _extract_code_blocks(text)
        restored = _restore_code_blocks(prose, blocks)
        assert restored == text

    def test_restore_leaves_orphan_markers(self) -> None:
        # If a chunk holds a marker whose code block isn't in its
        # `code_blocks` list (shouldn't happen in practice, but the
        # implementation should not blow up), leave it alone.
        restored = _restore_code_blocks("hello [CODE_BLOCK_5] world", [])
        assert restored == "hello [CODE_BLOCK_5] world"


# ─── Recursive split ─────────────────────────────────────────────────────


class TestRecursiveSplit:
    async def test_small_text_returns_single_chunk(self) -> None:
        text = "Hello world. This is short."
        chunks = await _recursive_split(text, chunk_size=512)
        assert chunks == [text]

    async def test_large_text_splits_on_paragraphs(self) -> None:
        # 50 paragraphs ≈ 1500 chars ≈ 333 tokens — far under 512, so we
        # use a small chunk_size to make sure splitting actually fires.
        paragraphs = "\n\n".join(
            f"Paragraph {i} with some content." for i in range(50)
        )
        chunks = await _recursive_split(paragraphs, chunk_size=64, overlap=0)
        assert len(chunks) > 1

    async def test_each_chunk_within_budget_for_well_formed_input(self) -> None:
        text = "\n\n".join([f"Section {i}. " * 20 for i in range(30)])
        chunk_size = 200
        chunks = await _recursive_split(text, chunk_size=chunk_size, overlap=0)
        # All chunks should fit within ~1.5x the budget (slack for the
        # last-resort char split + paragraph packing). Exact equality
        # isn't achievable without a real tokenizer.
        for chunk in chunks:
            assert _estimate_tokens(chunk) <= int(chunk_size * 1.5)

    async def test_overlap_prefixes_next_chunk_with_tail_of_previous(self) -> None:
        # Build input long enough to force >1 chunks.
        paragraphs = "\n\n".join(f"Paragraph {i} content." for i in range(200))
        chunks = await _recursive_split(paragraphs, chunk_size=64, overlap=10)
        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            tail = _get_last_n_tokens(chunks[i - 1], n_tokens=10)
            assert chunks[i].startswith(tail), (
                f"chunk {i} does not start with previous tail.\n"
                f"  tail:   {tail!r}\n"
                f"  chunk:  {chunks[i][:80]!r}"
            )

    async def test_zero_overlap_disables_overlap(self) -> None:
        text = "\n\n".join(f"Para {i} content." for i in range(50))
        chunks = await _recursive_split(text, chunk_size=64, overlap=0)
        if len(chunks) >= 2:
            # No overlap means chunk 1 doesn't repeat the end of chunk 0.
            tail = _get_last_n_tokens(chunks[0], n_tokens=10)
            assert not chunks[1].startswith(tail)

    async def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            await _recursive_split("hi", chunk_size=10, overlap=10)

    async def test_negative_overlap_rejected(self) -> None:
        with pytest.raises(ValueError):
            await _recursive_split("hi", chunk_size=512, overlap=-1)


# ─── Public chunk_document ──────────────────────────────────────────────


class TestChunkDocument:
    async def test_empty_text_returns_empty_list(self) -> None:
        assert await chunk_document("", source="x.md") == []
        assert await chunk_document("   \n  ", source="x.md") == []

    async def test_small_doc_one_chunk(self) -> None:
        text = "A short note about RAG."
        chunks = await chunk_document(text, source="note.md")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.text == text
        assert chunk.chunk_index == 0
        assert chunk.metadata.source == "note.md"
        assert chunk.metadata.content_type == "text/markdown"

    async def test_chunk_ids_zero_padded_and_unique(self) -> None:
        text = "\n\n".join(f"Paragraph {i} content." for i in range(100))
        chunks = await chunk_document(text, source="long.md", chunk_size=64, overlap=0)
        assert len(chunks) > 1
        ids = [c.chunk_id for c in chunks]
        # Zero-padded 4-digit suffix.
        for i, cid in enumerate(ids):
            assert cid.endswith(f"_{i:04d}"), f"chunk_id {cid!r} not zero-padded"
        # All unique.
        assert len(set(ids)) == len(ids)

    async def test_all_chunks_share_doc_id(self) -> None:
        text = "\n\n".join(f"Para {i}." for i in range(50))
        chunks = await chunk_document(text, source="x.md", chunk_size=64, overlap=0)
        doc_ids = {c.doc_id for c in chunks}
        assert len(doc_ids) == 1

    async def test_code_blocks_are_atomic(self) -> None:
        # The code block is large enough that, without protection, the
        # splitter would slice it in half. Confirm the block survives
        # intact in at least one chunk.
        code = "```python\n" + "print('x')\n" * 60 + "```"
        prose = "Intro paragraph.\n\n" + code + "\n\nClosing paragraph."
        chunks = await chunk_document(
            prose, source="doc.md", chunk_size=64, overlap=0
        )
        joined = "\n".join(c.text for c in chunks)
        # The code block appears verbatim somewhere in the output.
        assert code in joined, "code block was split despite atomicity guarantee"

    async def test_content_type_inferred_from_extension(self) -> None:
        chunks = await chunk_document("hi", source="a.pdf")
        assert chunks[0].metadata.content_type == "application/pdf"
        chunks = await chunk_document("hi", source="a.txt")
        assert chunks[0].metadata.content_type == "text/plain"

    async def test_extra_source_metadata_passed_through(self) -> None:
        chunks = await chunk_document(
            "hi",
            source="a.md",
            source_metadata={"page": 5, "section": "Intro"},
        )
        meta = chunks[0].metadata
        assert getattr(meta, "page") == 5
        assert getattr(meta, "section") == "Intro"


# ─── ID + content-type helpers ───────────────────────────────────────────


class TestDocIdAndContentType:
    def test_doc_id_format(self) -> None:
        # `rag_guide.md` → `rag_guide_md_<ts>`
        doc_id = _generate_doc_id("docs/rag_guide.md")
        assert doc_id.startswith("rag_guide_md_")
        # Suffix is a 10-digit unix timestamp.
        assert doc_id.split("_")[-1].isdigit()

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("a.pdf", "application/pdf"),
            ("README.md", "text/markdown"),
            ("notes.markdown", "text/markdown"),
            ("plain.txt", "text/plain"),
            ("doc.docx", (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )),
            ("mystery.xyz", "application/octet-stream"),
        ],
    )
    def test_infer_content_type(self, source: str, expected: str) -> None:
        assert _infer_content_type(source) == expected


# ─── Module-level constants are unchanged from spec ──────────────────────


def test_constants_match_spec() -> None:
    assert CHUNK_SIZE == 512
    assert CHUNK_OVERLAP == 50
    assert CHARS_PER_TOKEN == 4.5
