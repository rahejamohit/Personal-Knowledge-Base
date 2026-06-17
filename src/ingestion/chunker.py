"""Token-budgeted document chunking with sliding-window overlap.

Public surface:

* `chunk_document(text, source, ...) -> list[DocumentChunk]` — the
  pipeline's main entry point. Wraps the splitter, attaches metadata,
  and emits `DocumentChunk` objects ready for embedding.
* `extract_metadata_for_chunk(...)` — Phase 1.1 stub kept around for
  Phase 1.2's per-format metadata enrichment.

Private helpers (`_extract_code_blocks`, `_recursive_split`,
`_estimate_tokens`, `_get_last_n_tokens`) are intentionally not
exported — Phase 1.3 will replace `_estimate_tokens` with `tiktoken`
and may rewrite `_recursive_split` against the real tokenizer.

Token counting
--------------
Phase 1 uses a flat ``CHARS_PER_TOKEN = 4.5`` heuristic. That's about
right for English prose at the scale we care about for splitting
(off by ~10% — the overlap budget absorbs the slack). Phase 1.3 will
swap in `tiktoken`'s `cl100k_base` for accurate counts, no signature
changes.

Code blocks
-----------
Triple-backtick fenced blocks are extracted before splitting, replaced
with `[CODE_BLOCK_N]` markers in the prose, and restored at the end.
The result: code blocks are never split mid-block, even if it makes a
chunk exceed `CHUNK_SIZE`. That's the intended trade-off — splitting a
function mid-line is worse than an oversized chunk.

Overlap
-------
Sliding-window: the last `CHUNK_OVERLAP` tokens of chunk N are
prepended to chunk N+1. Preserves cross-chunk context so a sentence
straddling a boundary isn't lost. The overlap is applied AFTER
recursive splitting so it cleanly stacks.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.document import DocumentChunk, DocumentMetadata
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────


CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
MAX_CHUNK_SIZE: int = 2048
CHARS_PER_TOKEN: float = 4.5


# Boundary priority for recursive splitting. The empty string is the
# last-resort character-level split.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


# ─── Token estimation helpers ────────────────────────────────────────────


def _estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Char-based token estimate. Phase 1.3 will swap in `tiktoken`."""
    return int(len(text) / chars_per_token)


def _get_last_n_tokens(text: str, n_tokens: int) -> str:
    """Return the last ~`n_tokens` worth of characters from `text`.

    Phase 1 heuristic: `n_tokens * CHARS_PER_TOKEN` characters from the
    tail. Not exact, but stable enough for overlap.
    """
    chars_to_keep = int(n_tokens * CHARS_PER_TOKEN)
    return text[-chars_to_keep:] if len(text) > chars_to_keep else text


# ─── Code-block extraction ───────────────────────────────────────────────


# Triple-backtick fenced code blocks, lazy-matched so adjacent blocks
# don't collapse into one. The `[\s\S]` (any char including newlines)
# avoids needing `re.DOTALL` while staying explicit.
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_PLACEHOLDER_RE = re.compile(r"\[CODE_BLOCK_(\d+)\]")


def _extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Pull fenced code blocks out of `text` and replace each with a marker.

    Args:
        text: Source text, possibly containing ```...``` blocks.

    Returns:
        Tuple ``(prose_with_markers, code_blocks)``. Restore order with
        `_restore_code_blocks(prose, code_blocks)`.

    Note: The spec's *type annotation* in the linked design doc shows
    `tuple[list[str], list[str]]` but its *example* shows a single
    prose string. The example matches the natural behavior (one input
    → one prose stream + many code blocks), so we return `tuple[str,
    list[str]]`.
    """
    code_blocks: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"[CODE_BLOCK_{idx}]"

    prose = _CODE_BLOCK_RE.sub(_replace, text)
    return prose, code_blocks


def _restore_code_blocks(text: str, code_blocks: list[str]) -> str:
    """Inverse of `_extract_code_blocks` — substitute markers back."""
    if not code_blocks:
        return text

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(code_blocks):
            return code_blocks[idx]
        # Marker with no matching block: leave it as-is rather than
        # raise — chunk boundaries can sometimes orphan a marker.
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, text)


# ─── Recursive split ─────────────────────────────────────────────────────


def _pack_on_separator(text: str, sep: str, chunk_size: int) -> list[str]:
    """Greedy pack `text.split(sep)` into ~`chunk_size`-token chunks.

    May still emit chunks that exceed `chunk_size` if any individual
    *part* is itself too large (e.g. one paragraph with no internal
    newlines bigger than the budget). The recursive caller will handle
    those by re-splitting with a finer separator.
    """
    parts = list(text) if sep == "" else text.split(sep)
    join_token = sep
    chunks: list[str] = []
    current = ""
    for p in parts:
        candidate = f"{current}{join_token}{p}" if current else p
        if current and _estimate_tokens(candidate) > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_recursive_sync(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """Sync core of `_recursive_split` (no overlap yet).

    Tries separators in priority order. If a chunk still exceeds
    `chunk_size` after packing on the current separator, recurses with
    the next finer one.
    """
    if _estimate_tokens(text) <= chunk_size:
        return [text]
    if not separators:
        # Last resort: hard-truncate at the character boundary. Phase
        # 1.3's tiktoken-based version will at least cut on a token
        # boundary; for now we accept mid-word splits in this edge case.
        size_chars = int(chunk_size * CHARS_PER_TOKEN)
        return [text[i : i + size_chars] for i in range(0, len(text), size_chars)]

    sep, *rest = separators
    parts = _pack_on_separator(text, sep, chunk_size)
    out: list[str] = []
    for part in parts:
        if _estimate_tokens(part) > chunk_size:
            out.extend(_split_recursive_sync(part, chunk_size, rest))
        else:
            out.append(part)
    return out


async def _recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into ~`chunk_size`-token chunks with `overlap` overlap.

    Tries separators in priority order: paragraph → line → sentence →
    word → char. The result is then "windowed": the last `overlap`
    tokens of chunk N are prepended to chunk N+1 to preserve
    cross-chunk context.

    Args:
        text: Document text. Code blocks should already be replaced
            with placeholders by `_extract_code_blocks` before calling.
        chunk_size: Target chunk size in tokens.
        overlap: Tokens of overlap. `0` disables overlap.

    Returns:
        Ordered list of overlapping chunks. Length == 1 when text fits.

    Raises:
        ValueError: `chunk_size <= overlap` (would loop forever).
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    raw = _split_recursive_sync(text, chunk_size, _SEPARATORS)
    if overlap == 0 or len(raw) <= 1:
        return raw

    overlapped: list[str] = [raw[0]]
    for chunk in raw[1:]:
        prefix = _get_last_n_tokens(overlapped[-1], overlap)
        # `+ "\n"` would inject a synthetic newline; avoid that — keep
        # the join clean so consumers can recover original text by
        # stripping the prefix.
        overlapped.append(prefix + chunk)
    return overlapped


# ─── Public entry point ──────────────────────────────────────────────────


_CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


def _infer_content_type(source: str) -> str:
    """Map extension → MIME type. Defaults to `application/octet-stream`."""
    return _CONTENT_TYPES.get(Path(source).suffix.lower(), "application/octet-stream")


def _generate_doc_id(source: str) -> str:
    """`<basename-with-dots-as-underscores>_<unix-secs>`.

    Example: ``"docs/rag_guide.md"`` → ``"rag_guide_md_1717334400"``.

    Why include the timestamp: two ingests of the same file produce
    two distinct doc IDs, so the second doesn't clobber the first in
    the index. If you DO want re-ingestion to overwrite, pass an
    explicit `doc_id` (Phase 1.3 will expose that knob).
    """
    name = Path(source).name.replace(".", "_")
    return f"{name}_{int(time.time())}"


async def chunk_document(
    text: str,
    source: str,
    source_metadata: dict[str, Any] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split a full document into overlapping `DocumentChunk`s.

    Pipeline:
    1. Extract fenced code blocks → markers in prose.
    2. Recursively split the prose with sliding-window overlap.
    3. Restore code blocks into each chunk that holds their marker.
    4. Wrap each chunk in a `DocumentChunk` with `DocumentMetadata`.

    Args:
        text: Full document text.
        source: Source path or URL. Used for `metadata.source` and as
            the basis for `doc_id` / `chunk_id`.
        source_metadata: Extra per-document metadata fields
            (`page`, `section`, `title`, ...) — passed through to
            `DocumentMetadata` via `**kwargs`. Pydantic's
            `extra="allow"` config keeps unknown keys.
        chunk_size: Target tokens per chunk.
        overlap: Sliding-window overlap in tokens.

    Returns:
        Ordered list of `DocumentChunk`s. Empty list iff `text` is
        empty after stripping.
    """
    if not text or not text.strip():
        logger.info("chunk_document: empty input for source=%s", source)
        return []

    prose, code_blocks = _extract_code_blocks(text)
    logger.info("chunk_document: extracted %d code blocks for source=%s", len(code_blocks), source)
    logger.debug("chunk_document: prose length=%d chars", len(prose))
    raw_chunks = await _recursive_split(prose, chunk_size=chunk_size, overlap=overlap)
    restored = [_restore_code_blocks(c, code_blocks) for c in raw_chunks]

    doc_id = _generate_doc_id(source)
    content_type = _infer_content_type(source)
    now = datetime.now(timezone.utc)
    base_meta_kwargs: dict[str, Any] = {
        "source": source,
        "content_type": content_type,
        "ingested_at": now,
    }
    if source_metadata:
        # Avoid overwriting the canonical source-derived fields with
        # caller-supplied junk; everything else gets through.
        for k, v in source_metadata.items():
            base_meta_kwargs.setdefault(k, v)

    chunks: list[DocumentChunk] = []
    for i, chunk_text in enumerate(restored):
        chunk_id = f"{doc_id}_{i:04d}"
        metadata = DocumentMetadata(**base_meta_kwargs)
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=i,
                text=chunk_text,
                metadata=metadata,
                created_at=now,
            )
        )
    logger.info("chunk_document: %s → %d chunks (doc_id=%s)", source, len(chunks), doc_id)
    return chunks


# ─── Legacy stub (kept for back-compat with Phase-1.0 imports) ───────────


def extract_metadata_for_chunk(
    original_text: str,
    chunk_index: int,
    source_file: str,
    doc_id: str,
) -> DocumentMetadata:
    """Phase 1.1 stub — superseded by `chunk_document`'s inline metadata.

    Phase 1.2 will reintroduce this as the per-format enricher: PDF page
    numbers, Markdown section headers, etc. For now `chunk_document`
    builds metadata in-place; calling this helper directly is no longer
    necessary.

    Raises:
        NotImplementedError: Phase 1.2 will implement.
    """
    raise NotImplementedError("Phase 1.2 implements per-format metadata enrichment.")
