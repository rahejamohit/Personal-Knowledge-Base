"""Token-budgeted document chunking.

Phase 1.1 fills in the bodies; this module defines the constants and
function contracts so Phase 1.1's `IngestionPipeline` can be written
against a stable interface.

Why these defaults (512/50/2048)?
---------------------------------
* `CHUNK_SIZE=512` keeps ~5 retrieved chunks under 3K tokens, comfortably
  inside Gemini's prompt budget once history is added.
* `CHUNK_OVERLAP=50` (~10%) is the rule-of-thumb that bridges most
  sentence-boundary cuts without spending much extra budget.
* `MAX_CHUNK_SIZE=2048` is a guardrail for the recursive splitter: if a
  paragraph is bigger than this, force a split even if no natural
  boundary is found, to keep one giant chunk from poisoning retrieval.

The same defaults live on `Settings.chunk_size` / `Settings.chunk_overlap`
so they can be tuned per-deployment via env vars without code changes.
The constants here are the *library defaults*; production callers pass
`settings.chunk_size` explicitly.
"""

from __future__ import annotations

from src.models.document import DocumentMetadata
from src.utils import get_logger

logger = get_logger(__name__)


CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
MAX_CHUNK_SIZE: int = 2048


async def recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into ~`chunk_size`-token chunks with `overlap` overlap.

    Phase 1.1 implementation plan:
    * Use `tiktoken` (cl100k base) for accurate token counting — char
      counts diverge from token counts enough to matter at the 512 mark.
    * Recursively split on paragraph → sentence → word boundaries (the
      pattern LangChain's `RecursiveCharacterTextSplitter` popularized)
      so chunks land on semantic seams rather than mid-word.
    * Cap any single chunk at `MAX_CHUNK_SIZE` even if no clean split
      point is found.

    Args:
        text: Full document text to split.
        chunk_size: Target chunk size in tokens.
        overlap: Tokens of overlap between consecutive chunks. The
            overlap is taken from the END of the previous chunk and
            prepended to the next — preserves cross-chunk context for
            retrieval.

    Returns:
        Ordered list of text chunks ready to be embedded.

    Raises:
        NotImplementedError: until Phase 1.1.
        ValueError: `chunk_size <= overlap` (would loop forever).
    """
    raise NotImplementedError("Phase 1.1 implements chunking.")


def extract_metadata_for_chunk(
    original_text: str,
    chunk_index: int,
    source_file: str,
    doc_id: str,
) -> DocumentMetadata:
    """Build a `DocumentMetadata` instance for a single chunk.

    Phase 1.1 implementation plan, by format:
    * **PDF**: find the chunk's character offset inside the page-joined
      text, map it back to a page number, and look for the nearest
      heading above.
    * **Markdown**: walk back from the chunk to the most recent `##`
      header and use that as `section`.
    * **Any format**: always set `source` (filename) and `ingested_at`
      (UTC `datetime.now()`), and infer `content_type` from the
      extension.

    Args:
        original_text: The full document text the chunk came from. The
            implementation uses character-offset lookups against this
            to recover page/section info.
        chunk_index: 0-indexed position of the chunk within the doc.
        source_file: Path or URL of the source — stored as
            `DocumentMetadata.source` and used for citations.
        doc_id: ID of the parent document (the chunk's `doc_id` field).

    Returns:
        A populated `DocumentMetadata` instance ready to attach to a
        `DocumentChunk`.

    Raises:
        NotImplementedError: until Phase 1.1.
    """
    raise NotImplementedError("Phase 1.1 implements metadata extraction.")
