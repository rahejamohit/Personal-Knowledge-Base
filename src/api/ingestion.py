"""Internal debug endpoints for the Phase 1.1 ingestion pipeline.

**INTERNAL ONLY.** The single route here — `POST
/api/internal/debug/documents/ingest` — exposes raw chunker output
(text, token counts, internal IDs, loader class name) for manual
validation while Phase 1 is being built. The production retrieval
endpoint will live elsewhere with a sanitized contract; do NOT proxy
this route to end users.

We isolate this in its own `APIRouter` so the eventual production
ingest endpoint can stand alongside it without sharing schemas.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.chunker import _estimate_tokens, chunk_document
from src.ingestion.loaders import get_loader
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["internal-debug"])


# ─── Request / response models ───────────────────────────────────────────


class IngestDebugRequest(BaseModel):
    """Request body for the debug ingestion endpoint."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Local filesystem path to ingest. Resolved relative to the "
            "server's working directory."
        ),
    )
    pdf_type: Literal["text", "scanned", "auto"] = Field(
        default="auto",
        description=(
            "PDFs only — picks the loader strategy. "
            "`text`: fast text extraction, no OCR. "
            "`scanned`: batch OCR for image-only PDFs. "
            "`auto` (default): per-page hybrid — text extraction with "
            "OCR fallback for pages whose text layer is missing. "
            "Ignored for non-PDF files."
        ),
    )
    ocr_language: str = Field(
        default="eng",
        min_length=2,
        description=(
            "Tesseract language code passed to whichever loader does "
            "OCR. Defaults to English (`eng`). Examples: `deu` (German), "
            "`fra` (French). Ignored when no OCR runs."
        ),
    )


class ChunkDetail(BaseModel):
    """One chunk in the debug response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any]


class IngestDebugResponse(BaseModel):
    """Full debug payload — internal IDs + per-chunk content + stats."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    file_path: str
    loader_used: str
    chunks: list[ChunkDetail]
    stats: dict[str, Any]


# ─── Endpoint ────────────────────────────────────────────────────────────


@router.post(
    "/api/internal/debug/documents/ingest",
    response_model=IngestDebugResponse,
    summary="DEBUG: load + chunk a local file (internal)",
)
async def ingest_debug(request: IngestDebugRequest) -> IngestDebugResponse:
    """Run the loader + chunker on `file_path` and return the raw output.

    Status codes:

    * **200** — file ingested; response contains every chunk + stats.
    * **400** — file missing, unsupported extension, or malformed
      content. The `detail` field carries the human-readable reason.
    * **500** — anything else (genuine bug or transport failure). The
      server-side log has the traceback; the response carries a brief
      message.

    This endpoint is intentionally synchronous-end-to-end (no async
    polling). Phase 1.1 files are small; the production endpoint will
    queue large files.
    """
    start = perf_counter()
    file_path = request.file_path

    try:
        loader = get_loader(
            file_path,
            pdf_type=request.pdf_type,
            ocr_language=request.ocr_language,
        )
    except ValueError as e:
        # Bad extension or invalid `pdf_type` — surface as 400.
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        text_blocks = await loader.load(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}") from e
    except ImportError as e:
        # OCR deps (pytesseract / pdf2image) not installed — surface as
        # 400 with the loader's actionable install instructions rather
        # than a generic 500.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (UnicodeDecodeError, ValueError) as e:
        # Malformed file (corrupted PDF, non-UTF-8 markdown, etc.).
        raise HTTPException(status_code=400, detail=f"Failed to read {file_path}: {e}") from e
    except Exception as e:  # noqa: BLE001 — defensive final fence
        logger.exception("Unexpected loader error for %s", file_path)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}") from e

    full_text = "\n\n".join(text_blocks)

    try:
        doc_chunks = await chunk_document(text=full_text, source=file_path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected chunker error for %s", file_path)
        raise HTTPException(status_code=500, detail=f"Failed to chunk document: {e}") from e

    chunk_details = [
        ChunkDetail(
            id=chunk.chunk_id,
            content=chunk.text,
            chunk_index=chunk.chunk_index,
            token_count=_estimate_tokens(chunk.text),
            # `mode="json"` so datetimes are ISO strings rather than
            # raw `datetime` objects (which can't be JSON-serialized
            # without further work).
            metadata=chunk.metadata.model_dump(mode="json"),
        )
        for chunk in doc_chunks
    ]
    total_tokens = sum(c.token_count for c in chunk_details)
    processing_time_ms = (perf_counter() - start) * 1000
    document_id = doc_chunks[0].doc_id if doc_chunks else f"empty_{int(start)}"
    loader_used = loader.__class__.__name__

    logger.info(
        "Ingested %s with %s: %d chunks, %d tokens, %.1fms",
        file_path,
        loader_used,
        len(chunk_details),
        total_tokens,
        processing_time_ms,
    )

    return IngestDebugResponse(
        document_id=document_id,
        file_path=file_path,
        loader_used=loader_used,
        chunks=chunk_details,
        stats={
            "total_chunks": len(chunk_details),
            "total_tokens": total_tokens,
            "avg_chunk_tokens": (
                total_tokens / len(chunk_details) if chunk_details else 0.0
            ),
            "processing_time_ms": processing_time_ms,
        },
    )
