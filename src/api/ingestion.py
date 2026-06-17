"""Internal debug endpoint for the Phase 1.1 + 1.2 ingestion pipeline.

**INTERNAL ONLY.** The single route here — `POST
/api/internal/debug/documents/ingest` — runs a document end-to-end and
exposes the raw output for manual validation while Phase 1 is being
built:

    load → chunk → embed → store → echo everything back

It returns internal IDs, per-chunk content, the raw embedding vectors,
and a vector-store confirmation. That makes it a debugging tool, not a
production contract: the production retrieval/ingest endpoint will live
elsewhere with a sanitized payload. Do NOT proxy this route to end users.

We isolate this in its own `APIRouter` so the eventual production ingest
endpoint can stand alongside it without sharing schemas.

Dependency injection
--------------------
The embedding provider and vector store are pulled in via FastAPI
`Depends` (`get_embedder` / `get_vector_store`) rather than constructed
inline, so tests can override them with a fake embedder + a throwaway
Chroma dir and exercise the full wiring without an API key or polluting
the real index.
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.ingestion.chunker import _estimate_tokens, chunk_document
from src.ingestion.loaders import get_loader
from src.providers.base import EmbeddingProvider
from src.providers.factory import get_embedding_provider
from src.storage.ingest import ingest_chunks
from src.storage.vector_store import ChromaVectorStore
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["internal-debug"])

# Collection the debug endpoint writes to. Kept separate from any
# production collection name so debug ingests never mingle with real data.
_DEBUG_COLLECTION = "documents"


# ─── Dependencies (overridable in tests) ─────────────────────────────────


def get_embedder() -> EmbeddingProvider:
    """The configured embedding provider. Overridden in tests with a fake."""
    return get_embedding_provider()


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """A Chroma store rooted at the configured persist dir.

    Cached so every request shares one store instance (and one Chroma
    `PersistentClient`) rather than re-opening the index per call —
    mirroring how `get_embedding_provider()` is memoized. Config is read
    once, at first use; a long-running server's persist dir is fixed.

    Overridden in tests via FastAPI `dependency_overrides` (which bypasses
    this function entirely) to point at a throwaway `tmp_path`, so the
    debug endpoint never writes to the real on-disk index.
    """
    return ChromaVectorStore(
        collection_name=_DEBUG_COLLECTION,
        persist_dir=get_settings().pka_chroma_dir,
    )


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


class VectorStoreInfo(BaseModel):
    """Confirmation of what landed in the vector store."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    chunks_stored: int
    embedding_dimension: int
    embedding_model: str
    persist_dir: str


class IngestDebugResponse(BaseModel):
    """Full debug payload — IDs + per-chunk content + embeddings + stats."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    file_path: str
    loader_used: str
    chunks: list[ChunkDetail]
    embeddings: list[list[float]]
    vector_store: VectorStoreInfo
    stats: dict[str, Any]


# ─── Endpoint ────────────────────────────────────────────────────────────


@router.post(
    "/api/internal/debug/documents/ingest",
    response_model=IngestDebugResponse,
    summary="DEBUG: load + chunk + embed + store a local file (internal)",
)
async def ingest_debug(
    request: IngestDebugRequest,
    embedder: Annotated[EmbeddingProvider, Depends(get_embedder)],
    vector_store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
) -> IngestDebugResponse:
    """Run the full pipeline on `file_path` and return everything it touched.

    Pipeline: load → chunk → embed → store in Chroma → echo back the
    chunks, their embedding vectors, and a store confirmation.

    Status codes:

    * **200** — file ingested + embedded + stored; response carries every
      chunk, its embedding, and stats.
    * **400** — file missing, unsupported extension, or malformed content.
      The `detail` field carries the human-readable reason.
    * **500** — embedding/storage failure (e.g. embedding provider not
      configured) or any other unexpected error. The server-side log has
      the traceback; the response carries a brief message and `failed_at`.

    This endpoint is intentionally synchronous end-to-end (no async
    polling). Phase 1.1 files are small; the production endpoint will queue
    large files.
    """
    start = perf_counter()
    file_path = request.file_path

    # ── Load ──────────────────────────────────────────────────────────
    try:
        loader = get_loader(
            file_path,
            pdf_type=request.pdf_type,
            ocr_language=request.ocr_language,
        )
    except ValueError as e:
        # Bad extension or invalid `pdf_type` — surface as 400.
        raise HTTPException(status_code=400, detail=str(e)) from e

    load_start = perf_counter()
    try:
        text_blocks = await loader.load(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}") from e
    except ImportError as e:
        # OCR deps (pytesseract / pdf2image) not installed — surface as 400
        # with the loader's actionable install instructions, not a 500.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (UnicodeDecodeError, ValueError) as e:
        # Malformed file (corrupted PDF, non-UTF-8 markdown, etc.).
        raise HTTPException(status_code=400, detail=f"Failed to read {file_path}: {e}") from e
    except Exception as e:  # noqa: BLE001 — defensive final fence
        logger.exception("Unexpected loader error for %s", file_path)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}") from e
    load_time_ms = (perf_counter() - load_start) * 1000

    full_text = "\n\n".join(text_blocks)

    # ── Chunk ─────────────────────────────────────────────────────────
    chunk_start = perf_counter()
    try:
        doc_chunks = await chunk_document(text=full_text, source=file_path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected chunker error for %s", file_path)
        raise HTTPException(status_code=500, detail=f"Failed to chunk document: {e}") from e
    chunk_time_ms = (perf_counter() - chunk_start) * 1000

    # ── Embed + store ─────────────────────────────────────────────────
    embed_start = perf_counter()
    try:
        chunks_stored, embeddings = await ingest_chunks(
            doc_chunks, vector_store, embedder=embedder
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Embed/store failed for %s", file_path)
        # Distinguish a metadata-schema rejection (ChunkSchema raised) from a
        # genuine embedding/storage failure so the caller knows where to look.
        error_msg = str(e).lower()
        failed_at = (
            "metadata_validation_step"
            if "metadata" in error_msg or "scalar" in error_msg
            else "embed_store_step"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Embed/store failed",
                "detail": str(e),
                "failed_at": failed_at,
            },
        ) from e
    embed_store_time_ms = (perf_counter() - embed_start) * 1000

    # ── Shape the response ────────────────────────────────────────────
    chunk_details = [
        ChunkDetail(
            id=chunk.chunk_id,
            content=chunk.text,
            chunk_index=chunk.chunk_index,
            token_count=_estimate_tokens(chunk.text),
            # `mode="json"` so datetimes are ISO strings, not raw datetimes.
            metadata=chunk.metadata.model_dump(mode="json"),
        )
        for chunk in doc_chunks
    ]
    total_tokens = sum(c.token_count for c in chunk_details)
    total_time_ms = (perf_counter() - start) * 1000
    document_id = doc_chunks[0].doc_id if doc_chunks else f"empty_{int(start)}"
    loader_used = loader.__class__.__name__
    embedding_model = getattr(embedder, "model_name", type(embedder).__name__)
    store_stats = await vector_store.get_stats()

    logger.info(
        "Ingested %s with %s: %d chunks, %d tokens, %d stored, %.1fms",
        file_path,
        loader_used,
        len(chunk_details),
        total_tokens,
        chunks_stored,
        total_time_ms,
    )

    return IngestDebugResponse(
        document_id=document_id,
        file_path=file_path,
        loader_used=loader_used,
        chunks=chunk_details,
        embeddings=embeddings,
        vector_store=VectorStoreInfo(
            collection=_DEBUG_COLLECTION,
            chunks_stored=chunks_stored,
            embedding_dimension=embedder.embedding_dimension,
            embedding_model=embedding_model,
            persist_dir=str(store_stats["persist_dir"]),
        ),
        stats={
            "total_chunks": len(chunk_details),
            "total_tokens": total_tokens,
            "avg_chunk_tokens": (
                total_tokens / len(chunk_details) if chunk_details else 0.0
            ),
            "load_time_ms": load_time_ms,
            "chunk_time_ms": chunk_time_ms,
            "embed_store_time_ms": embed_store_time_ms,
            "total_time_ms": total_time_ms,
        },
    )
