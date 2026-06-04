"""Integration tests for `POST /api/internal/debug/documents/ingest`.

Exercises the full Phase 1.1 pipeline through FastAPI's `TestClient`:
loader factory → loader → chunker → response shaping. No real network
or LLM calls. Sample documents come from the on-disk fixtures (see
`tests/conftest.py` for the binary autogen).

Why these are `tests/integration/` rather than `tests/unit/`: they
exercise the cross-module wiring (server → router → ingestion module
→ loaders → chunker), not any single unit. They still run fast and
don't hit external services, so we don't put them behind the
`integration` marker — the path placement is the only gate.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


# ─── Test client fixture ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """One `TestClient` per module — cheap to share since the app is
    stateless for ingestion (no per-test session state involved)."""
    with TestClient(app) as c:
        yield c


# ─── Markdown happy path ─────────────────────────────────────────────────


def test_ingest_markdown_returns_200_with_chunks(client: TestClient) -> None:
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/rag_guide.md"},
    )
    assert response.status_code == 200, response.text

    data = response.json()
    assert "document_id" in data
    assert data["file_path"] == "tests/evals/fixtures/sample_docs/rag_guide.md"
    assert data["loader_used"] == "MarkdownLoader"
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0

    # First chunk has the expected shape.
    chunk = data["chunks"][0]
    assert {"id", "content", "chunk_index", "token_count", "metadata"} <= chunk.keys()
    assert chunk["chunk_index"] == 0
    assert chunk["token_count"] > 0
    assert chunk["metadata"]["source"].endswith("rag_guide.md")
    assert chunk["metadata"]["content_type"] == "text/markdown"


def test_ingest_stats_block(client: TestClient) -> None:
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/faq.md"},
    )
    assert response.status_code == 200
    stats = response.json()["stats"]
    assert stats["total_chunks"] >= 1
    assert stats["total_tokens"] > 0
    assert stats["avg_chunk_tokens"] > 0
    # Token counts add up.
    chunks = response.json()["chunks"]
    assert stats["total_tokens"] == sum(c["token_count"] for c in chunks)
    # Latency reported, non-negative.
    assert stats["processing_time_ms"] >= 0


# ─── PDF (uses autogen fixture) ──────────────────────────────────────────


def test_ingest_pdf_returns_pdf_loader(client: TestClient) -> None:
    pytest.importorskip("pypdf")
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/sample_pdf_simple.pdf"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["loader_used"] == "PDFLoader"
    assert data["chunks"], "expected at least one chunk from the 2-page PDF"


# ─── DOCX (uses autogen fixture) ─────────────────────────────────────────


def test_ingest_docx_returns_docx_loader(client: TestClient) -> None:
    pytest.importorskip("docx")
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/sample_document.docx"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["loader_used"] == "DocxLoader"
    assert data["chunks"]


# ─── Error contracts ─────────────────────────────────────────────────────


def test_missing_file_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/does_not_exist.md"},
    )
    assert response.status_code == 400
    assert "File not found" in response.json()["detail"]


def test_unsupported_extension_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": "tests/evals/fixtures/sample_docs/something.xyz"},
    )
    assert response.status_code == 400
    # The loader factory's message starts with "Unsupported file format".
    assert "Unsupported file format" in response.json()["detail"]


def test_missing_request_body_field_returns_422(client: TestClient) -> None:
    # Pydantic catches missing required fields and emits 422 — we don't
    # try to coerce that into 400 (FastAPI's default is fine here).
    response = client.post("/api/internal/debug/documents/ingest", json={})
    assert response.status_code == 422


def test_empty_file_path_returns_422(client: TestClient) -> None:
    # `min_length=1` on `file_path` → Pydantic rejects empty strings.
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": ""},
    )
    assert response.status_code == 422


# ─── pdf_type dispatch ───────────────────────────────────────────────────


def _ocr_runtime_available() -> bool:
    """Mirror of the helper in tests/unit/test_loaders.py."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_pdf_type_text_uses_text_only_loader(client: TestClient) -> None:
    """`pdf_type=text` → `PDFLoader(enable_ocr_fallback=False)` — works
    even without the OCR stack installed."""
    pytest.importorskip("pypdf")
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={
            "file_path": "tests/evals/fixtures/sample_docs/sample_pdf_simple.pdf",
            "pdf_type": "text",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["loader_used"] == "PDFLoader"


def test_pdf_type_auto_returns_pdf_loader_on_text_pdf(client: TestClient) -> None:
    """`pdf_type=auto` → hybrid `PDFLoader`. On a text PDF the fallback
    never fires, so this works without OCR installed."""
    pytest.importorskip("pypdf")
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={
            "file_path": "tests/evals/fixtures/sample_docs/sample_pdf_simple.pdf",
            "pdf_type": "auto",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["loader_used"] == "PDFLoader"
    assert data["chunks"]


def test_pdf_type_scanned_uses_scanned_loader(client: TestClient) -> None:
    if not _ocr_runtime_available():
        pytest.skip("tesseract / pytesseract / pdf2image not installed")
    scanned = "tests/evals/fixtures/sample_docs/sample_pdf_scanned.pdf"
    from pathlib import Path

    if not Path(scanned).exists():
        pytest.skip("scanned PDF fixture not generated (PIL/reportlab missing)")
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={"file_path": scanned, "pdf_type": "scanned"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["loader_used"] == "ScannedPDFLoader"


def test_invalid_pdf_type_returns_422(client: TestClient) -> None:
    """`Literal["text","scanned","auto"]` rejects unknown values at the
    Pydantic layer → 422 Unprocessable Entity, not a 500."""
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={
            "file_path": "tests/evals/fixtures/sample_docs/sample_pdf_simple.pdf",
            "pdf_type": "bogus",
        },
    )
    assert response.status_code == 422


def test_pdf_type_ignored_for_non_pdf(client: TestClient) -> None:
    """A `pdf_type` value on a non-PDF request is silently ignored; the
    matching loader is selected by extension."""
    response = client.post(
        "/api/internal/debug/documents/ingest",
        json={
            "file_path": "tests/evals/fixtures/sample_docs/rag_guide.md",
            "pdf_type": "scanned",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["loader_used"] == "MarkdownLoader"
