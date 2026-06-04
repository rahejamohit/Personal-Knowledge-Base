"""Document loaders, one per file format.

Each loader reads ONE format and returns the document split into
*coarse text blocks* (pages for PDFs, H2-rooted sections for Markdown,
the whole file as one block for plain text, paragraphs for DOCX).
Downstream, `src.ingestion.chunker` further splits those blocks into
embedding-sized chunks.

Implementation choices
----------------------
* **Sync libraries inside `async def`.** `pypdf` and `python-docx` are
  both synchronous. We wrap calls in `asyncio.to_thread` so a long
  file doesn't block the event loop — matters for the FastAPI debug
  endpoint, where multiple ingestion requests may arrive in parallel.
* **Error contract.** Missing files → `FileNotFoundError`; malformed
  files → `ValueError`. The API layer maps both to HTTP 400.
* **Encoding.** Markdown and text are read as strict UTF-8 — anything
  else raises `UnicodeDecodeError`, which the API also maps to 400.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


# When a PDF page has more than this many characters, we split it on
# blank-line paragraph boundaries to keep block sizes manageable. Most
# pages stay well under this; the limit only kicks in for diagrams or
# tables whose text extracts as one giant line.
_PDF_PAGE_SOFT_LIMIT = 10_000


class DocumentLoader(ABC):
    """Base class for format-specific document loaders.

    Subclasses must be async-callable so the ingestion pipeline can
    parallelize across files. Synchronous parsers (e.g. `pypdf`) should
    be wrapped with `asyncio.to_thread` inside `load()`.
    """

    @abstractmethod
    async def load(self, path: str) -> list[str]:
        """Read `path` and return its content split into coarse blocks.

        Args:
            path: Filesystem path pointing at one document.

        Returns:
            A list of text blocks (granularity is format-dependent).

        Raises:
            FileNotFoundError: `path` does not exist or is unreadable.
            ValueError: file format mismatch / file is malformed.
            UnicodeDecodeError: text-based formats with bad encoding.
        """


# Per-page text length, in characters, below which hybrid mode falls
# back to OCR. Scanned pages typically extract as "" or a few whitespace
# artifacts; 50 chars is the safe threshold from the spec. Hoisted to a
# module constant so it's tunable without touching the body.
_OCR_FALLBACK_THRESHOLD: int = 50


# ─── PDF ─────────────────────────────────────────────────────────────────


class PDFLoader(DocumentLoader):
    """Load `.pdf` files via `pypdf`, with optional per-page OCR fallback.

    Two modes:

    * **Text-only** (``enable_ocr_fallback=False``, the constructor
      default). Fast — `pypdf.extract_text()` per page, no OCR, no
      imports of pytesseract / pdf2image. Best when you know the PDF
      has a real text layer.

    * **Hybrid** (``enable_ocr_fallback=True``). Tries `extract_text`
      first; if the result strips to fewer than ``_OCR_FALLBACK_THRESHOLD``
      characters, OCRs just that page via `pdf2image` + `pytesseract`.
      Handles mixed documents (some pages text, some scanned) optimally
      without paying OCR cost on the text pages. This is what
      ``get_loader(..., pdf_type="auto")`` returns.

    One block per page. Pages whose text exceeds ``_PDF_PAGE_SOFT_LIMIT``
    are further split on blank-line paragraph boundaries to keep the
    chunker's input block sizes predictable. Page order is preserved.
    """

    def __init__(
        self,
        enable_ocr_fallback: bool = False,
        ocr_language: str = "eng",
    ) -> None:
        """Args:
            enable_ocr_fallback: If True, OCR pages whose extracted text
                falls below ``_OCR_FALLBACK_THRESHOLD``. If False, those
                pages are emitted as empty blocks and no OCR imports
                happen.
            ocr_language: Tesseract language code (e.g. ``"eng"``,
                ``"deu"``). Only consulted when fallback fires.
        """
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_language = ocr_language

    async def load(self, path: str) -> list[str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            blocks, text_pages, ocr_pages = await asyncio.to_thread(self._read_sync, p)
        except FileNotFoundError:
            raise
        except ImportError:
            # OCR fallback was needed but its deps aren't installed.
            # Propagate so the API layer can surface a 400 with a
            # helpful message rather than collapsing to a generic 500.
            raise
        except Exception as e:  # pypdf can raise many vendor-specific errors
            raise ValueError(f"Failed to read PDF {path!r}: {e}") from e

        logger.info(
            "PDFLoader: %s → %d blocks (text_pages=%d, ocr_pages=%d, hybrid=%s)",
            path,
            len(blocks),
            text_pages,
            ocr_pages,
            self.enable_ocr_fallback,
        )
        return blocks

    def _read_sync(self, p: Path) -> tuple[list[str], int, int]:
        """Returns (blocks, text_pages_count, ocr_pages_count)."""
        import pypdf  # imported inside the threadpool callback

        reader = pypdf.PdfReader(str(p))
        blocks: list[str] = []
        text_pages = 0
        ocr_pages = 0
        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if (
                self.enable_ocr_fallback
                and len(text) < _OCR_FALLBACK_THRESHOLD
            ):
                # Per-page OCR. This page only — not the whole document.
                text = self._ocr_single_page(p, page_idx).strip()
                ocr_pages += 1
                logger.debug(
                    "PDFLoader OCR fallback: page %d → %d chars",
                    page_idx + 1,
                    len(text),
                )
            else:
                text_pages += 1

            if not text:
                blocks.append("")
                continue
            if len(text) > _PDF_PAGE_SOFT_LIMIT:
                blocks.extend(part for part in text.split("\n\n") if part.strip())
            else:
                blocks.append(text)
        return blocks, text_pages, ocr_pages

    def _ocr_single_page(self, p: Path, page_idx: int) -> str:
        """OCR exactly one page of `p`. Lazy-imports the OCR stack."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "OCR fallback requires pytesseract and pdf2image. "
                "Install Python packages with `uv sync` and the system "
                "binaries: `brew install tesseract poppler` on macOS, "
                f"`apt-get install tesseract-ocr poppler-utils` on Linux. ({e})"
            ) from e

        # pdf2image is 1-indexed.
        images = convert_from_path(
            str(p), first_page=page_idx + 1, last_page=page_idx + 1
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang=self.ocr_language)


# ─── Scanned PDF (OCR-only) ──────────────────────────────────────────────


class ScannedPDFLoader(DocumentLoader):
    """Load image-based `.pdf` files entirely via OCR.

    Convert the whole document to images in one ``convert_from_path``
    call (a single fork of `pdftoppm` is much faster than N invocations),
    then OCR each page. Best when you know every page is scanned —
    `PDFLoader(enable_ocr_fallback=True)` is preferable for mixed
    documents because it skips OCR on the pages that don't need it.

    Performance: ~5–10s per page on a CPU laptop. Unavoidable; OCR cost
    is dominated by `tesseract`.
    """

    def __init__(self, ocr_language: str = "eng") -> None:
        self.ocr_language = ocr_language

    async def load(self, path: str) -> list[str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        try:
            blocks = await asyncio.to_thread(self._read_sync, p)
        except FileNotFoundError:
            raise
        except ImportError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to OCR PDF {path!r}: {e}") from e
        logger.info(
            "ScannedPDFLoader: %s → %d blocks (lang=%s)",
            path,
            len(blocks),
            self.ocr_language,
        )
        return blocks

    def _read_sync(self, p: Path) -> list[str]:
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "ScannedPDFLoader requires pytesseract and pdf2image. "
                "Install Python packages with `uv sync` and the system "
                "binaries: `brew install tesseract poppler` on macOS, "
                f"`apt-get install tesseract-ocr poppler-utils` on Linux. ({e})"
            ) from e

        images = convert_from_path(str(p))  # batch — all pages at once
        blocks: list[str] = []
        for page_idx, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang=self.ocr_language).strip()
            logger.debug(
                "ScannedPDFLoader page %d → %d chars", page_idx + 1, len(text)
            )
            if not text:
                blocks.append("")
                continue
            if len(text) > _PDF_PAGE_SOFT_LIMIT:
                blocks.extend(part for part in text.split("\n\n") if part.strip())
            else:
                blocks.append(text)
        return blocks


# ─── Markdown ────────────────────────────────────────────────────────────


# A `##`-prefixed line at the start of a line, allowing leading whitespace
# inside fenced code blocks to be ignored (we don't try to handle that —
# Phase 1 takes the simple `re.split` and accepts that a `##` inside a code
# block would be (incorrectly) treated as a section boundary. The chunker's
# code-block extraction makes this rare in practice).
_H2_BOUNDARY_RE = re.compile(r"(?m)^(?=##\s)")


class MarkdownLoader(DocumentLoader):
    """Load `.md` / `.markdown` files. Splits on H2 headers.

    The header line is *kept* at the start of its section so the
    chunker can use the heading as context. A file with no H2 headers
    returns one block (everything).
    """

    async def load(self, path: str) -> list[str]:
        text = await _read_text(path)
        if "##" not in text:
            return [text.strip()] if text.strip() else []
        # `re.split` with a lookahead keeps the H2 header inside its section.
        parts = _H2_BOUNDARY_RE.split(text)
        blocks = [p.strip() for p in parts if p.strip()]
        logger.info("MarkdownLoader: %s → %d sections", path, len(blocks))
        return blocks


# ─── Text ────────────────────────────────────────────────────────────────


class TextLoader(DocumentLoader):
    """Load plain `.txt` files. One block: the whole file.

    The chunker is responsible for splitting. We deliberately don't try
    to be clever about paragraph boundaries here — that's a chunker
    concern and there's no plain-text equivalent of an H2 header that
    we could use as a coarse boundary.
    """

    async def load(self, path: str) -> list[str]:
        text = await _read_text(path)
        # Preserve line breaks; only strip leading/trailing whitespace.
        return [text.strip()] if text.strip() else []


# ─── DOCX ────────────────────────────────────────────────────────────────


class DocxLoader(DocumentLoader):
    """Load `.docx` files via `python-docx`.

    One block per non-empty paragraph. Headers count as paragraphs.
    Tables, footnotes, and footers are intentionally skipped — Phase
    1.1 is prose-only. Order is preserved.
    """

    async def load(self, path: str) -> list[str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"DOCX not found: {path}")
        try:
            blocks = await asyncio.to_thread(self._read_sync, p)
        except Exception as e:  # python-docx raises BadZipFile / KeyError / ...
            raise ValueError(f"Failed to read DOCX {path!r}: {e}") from e
        logger.info("DocxLoader: %s → %d paragraphs", path, len(blocks))
        return blocks

    @staticmethod
    def _read_sync(p: Path) -> list[str]:
        from docx import Document  # imported inside the threadpool callback

        doc = Document(str(p))
        return [para.text for para in doc.paragraphs if para.text.strip()]


# ─── Shared helpers ──────────────────────────────────────────────────────


async def _read_text(path: str) -> str:
    """UTF-8 read for the text-based loaders.

    `Path.read_text` is fast enough for ingestion volumes; we still
    wrap in `to_thread` so a 100MB note file doesn't pause the event
    loop. Raises `FileNotFoundError` / `UnicodeDecodeError` cleanly.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return await asyncio.to_thread(p.read_text, encoding="utf-8")


# ─── Factory ─────────────────────────────────────────────────────────────


# Non-PDF formats dispatch by extension as before; PDFs go through the
# `pdf_type`-aware branch in `get_loader`.
_EXTENSION_LOADERS: dict[str, type[DocumentLoader]] = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".docx": DocxLoader,
}

_VALID_PDF_TYPES: frozenset[str] = frozenset({"text", "scanned", "auto"})


def get_loader(
    file_path: str,
    pdf_type: str = "auto",
    ocr_language: str = "eng",
) -> DocumentLoader:
    """Return the right `DocumentLoader` for `file_path`.

    Args:
        file_path: Path with a recognized extension. The file does not
            need to exist yet — only its extension is consulted.
        pdf_type: For PDFs only. One of:

            * ``"text"`` — `PDFLoader(enable_ocr_fallback=False)`. Pure
              text extraction, no OCR. Fast.
            * ``"scanned"`` — `ScannedPDFLoader()`. Batch OCR of every
              page. Use when every page is scanned.
            * ``"auto"`` (default) — `PDFLoader(enable_ocr_fallback=True)`.
              Hybrid: text extraction per page with per-page OCR
              fallback for pages whose text layer is missing. Optimal
              for mixed documents.

            Ignored for non-PDF extensions.
        ocr_language: Tesseract language code. Passed through to
            whichever loader actually does OCR.

    Returns:
        A fresh `DocumentLoader` subclass instance.

    Raises:
        ValueError: extension not in `_EXTENSION_LOADERS`, or
            `pdf_type` not in ``{"text", "scanned", "auto"}``.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        if pdf_type not in _VALID_PDF_TYPES:
            raise ValueError(
                f"Invalid pdf_type {pdf_type!r}. "
                f"Must be one of {sorted(_VALID_PDF_TYPES)}."
            )
        if pdf_type == "scanned":
            logger.info("get_loader: pdf_type=scanned → ScannedPDFLoader")
            return ScannedPDFLoader(ocr_language=ocr_language)
        # "text" → fallback off; "auto" → fallback on. Same class, two modes.
        enable_fallback = pdf_type == "auto"
        logger.info(
            "get_loader: pdf_type=%s → PDFLoader(enable_ocr_fallback=%s)",
            pdf_type,
            enable_fallback,
        )
        return PDFLoader(
            enable_ocr_fallback=enable_fallback,
            ocr_language=ocr_language,
        )

    loader_cls = _EXTENSION_LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file format {ext!r} for {file_path!r}. "
            f"Supported: {sorted(_EXTENSION_LOADERS)}"
        )
    return loader_cls()
