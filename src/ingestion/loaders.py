"""Document loaders, one per file format.

Each loader reads ONE format and returns the document split into
*coarse text blocks* (pages for PDFs, H2-rooted sections for Markdown,
paragraphs for plain text). Downstream, `src.ingestion.chunker` further
splits those blocks into embedding-sized chunks.

Phase 1.1 fills in each `load()` body — the abstract class + format
registry below is the long-term interface, intended to stay stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


class DocumentLoader(ABC):
    """Base class for format-specific document loaders.

    Implementations must be async-callable so the ingestion pipeline
    can parallelize across files (Phase 1.1 fans out via
    `asyncio.gather`). Synchronous parsers (e.g. `pypdf`) should be
    wrapped with `asyncio.to_thread` inside `load()` to avoid blocking
    the event loop.
    """

    @abstractmethod
    async def load(self, path: str) -> list[str]:
        """Read `path` and return its content split into coarse blocks.

        Args:
            path: Filesystem path or URL pointing at one document.

        Returns:
            A list of text blocks. The exact granularity is
            format-dependent: pages for PDFs, sections for Markdown,
            paragraphs for plain text. Phase 1.1 documents each
            decision per loader.

        Raises:
            FileNotFoundError: `path` does not exist or is unreadable.
            ValueError: file extension doesn't match the loader.
        """


# ─── Format-specific stubs (Phase 1.1 fills in) ──────────────────────────


class PDFLoader(DocumentLoader):
    """Load `.pdf` files.

    Phase 1.1 implementation plan:
    * Use `pypdf.PdfReader` for text-layer extraction.
    * Fall back to `pdfplumber` for layout-heavy pages.
    * Emit one block per page so `extract_metadata_for_chunk` can
      attach page numbers.
    """

    async def load(self, path: str) -> list[str]:
        raise NotImplementedError("Phase 1.1 implements PDF loading.")


class MarkdownLoader(DocumentLoader):
    """Load `.md` / `.markdown` files.

    Phase 1.1 implementation plan:
    * Split on `##` headers; nested `###` sections stay with their
      parent so the LLM gets enough context per chunk.
    * Preserve the header text in `DocumentMetadata.section` so
      citations can reference it.
    """

    async def load(self, path: str) -> list[str]:
        raise NotImplementedError("Phase 1.1 implements Markdown loading.")


class TextLoader(DocumentLoader):
    """Load plain `.txt` files.

    Phase 1.1 implementation plan:
    * Split on blank-line paragraphs.
    * No structural metadata available beyond `source` + line numbers.
    """

    async def load(self, path: str) -> list[str]:
        raise NotImplementedError("Phase 1.1 implements text loading.")


class DocxLoader(DocumentLoader):
    """Load `.docx` files (Phase 1.1 will wire `python-docx`).

    Implementation plan:
    * Extract paragraphs with `python-docx.Document(...).paragraphs`.
    * Treat `Heading 2` paragraphs as section boundaries.
    """

    async def load(self, path: str) -> list[str]:
        raise NotImplementedError("Phase 1.1 implements DOCX loading.")


# ─── Factory ─────────────────────────────────────────────────────────────


_EXTENSION_LOADERS: dict[str, type[DocumentLoader]] = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".docx": DocxLoader,
}


def get_loader(file_path: str) -> DocumentLoader:
    """Return the right `DocumentLoader` instance for `file_path`.

    Dispatches on the file extension. Centralizing the dispatch here
    means Phase 1.1's `IngestionPipeline.ingest_folder` doesn't need
    its own if/elif chain.

    Args:
        file_path: Path or URL with a recognized extension.

    Returns:
        A fresh `DocumentLoader` subclass instance.

    Raises:
        ValueError: extension not registered in `_EXTENSION_LOADERS`.
    """
    ext = Path(file_path).suffix.lower()
    loader_cls = _EXTENSION_LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file format {ext!r} for {file_path!r}. "
            f"Supported: {sorted(_EXTENSION_LOADERS)}"
        )
    return loader_cls()
