"""Phase 1 document ingestion: format loaders + token-budget chunker."""

from src.ingestion.chunker import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    chunk_document,
    extract_metadata_for_chunk,
)
from src.ingestion.loaders import (
    DocxLoader,
    DocumentLoader,
    MarkdownLoader,
    PDFLoader,
    ScannedPDFLoader,
    TextLoader,
    get_loader,
)

__all__ = [
    "CHARS_PER_TOKEN",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DocumentLoader",
    "DocxLoader",
    "MAX_CHUNK_SIZE",
    "MarkdownLoader",
    "PDFLoader",
    "ScannedPDFLoader",
    "TextLoader",
    "chunk_document",
    "extract_metadata_for_chunk",
    "get_loader",
]
