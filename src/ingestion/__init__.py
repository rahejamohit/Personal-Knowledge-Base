"""Phase 1 document ingestion: format loaders + token-budget chunker."""

from src.ingestion.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    extract_metadata_for_chunk,
    recursive_split,
)
from src.ingestion.loaders import (
    DocxLoader,
    DocumentLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    get_loader,
)

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DocumentLoader",
    "DocxLoader",
    "MAX_CHUNK_SIZE",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "extract_metadata_for_chunk",
    "get_loader",
    "recursive_split",
]
