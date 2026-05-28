"""Identifier generation helpers.

We use ULIDs-via-uuid4 (time-prefixed string IDs) for sessions and turns:
sortable-ish, URL-safe, and easy to inspect in logs. We don't pull a real
ULID library to keep the dependency surface small — uuid4 hex is good enough
for Phase 1 and the call sites are centralized here so we can swap later.
"""

from __future__ import annotations

import time
import uuid


def new_session_id() -> str:
    """Generate a new session ID. Format: `sess_<unix_ms>_<8 hex>`."""
    return f"sess_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def new_turn_id() -> str:
    """Generate a new turn ID. Format: `turn_<unix_ms>_<8 hex>`."""
    return f"turn_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def new_doc_id() -> str:
    """Generate a new document ID. Format: `doc_<8 hex>`."""
    return f"doc_{uuid.uuid4().hex[:8]}"


def new_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID: same doc + index always produces the same ID.

    This matters for idempotent ingestion — re-ingesting a document should
    update existing chunks in Chroma rather than creating duplicates.
    """
    return f"{doc_id}::chunk_{chunk_index:04d}"
