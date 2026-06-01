"""SQLite-backed session and turn persistence.

Phase 1 storage. Two flat tables (`sessions`, `conversation_turns`) with
JSON-typed columns for nested Pydantic data. Phase 2 will swap this for
PostgreSQL + SQLAlchemy — this module is the migration boundary; the
`SessionManager` public surface stays stable across both.

Design notes
------------
* **Sync `sqlite3` inside `async def` methods.** Phase 1 trade-off: keeps
  the call sites uniform with Phase 2's true-async DB API while avoiding
  `aiosqlite` as a dependency. Each `await` resolves immediately because
  the work is synchronous, so this DOES block the event loop on every
  call — fine for a single-user CLI / API, not fine at >50 RPS. Phase 2
  fixes it.
* **One shared connection, `check_same_thread=False`.** SQLite serializes
  writes internally; we add a module-level `Lock` to keep our own
  back-to-back writes ordered. The connection lives for the process.
* **WAL journaling.** `journal_mode=WAL` lets readers proceed during a
  write, which matters once the session list grows.
* **No FK enforcement.** Per spec: "No foreign key constraints initially
  (Phase 1 simplicity)". `delete_session` manually cascades to turns.
* **JSON via Pydantic.** `model_dump(mode="json")` is the single source
  of truth for serialization — matches what the FastAPI layer emits, so
  a turn round-trips through SQLite without schema drift.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from src.models.conversation import ConversationTurn
from src.utils import get_logger
from src.utils.ids import new_session_id

logger = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user      ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated   ON sessions(updated_at);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    user_message    TEXT NOT NULL,
    agent_response  TEXT NOT NULL,
    retrieved_docs  TEXT NOT NULL DEFAULT '[]',
    tool_calls      TEXT NOT NULL DEFAULT '[]',
    tool_results    TEXT NOT NULL DEFAULT '[]',
    token_usage     TEXT NOT NULL DEFAULT '{}',
    metadata        TEXT NOT NULL DEFAULT '{}',
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session      ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_turn ON conversation_turns(session_id, turn_number);
"""


def _now_iso() -> str:
    """UTC `datetime.now()` as ISO 8601 — the only timestamp format we store."""
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    """Sessions + turns over SQLite.

    Args:
        db_path: Filesystem path to the SQLite database. The parent
            directory is created on init so the first run doesn't fail
            with `No such file or directory`.
    """

    def __init__(self, db_path: str | Path = ".data/sessions.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` so FastAPI's thread-pool can share the
        # manager. SQLite still serializes writes; `_write_lock` keeps our
        # own multi-statement writes ordered.
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit txns where needed
        )
        self._conn.row_factory = sqlite3.Row
        # WAL: lets readers run during a write. Cheap; opt-in once per file.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._write_lock = RLock()
        logger.info("SessionManager opened db=%s", self._db_path)

    # ─── Mutations ─────────────────────────────────────────────

    async def create_session(self, user_id: str, title: str | None = None) -> str:
        """Insert a new session and return its server-generated ID."""
        sid = new_session_id()
        now = _now_iso()
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO sessions "
                "(id, user_id, title, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, user_id, title, now, now, "{}"),
            )
        logger.info("session created id=%s user=%s", sid, user_id)
        return sid

    async def save_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Persist a `ConversationTurn` and bump the parent session's
        `updated_at` to the turn's timestamp.

        Raises:
            sqlite3.IntegrityError: if `turn.id` collides with an existing row.
        """
        payload = turn.model_dump(mode="json")
        with self._write_lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "INSERT INTO conversation_turns "
                    "(id, session_id, turn_number, user_message, agent_response, "
                    " retrieved_docs, tool_calls, tool_results, token_usage, "
                    " metadata, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        turn.id,
                        session_id,
                        turn.turn_number,
                        turn.user_message,
                        turn.agent_response,
                        json.dumps(payload["retrieved_docs"]),
                        json.dumps(payload["tool_calls"]),
                        json.dumps(payload["tool_results"]),
                        json.dumps(payload["token_usage"]),
                        json.dumps(payload["metadata"]),
                        payload["timestamp"],
                    ),
                )
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (payload["timestamp"], session_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    async def delete_session(self, session_id: str) -> None:
        """Hard-delete a session and all its turns. Silent no-op if missing."""
        with self._write_lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "DELETE FROM conversation_turns WHERE session_id = ?", (session_id,)
                )
                self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        logger.info("session deleted id=%s", session_id)

    # ─── Reads ─────────────────────────────────────────────────

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Return session metadata + turn count, or `None` if not found."""
        row = self._conn.execute(
            "SELECT id, user_id, title, created_at, updated_at, metadata "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        count_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "turn_count": int(count_row["n"]),
        }

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Most-recently-updated sessions for `user_id`. Capped at `limit`."""
        rows = self._conn.execute(
            "SELECT id, user_id, title, created_at, updated_at, metadata "
            "FROM sessions WHERE user_id = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        ]

    async def get_or_create_session(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> str:
        """Return an existing session's ID or create a new one.

        If `session_id` is provided AND it exists AND belongs to
        `user_id`, return it as-is. Otherwise create a fresh session and
        return its new ID — we never invent rows for unknown session IDs
        (that would let any caller squat on any ID).

        Note: The check-and-create is wrapped in a lock to prevent race
        conditions where two concurrent calls could both observe missing
        session and create duplicates.
        """
        with self._write_lock:
            if session_id:
                existing = await self.load_session(session_id)
                if existing is not None and existing["user_id"] == user_id:
                    return session_id
            return await self.create_session(user_id)

    # ─── Lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying connection. Test-only — in real use the
        connection lives for the process lifetime."""
        with self._write_lock:
            self._conn.close()
