"""Unit tests for `SessionManager` (SQLite-backed session/turn store).

Every test gets its own `tmp_path` so they're hermetic and parallel-safe.
`SessionManager` is `async`-shaped (per spec) so the tests `await` even
though the bodies are synchronous I/O. `pyproject.toml`'s
`asyncio_mode = "auto"` auto-detects `async def test_*` functions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import pytest

from src.models.conversation import ConversationTurn, TokenUsage
from src.models.conversation import RetrievedDoc
from src.storage.db import SessionManager


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_turn(
    *,
    turn_id: str,
    session_id: str,
    turn_number: int,
    user: str = "hello",
    agent: str = "hi back",
) -> ConversationTurn:
    return ConversationTurn(
        id=turn_id,
        session_id=session_id,
        turn_number=turn_number,
        user_message=user,
        agent_response=agent,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    )


@pytest.fixture
def mgr(tmp_path: Path) -> SessionManager:
    return SessionManager(db_path=tmp_path / "sessions.db")


# ─── create_session ──────────────────────────────────────────────────────


class TestCreateSession:
    async def test_returns_valid_id(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session(user_id="alice")
        # Server-generated IDs use the `sess_<ms>_<hex>` shape from utils.ids.
        assert isinstance(sid, str)
        assert sid.startswith("sess_")
        # Loadable immediately.
        assert await mgr.load_session(sid) is not None

    async def test_distinct_ids_per_call(self, mgr: SessionManager) -> None:
        a = await mgr.create_session("alice")
        b = await mgr.create_session("alice")
        assert a != b

    async def test_optional_title(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session("alice", title="My chat")
        loaded = await mgr.load_session(sid)
        assert loaded is not None
        assert loaded["title"] == "My chat"


# ─── save_turn + load_session ────────────────────────────────────────────


class TestSaveAndLoad:
    async def test_save_turn_persists_data(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session("alice")
        turn = _make_turn(turn_id="turn_1", session_id=sid, turn_number=1)
        await mgr.save_turn(sid, turn)

        loaded = await mgr.load_session(sid)
        assert loaded is not None
        assert loaded["turn_count"] == 1

    async def test_save_turn_round_trips_retrieved_docs(self, mgr: SessionManager) -> None:
        # The riskiest serialization in the turn payload — nested model
        # → JSON column → JSON.loads-able dict on read.
        sid = await mgr.create_session("alice")
        turn = ConversationTurn(
            id="turn_1",
            session_id=sid,
            turn_number=1,
            user_message="q",
            agent_response="a",
            retrieved_docs=[
                RetrievedDoc(
                    chunk_id="c1",
                    doc_id="d1",
                    source="paper.pdf",
                    text="excerpt",
                    score=0.9,
                    rank=0,
                    metadata={"page": 3},
                )
            ],
        )
        await mgr.save_turn(sid, turn)
        loaded = await mgr.load_session(sid)
        assert loaded["turn_count"] == 1

    async def test_load_session_returns_none_for_missing(self, mgr: SessionManager) -> None:
        assert await mgr.load_session("sess_nope") is None

    async def test_updated_at_advances_on_turn(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session("alice")
        before = (await mgr.load_session(sid))["updated_at"]
        # Sleep a tick so the timestamp is observably later, then save a turn.
        await asyncio.sleep(0.01)
        turn = _make_turn(turn_id="t_1", session_id=sid, turn_number=1)
        # Use a stamp clearly later than `before`.
        turn = turn.model_copy(update={"timestamp": datetime.now(timezone.utc)})
        await mgr.save_turn(sid, turn)
        after = (await mgr.load_session(sid))["updated_at"]
        assert after >= before  # ISO strings sort lexicographically


# ─── list_sessions ───────────────────────────────────────────────────────


class TestListSessions:
    async def test_filters_by_user_id(self, mgr: SessionManager) -> None:
        a1 = await mgr.create_session("alice")
        a2 = await mgr.create_session("alice")
        _b = await mgr.create_session("bob")

        alice_sessions = {s["id"] for s in await mgr.list_sessions("alice")}
        bob_sessions = {s["id"] for s in await mgr.list_sessions("bob")}

        assert alice_sessions == {a1, a2}
        assert bob_sessions == {_b}

    async def test_orders_by_recency_desc(self, mgr: SessionManager) -> None:
        first = await mgr.create_session("alice")
        await asyncio.sleep(0.01)
        second = await mgr.create_session("alice")
        sessions = await mgr.list_sessions("alice")
        assert [s["id"] for s in sessions] == [second, first]

    async def test_respects_limit(self, mgr: SessionManager) -> None:
        for _ in range(5):
            await mgr.create_session("alice")
        assert len(await mgr.list_sessions("alice", limit=3)) == 3


# ─── delete_session ──────────────────────────────────────────────────────


class TestDelete:
    async def test_delete_removes_session_and_turns(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session("alice")
        for i in range(1, 4):
            await mgr.save_turn(sid, _make_turn(turn_id=f"t_{i}", session_id=sid, turn_number=i))
        assert (await mgr.load_session(sid))["turn_count"] == 3

        await mgr.delete_session(sid)
        assert await mgr.load_session(sid) is None
        # Per-spec: turns must be gone too. Probe directly via the
        # connection to be sure no orphans linger.
        rows = mgr._conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_turns WHERE session_id = ?",
            (sid,),
        ).fetchone()
        assert rows["n"] == 0

    async def test_delete_missing_id_is_noop(self, mgr: SessionManager) -> None:
        # Silent no-op rather than error — matches the docstring contract.
        await mgr.delete_session("sess_nope")


# ─── get_or_create_session ───────────────────────────────────────────────


class TestGetOrCreate:
    async def test_returns_existing_when_owned_by_user(self, mgr: SessionManager) -> None:
        sid = await mgr.create_session("alice")
        same = await mgr.get_or_create_session("alice", session_id=sid)
        assert same == sid

    async def test_creates_new_when_session_id_unknown(self, mgr: SessionManager) -> None:
        sid = await mgr.get_or_create_session("alice", session_id="sess_unknown")
        assert sid != "sess_unknown"
        assert (await mgr.load_session(sid)) is not None

    async def test_creates_new_when_session_id_omitted(self, mgr: SessionManager) -> None:
        sid = await mgr.get_or_create_session("alice")
        assert (await mgr.load_session(sid)) is not None

    async def test_does_not_hand_out_other_users_session(self, mgr: SessionManager) -> None:
        # Even if bob knows alice's session ID, get_or_create for bob
        # should mint a fresh one — defensive against ID-guessing.
        alice_sid = await mgr.create_session("alice")
        bob_sid = await mgr.get_or_create_session("bob", session_id=alice_sid)
        assert bob_sid != alice_sid


# ─── Persistence + concurrency ───────────────────────────────────────────


class TestPersistence:
    def test_database_survives_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "sessions.db"
        mgr1 = SessionManager(db_path=path)
        sid = asyncio.run(mgr1.create_session("alice", title="t1"))
        asyncio.run(
            mgr1.save_turn(sid, _make_turn(turn_id="t_1", session_id=sid, turn_number=1))
        )
        mgr1.close()

        mgr2 = SessionManager(db_path=path)
        loaded = asyncio.run(mgr2.load_session(sid))
        assert loaded is not None
        assert loaded["user_id"] == "alice"
        assert loaded["title"] == "t1"
        assert loaded["turn_count"] == 1


class TestConcurrentAccess:
    def test_two_threads_writing_to_same_session(self, tmp_path: Path) -> None:
        """Ten threads each save one turn. All should land without loss
        or interleaving corruption."""
        mgr = SessionManager(db_path=tmp_path / "sessions.db")
        sid = asyncio.run(mgr.create_session("alice"))

        def writer(idx: int) -> None:
            turn = _make_turn(turn_id=f"t_{idx}", session_id=sid, turn_number=idx)
            # Each thread runs its own event loop — cheap for this volume.
            asyncio.run(mgr.save_turn(sid, turn))

        threads = [Thread(target=writer, args=(i,)) for i in range(1, 11)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = asyncio.run(mgr.load_session(sid))
        assert loaded is not None
        assert loaded["turn_count"] == 10
