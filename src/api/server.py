"""FastAPI server exposing the Personal Knowledge Agent over HTTP.

This module is a thin wrapper. It does NOT change any agent behavior — it
just translates HTTP requests into calls on the existing
`ConversationManager`/`KnowledgeAgent` stack.

Architectural decisions
-----------------------
* **One process, in-memory sessions (Phase 0).** Sessions live in a module-
  level dict keyed by `session_id`. They evaporate on restart. Phase 1
  swaps `_get_or_create_manager` to load/persist via SQLite without
  touching the endpoints.
* **Lazy agent construction.** The Gemini-backed `KnowledgeAgent` is built
  on first use (not at import time) so the server can boot and serve
  `/docs` even when `GOOGLE_API_KEY` is missing — failure surfaces only
  when someone actually posts a turn.
* **Errors stay inside the turn record.** `ConversationManager.process_turn`
  already catches crew failures and returns a graceful "internal error"
  turn with `metadata.errored=True`. We propagate that to the client as a
  200 — the contract is "the conversation continues" not "HTTP fails." We
  only return 500 for unexpected exceptions *outside* the manager (e.g. a
  bad LLM key surfacing during agent construction).
* **Session ID comes from the caller, not generated server-side.** The
  Phase 0 CLI generates a `sess_<ms>_<hex>` ID; the API instead trusts the
  client's choice (e.g. `user-123`). This means `Session(id=...)` is
  constructed manually rather than relying on the default factory.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware

from src.agent.conversation_manager import ConversationManager
from src.agent.orchestrator import KnowledgeAgent
from src.agent.tools import build_default_tools
from src.api.models import (
    SessionMetadata,
    TurnResponse,
    TurnsListResponse,
)
from src.config import get_settings
from src.models.conversation import ConversationTurn, Session
from src.utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ─── App + middleware ─────────────────────────────────────────────────────

app = FastAPI(
    title="Personal Knowledge Agent API",
    description=(
        "REST surface over the CrewAI + Gemini knowledge agent. Phase 0: "
        "in-memory sessions, stub retrieval. See /docs for interactive API."
    ),
    version="0.1.0",
)

# Frontend (e.g. Create-React-App dev server) talks to us cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session registry (Phase 0) ─────────────────────────────────

# Maps session_id -> ConversationManager. Lost on process restart.
# Phase 1: replace with a SQLite-backed loader behind the same `_get_*` API.
_SESSIONS: Final[dict[str, ConversationManager]] = {}


@lru_cache(maxsize=1)
def _get_agent() -> KnowledgeAgent:
    """Build the shared `KnowledgeAgent` exactly once (process-wide).

    Doing this lazily (rather than at module import) keeps the server
    bootable without `GOOGLE_API_KEY` set, which is helpful when only
    `/docs` is being explored.
    """
    settings = get_settings()
    if not settings.has_gemini:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Configure your .env before posting turns."
        )
    settings.ensure_storage_dirs()
    return KnowledgeAgent(build_default_tools(), settings=settings, verbose=False)


def _get_or_create_manager(session_id: str) -> ConversationManager:
    """Return the `ConversationManager` for this session, creating if needed.

    The session ID is supplied by the *client* (path param), which differs
    from the CLI's auto-generated IDs. We override the dataclass default by
    passing `session=Session(id=session_id)`.
    """
    mgr = _SESSIONS.get(session_id)
    if mgr is None:
        settings = get_settings()
        mgr = ConversationManager(
            agent=_get_agent(),
            settings=settings,
            session=Session(id=session_id),
        )
        _SESSIONS[session_id] = mgr
        logger.info("Created new session: %s", session_id)
    return mgr


def _get_manager_or_404(session_id: str) -> ConversationManager:
    """Look up an existing session, or raise 404. Does NOT create."""
    mgr = _SESSIONS.get(session_id)
    if mgr is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return mgr


# ─── Adapters: internal → wire ────────────────────────────────────────────

def _turn_to_response(turn: ConversationTurn) -> TurnResponse:
    """Convert an internal `ConversationTurn` into the API's `TurnResponse`.

    The only translation is `id` → `turn_id`; everything else flows through.
    """
    return TurnResponse(
        turn_id=turn.id,
        session_id=turn.session_id,
        turn_number=turn.turn_number,
        user_message=turn.user_message,
        agent_response=turn.agent_response,
        retrieved_docs=turn.retrieved_docs,
        token_usage=turn.token_usage,
        timestamp=turn.timestamp,
        metadata=turn.metadata,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> dict[str, str]:
    """Cheap liveness probe — does not touch the LLM."""
    return {"status": "ok"}


@app.post(
    "/api/sessions/{session_id}/turns",
    response_model=TurnResponse,
    tags=["conversation"],
    summary="Send a message and get an answer",
)
def post_turn(
    session_id: str = Path(..., description="Stable conversation ID supplied by the client."),
    query: str = Query(..., description="The user's question or message."),
) -> TurnResponse:
    """Process one user turn. Creates the session implicitly on first call.

    Returns 200 even if the agent itself fails — the failure is captured in
    `metadata.errored` so the conversation can continue. Only structural
    problems (missing API key, etc.) produce a 5xx.
    """
    # FastAPI already 422s on a missing `query`, but an *empty* string is
    # still technically present — guard explicitly to match the spec.
    if not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")

    try:
        manager = _get_or_create_manager(session_id)
    except RuntimeError as e:
        # E.g. agent construction failed because GOOGLE_API_KEY is missing.
        logger.exception("Failed to initialize agent for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Error processing turn: {e}") from e

    logger.info(
        "POST /turns session=%s turn_n=%d query=%r",
        session_id,
        len(manager.turns) + 1,
        query[:80],
    )

    try:
        turn = manager.process_turn(query)
    except ValueError as e:
        # ConversationManager raises ValueError on empty input — already
        # filtered above, but defensive in case of future invariants.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 — last-resort fence so we don't 500-crash
        # ConversationManager catches crew failures internally, so reaching
        # here means something more fundamental broke (e.g. transport).
        logger.exception("Unhandled error processing turn for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Error processing turn: {e}") from e

    logger.info(
        "POST /turns done session=%s turn=%d tokens=%d latency=%.0fms",
        session_id,
        turn.turn_number,
        turn.token_usage.total_tokens,
        turn.metadata.get("latency_ms", 0.0),
    )
    return _turn_to_response(turn)


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionMetadata,
    tags=["conversation"],
    summary="Get session metadata",
)
def get_session(session_id: str = Path(...)) -> SessionMetadata:
    """Return session metadata + turn count. Does NOT include turn payloads."""
    manager = _get_manager_or_404(session_id)
    session = manager.session
    # `updated_at` should reflect the most recent activity. The Session
    # object itself isn't mutated by the Phase 0 manager, so fall back to
    # the latest turn's timestamp when there is one.
    updated_at = manager.turns[-1].timestamp if manager.turns else session.updated_at
    return SessionMetadata(
        session_id=session.id,
        user_id=session.user_id,
        title=session.title,
        turn_count=len(manager.turns),
        created_at=session.created_at,
        updated_at=updated_at,
        metadata=session.metadata,
    )


@app.get(
    "/api/sessions/{session_id}/turns",
    response_model=TurnsListResponse,
    tags=["conversation"],
    summary="Get conversation history",
)
def get_session_turns(
    session_id: str = Path(...),
    limit: int = Query(100, ge=1, le=1000, description="Max turns to return."),
    offset: int = Query(0, ge=0, description="Skip the first N turns."),
) -> TurnsListResponse:
    """Return turns for a session, with simple offset+limit pagination.

    Phase 0 keeps turns in memory in chronological order (turn_number=1
    first), so slicing is O(1).
    """
    manager = _get_manager_or_404(session_id)
    total = len(manager.turns)
    window = manager.turns[offset : offset + limit]
    return TurnsListResponse(
        session_id=session_id,
        total_turns=total,
        turns=[_turn_to_response(t) for t in window],
    )
