# Personal Knowledge Agent

A multi-turn conversational agent that answers questions about your personal
documents with citations, persistent memory, and a CrewAI multi-agent
backbone powered by Gemini 2.0 Flash.

This repo is structured around the four phases laid out in the architecture
docs (`Personal_Knowledge_Agent_Architecture.md`, `Detailed_Task_Breakdown.md`).
**This commit implements Phase 0 only** — project skeleton, data models,
config, and a working CrewAI agent loop with a CLI you can talk to.

## Phase 0 status

| Task | What it covers | Where it lives |
|------|----------------|----------------|
| 0.1 Project setup | uv project, dirs, logging, gitignore | `pyproject.toml`, `src/utils/logging.py` |
| 0.2 Data models | Pydantic v2 models for turns, docs, memory, tools | `src/models/` |
| 0.3 API keys | `.env` template, settings loader, verify script | `env.example`, `src/config/settings.py`, `scripts/verify_apis.py` |
| 0.4 Agent loop | CrewAI agents/tasks/crew, conversation manager, CLI | `src/agent/`, `src/cli/main.py` |

## Stack (Phase 1)

| Layer | Choice | Why |
|-------|--------|-----|
| LLM | Gemini 2.0 Flash | 1M context, fast, ~$0.075/1M in |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-d, cheap, strong baseline |
| Agent framework | CrewAI | Multi-agent native, tiny boilerplate |
| Vector DB | Chroma (local) | Zero setup, free, easy Pinecone migration |
| Session store | SQLite | Embedded, durable, no service to run |
| Package mgr | uv | Fast, reproducible lockfile |

## Setup

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv + install deps from pyproject.toml
uv sync --extra dev

# 3. Config
cp env.example .env       # then edit .env with real API keys
#  - GOOGLE_API_KEY  → https://aistudio.google.com/app/apikey
#  - OPENAI_API_KEY  → https://platform.openai.com/api-keys

# 4. Verify both APIs work
uv run python scripts/verify_apis.py
```

> **Note on the env template:** the file is named `env.example` (no leading
> dot) because some sandboxed environments treat `.env*` as protected. The
> destination file is the conventional `.env`.

## Try the agent

```bash
# Interactive chat (multi-turn, in-memory; persistence lands in Phase 1)
uv run pka chat

# Or directly
uv run python -m src.cli.main chat
```

You should see something like:

```
You: What is RAG?
Agent: Retrieval-Augmented Generation (RAG) is a technique that ...
```

Multi-turn context is maintained for the duration of the CLI session. Phase 1
adds disk-backed session storage so context survives restarts.

## Layout

```
.
├── pyproject.toml
├── env.example                # template → copy to .env
├── README.md
├── src/
│   ├── agent/                 # CrewAI agents, tasks, orchestrator, conv mgr
│   ├── cli/                   # `pka` Typer CLI
│   ├── config/                # pydantic-settings: env → typed Settings
│   ├── memory/                # Phase 2: hierarchical memory
│   ├── models/                # Pydantic v2 data models (Task 0.2)
│   ├── rag/                   # Phase 1: retrieval + reranking
│   ├── storage/               # Phase 1: Chroma + SQLite
│   └── utils/                 # logging, id generation
├── scripts/
│   └── verify_apis.py         # Task 0.3 acceptance check
└── tests/
    ├── unit/
    └── integration/
```

## Run tests

```bash
uv run pytest                       # unit tests only by default
uv run pytest -m integration        # hits real APIs (needs .env)
uv run pytest --cov=src             # coverage
```

## What's next (Phase 1)

The agent loop in Phase 0 wires up the *shape* of the system but the
`retrieve` tool currently returns a stub. Phase 1 fills in:

1. Document ingestion (PDF/MD/TXT → chunks → embeddings → Chroma)
2. RAG retrieval engine (top-K + metadata filters)
3. SQLite session/turn persistence
4. Real `retrieve` and `cite` tool implementations

See `Detailed_Task_Breakdown.md` for the full breakdown.
