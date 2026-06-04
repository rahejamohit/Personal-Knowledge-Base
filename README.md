# Personal Knowledge Agent

[![Tests](https://github.com/rahejamohit/Personal-Knowledge-Base/actions/workflows/test.yml/badge.svg)](https://github.com/rahejamohit/Personal-Knowledge-Base/actions/workflows/test.yml)

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

# 2. Install system binaries for scanned-PDF OCR (Phase 1.1)
#    macOS:  brew install tesseract poppler
#    Linux:  apt-get install tesseract-ocr poppler-utils
#    These are only needed if you ingest scanned (image-only) PDFs.
#    Pure-text PDFs / Markdown / TXT / DOCX work without them.

# 3. Create venv + install deps from pyproject.toml
uv sync --extra dev

# 4. Config
cp env.example .env       # then edit .env with real API keys
#  - GOOGLE_API_KEY  → https://aistudio.google.com/app/apikey
#  - OPENAI_API_KEY  → https://platform.openai.com/api-keys

# 5. Verify both APIs work
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

### Testing strategy

```
        🔺 Eval tests (~10%)         tests/evals/
       ├─ Quality metrics (recall, accuracy, citations)
       └─ Full pipeline against 12 curated test cases

      🔻 Integration tests (~20%)    tests/integration/
     ├─ End-to-end CrewAI run against real Gemini
     └─ SessionManager round-trips, vector store persistence

   📦 Unit tests (~70%)              tests/unit/
  ├─ Data models, settings, conversation manager
  ├─ Vector store ops (upsert / search / delete)
  └─ Agent + RAG tool stubs
```

Unit tests run on every push (GitHub Actions). Integration tests are
gated by `-m integration` and require live API keys. Eval tests run
manually pre-release to score retrieval quality on the
[sample corpus](tests/evals/fixtures/sample_docs/).

## Embeddings

The agent has two pluggable embedding paths, switched by
`EMBEDDING_PROVIDER` in `.env`:

| Phase | Provider | Model | Dims | Cost | Speed (laptop) |
|-------|----------|-------|------|------|----------------|
| Phase 1 (default) | Ollama | `nomic-embed-text` | 768 | $0 | ~500 chunks/sec on CPU |
| Phase 2+ | OpenAI | `text-embedding-3-small` | 1536 | $0.02 / 1M tokens | ~5000 chunks/sec (API) |

```bash
# Phase 1 setup
ollama pull nomic-embed-text     # ~270 MB
# Phase 2+ setup
echo "EMBEDDING_PROVIDER=openai" >> .env
echo "OPENAI_API_KEY=sk-..."     >> .env
```

**Switching embedding models invalidates the index** — Chroma stores
vectors at a fixed dimension, and similarity scoring breaks if you
mix models. After flipping `EMBEDDING_PROVIDER`, `rm -rf .chroma/`
and re-ingest.

## API costs

### Phase 1 — local-only ($0 / month)

| Component | Cost |
|-----------|------|
| Ollama LLM (`mistral`) | $0 |
| Ollama embeddings (`nomic-embed-text`) | $0 |
| Chroma vector DB | $0 (local disk) |
| SQLite session store | $0 (local disk) |

**Phase 1 total: $0.** Once the two Ollama models are pulled, no
network calls are made.

### Phase 2+ — managed providers (estimated)

| Component | Approx. cost |
|-----------|--------------|
| Gemini 2.0 Flash (chat) | ~$0.001 / query, 1500 req/day free tier |
| OpenAI `text-embedding-3-small` | $0.02 / 1M input tokens (~$0.0001 / 1K) |
| Pinecone (starter index) | Free tier ≤ 1M vectors, then ~$0.025 / GB-mo |
| Managed Postgres (session DB) | ~$15–50 / month, if you outgrow SQLite |

**Phase 2+ working estimate: $0–100 / month** depending on traffic and
which providers you enable. The provider selector supports mixing
(e.g. local Ollama chat + cloud OpenAI embeddings for better recall).

## Verification commands

Quick health checks for what's wired up today. Each exits non-zero on
failure, so they're safe to chain in a setup script.

```bash
# 1. Settings parse + provider config
uv run python -c "from src.config import get_settings; \
  s = get_settings(); print(f'OK config — llm={s.llm_provider!r}, emb={s.embedding_provider!r}')"

# 2. Data models import + Phase-1 schema
uv run python -c "from src.models.document import DocumentChunk, RetrievedDoc; \
  from src.models.conversation import ConversationTurn, TokenUsage; print('OK models')"

# 3. Ingestion loaders / chunker stubs resolve
uv run python -c "from src.ingestion import get_loader, recursive_split, CHUNK_SIZE; \
  print(f'OK ingestion — chunk_size={CHUNK_SIZE}, loader for .md is {type(get_loader(\"x.md\")).__name__}')"

# 4. Vector store boots and reports stats
uv run python -c "from src.storage.vector_store import ChromaVectorStore; \
  import tempfile, asyncio; \
  s = ChromaVectorStore(persist_dir=tempfile.mkdtemp()); \
  print('OK vector store —', asyncio.run(s.get_stats()))"

# 5. SessionManager opens its DB + creates a row
uv run python -c "from src.storage.db import SessionManager; \
  import tempfile, asyncio, os; \
  m = SessionManager(db_path=os.path.join(tempfile.mkdtemp(), 'sessions.db')); \
  sid = asyncio.run(m.create_session('alice')); print('OK session manager —', sid)"

# 6. RAG tool stubs callable
uv run python -c "import asyncio; from src.agent.tools import retrieve, cite; \
  print('OK tools —', asyncio.run(retrieve('test')), asyncio.run(cite('c1')))"

# 7. External APIs reachable (only if Gemini / OpenAI keys set in .env)
uv run python scripts/verify_apis.py
```

## What's next (Phase 1)

The agent loop in Phase 0 wires up the *shape* of the system but the
`retrieve` tool currently returns a stub. Phase 1 fills in:

1. Document ingestion (PDF/MD/TXT → chunks → embeddings → Chroma)
2. RAG retrieval engine (top-K + metadata filters)
3. SQLite session/turn persistence
4. Real `retrieve` and `cite` tool implementations

See `Detailed_Task_Breakdown.md` for the full breakdown.
