"""Typed application settings via pydantic-settings.

Why pydantic-settings?
----------------------
* Single source of truth for *all* configuration, with types validated at
  startup (a typo'd env var fails fast rather than 30 minutes into an API
  call).
* Reads from `.env` automatically — no manual `os.getenv` scattered around.
* `get_settings()` is memoized so we parse the env exactly once per process.

Secrets are `SecretStr`s — they refuse to render via `repr()`/`str()`, which
keeps API keys out of logs and tracebacks by default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "gemini", "openai"]


class Settings(BaseSettings):
    """All runtime configuration. Populated from environment + `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Don't crash if `.env` is missing — useful for CI / tests that
        # pass values directly.
        extra="ignore",
    )

    # ─── Provider selection ───────────────────────────────
    # Chosen at runtime via env (`LLM_PROVIDER=ollama|gemini|openai`).
    # Defaults to `ollama` so a fresh checkout works without any cloud
    # credentials — `ollama serve` on localhost is all that's needed.
    llm_provider: ProviderName = Field(
        default="ollama",
        description="Which LLM backend the agent talks to.",
    )
    embedding_provider: ProviderName = Field(
        default="ollama",
        description="Which embedding backend Phase 1 RAG will use.",
    )

    # ─── API keys ─────────────────────────────────────────
    # Required only when the matching provider is selected. Marked SecretStr
    # so the value won't leak into logs / `repr(settings)` output.
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google AI Studio API key (required when llm/embedding_provider=gemini).",
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key (required when llm/embedding_provider=openai).",
    )

    # ─── Ollama configuration ─────────────────────────────
    # Local-first defaults. `nomic-embed-text` emits 768-d vectors.
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )
    ollama_llm_model: str = Field(
        default="mistral",
        description="Ollama LLM model tag (e.g. `mistral`, `llama3.2`).",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model tag.",
    )

    # ─── Gemini configuration ─────────────────────────────
    gemini_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model name (no provider prefix).",
    )
    gemini_embedding_model: str = Field(
        default="models/embedding-001",
        description="Google embedding model identifier.",
    )

    # ─── OpenAI configuration ─────────────────────────────
    openai_llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model name.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    # ─── Model selection ──────────────────────────────────
    # CrewAI delegates to LiteLLM under the hood; the model string uses
    # LiteLLM's <provider>/<model> convention.
    pka_llm_model: str = Field(
        default="gemini/gemini-2.0-flash",
        description="LiteLLM-style provider/model identifier.",
    )
    pka_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    # ─── Storage paths (Phase 1: all local) ───────────────
    pka_chroma_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Chroma's persistent-client directory.",
    )
    pka_sqlite_path: Path = Field(
        default=Path("./data/sessions.db"),
        description="SQLite file for session/turn persistence.",
    )

    # ─── Runtime knobs ────────────────────────────────────
    pka_log_level: str = Field(default="INFO")
    pka_log_format: str = Field(
        default="text", description="'text' (dev) or 'json' (production)."
    )
    pka_max_context_tokens: int = Field(
        default=32_000,
        ge=1_000,
        description="Soft cap on tokens packed into one Gemini call.",
    )
    pka_history_turns: int = Field(
        default=10,
        ge=1,
        description="Recent turns to include verbatim before summarization (Phase 2).",
    )

    # ─── Phase 1: ingestion chunking knobs ────────────────
    # Token-budgeted chunking. 512/50 is the same default the architecture
    # doc recommends — large enough that a chunk is a meaningful unit of
    # text, small enough to keep ~5 chunks within a single Gemini call.
    chunk_size: int = Field(
        default=512,
        ge=64,
        description="Target tokens per chunk during ingestion.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Tokens of overlap between adjacent chunks (preserves cross-chunk context).",
    )

    # ─── Convenience accessors ────────────────────────────
    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key.get_secret_value())

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.get_secret_value())

    @property
    def has_ollama(self) -> bool:
        """Cheap reachability probe for the local Ollama server.

        Imports `requests` lazily so that callers who never invoke this
        property don't pay the import cost (and so we don't add a hard
        dependency on `requests` for non-Ollama setups, even though it's
        in `pyproject.toml`).
        """
        try:
            import requests  # local import — see docstring
        except ImportError:
            return False
        try:
            return requests.get(f"{self.ollama_base_url}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def ensure_storage_dirs(self) -> None:
        """Create local storage directories if they don't exist yet.

        Called by the CLI on startup so the first run doesn't fail with a
        confusing "No such file or directory" from SQLite or Chroma.
        """
        self.pka_chroma_dir.mkdir(parents=True, exist_ok=True)
        self.pka_sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Memoized so repeated calls don't re-parse the env. Tests that need
    different settings should call `get_settings.cache_clear()` after
    mutating the environment.
    """
    return Settings()
