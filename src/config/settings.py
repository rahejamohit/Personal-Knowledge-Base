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

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ─── API keys ─────────────────────────────────────────
    # Required for the agent loop. Marked SecretStr so the value won't leak
    # into logs / `repr(settings)` output.
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google AI Studio API key for Gemini.",
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key (used only for embeddings in Phase 1).",
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

    # ─── Convenience accessors ────────────────────────────
    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key.get_secret_value())

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.get_secret_value())

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
