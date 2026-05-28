"""Unit tests for `src.config.settings`."""

from __future__ import annotations

from src.config.settings import Settings


def test_defaults_load_without_env(monkeypatch) -> None:
    """Without any env vars, defaults still produce a valid Settings object."""
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "PKA_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.pka_llm_model == "gemini/gemini-2.0-flash"
    assert s.pka_embedding_model == "text-embedding-3-small"
    assert s.has_gemini is False
    assert s.has_openai is False


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    monkeypatch.setenv("OPENAI_API_KEY", "o-test")
    monkeypatch.setenv("PKA_LLM_MODEL", "gemini/gemini-2.0-flash-exp")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.has_gemini and s.has_openai
    assert s.pka_llm_model == "gemini/gemini-2.0-flash-exp"


def test_secret_does_not_leak_in_repr(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-token")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "super-secret-token" not in repr(s)
    assert "super-secret-token" not in str(s)
    # But the real value is still accessible explicitly
    assert s.google_api_key.get_secret_value() == "super-secret-token"


def test_ensure_storage_dirs_creates_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PKA_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("PKA_SQLITE_PATH", str(tmp_path / "db" / "sessions.db"))
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.ensure_storage_dirs()
    assert (tmp_path / "chroma").is_dir()
    assert (tmp_path / "db").is_dir()
