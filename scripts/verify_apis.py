"""Acceptance check for Task 0.3 — verify both external APIs are reachable.

Run: `uv run python scripts/verify_apis.py`

Exits non-zero if any check fails, so this script doubles as a CI smoke
test. The two calls are intentionally cheap (single token / short prompt)
so running the check costs essentially nothing.
"""

from __future__ import annotations

import sys
from typing import Callable

from src.config import get_settings
from src.utils import configure_logging, get_logger

logger = get_logger(__name__)


def _check(name: str, fn: Callable[[], str]) -> bool:
    """Run one named check, log the result, return success."""
    try:
        detail = fn()
    except Exception as e:  # noqa: BLE001 — top-level smoke test wants any failure caught
        logger.error("❌ %s FAILED: %s", name, e)
        return False
    logger.info("✅ %s OK — %s", name, detail)
    return True


def check_gemini() -> str:
    """Hit Gemini with a one-token generation. Returns a status string."""
    import google.generativeai as genai

    settings = get_settings()
    if not settings.has_gemini:
        raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")

    genai.configure(api_key=settings.google_api_key.get_secret_value())
    # Use the bare model name without the LiteLLM provider prefix here, since
    # we're talking directly to the Google SDK (not via CrewAI/LiteLLM).
    model_name = settings.pka_llm_model.split("/", 1)[-1]
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content("Reply with the single word: ok")
    text = (resp.text or "").strip()
    return f"model={model_name!r} replied {text!r}"


def check_openai_embeddings() -> str:
    """Embed a short string. Confirms the embeddings endpoint is reachable."""
    from openai import OpenAI

    settings = get_settings()
    if not settings.has_openai:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    resp = client.embeddings.create(
        model=settings.pka_embedding_model,
        input="hello",
    )
    dim = len(resp.data[0].embedding)
    return f"model={settings.pka_embedding_model!r} returned {dim}-d vector"


def main() -> int:
    configure_logging()
    logger.info("Verifying Phase 1 API credentials...")
    results = [
        _check("Gemini 2.0 Flash", check_gemini),
        _check("OpenAI Embeddings", check_openai_embeddings),
    ]
    if all(results):
        logger.info("All checks passed — you're ready to build.")
        return 0
    logger.error("One or more checks failed. See errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
