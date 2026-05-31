"""Factory: settings → concrete provider instance.

The two `get_*_provider` functions are the only public API for vendor
selection. Anything that needs an LLM or embedding model should call
these rather than constructing a vendor class directly — that keeps the
"how do I pick a provider" logic in exactly one place.

Caching
-------
We memoize on the *resolved provider name*, not on the optional override
arg, so:

    get_llm_provider()           # reads settings.llm_provider, caches
    get_llm_provider("openai")   # different key, separate cache entry

This lets tests fetch each provider in isolation without `cache_clear`.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.config.settings import ProviderName
from src.providers.base import EmbeddingProvider, LLMProvider


def get_llm_provider(provider: ProviderName | None = None) -> LLMProvider:
    """Return a configured `LLMProvider` for the requested vendor.

    `provider=None` (the default) reads `settings.llm_provider`.
    """
    name = provider or get_settings().llm_provider
    return _cached_llm(name)


def get_embedding_provider(provider: ProviderName | None = None) -> EmbeddingProvider:
    """Return a configured `EmbeddingProvider` for the requested vendor."""
    name = provider or get_settings().embedding_provider
    return _cached_embedding(name)


def get_litellm_model_and_key() -> tuple[str, str]:
    """Return the `(model, api_key)` pair ready to hand to CrewAI's `LLM(...)`.

    Centralizes everything CrewAI's `LLM` wrapper needs to talk to the
    currently-configured chat provider:

    * Reads the active provider from `settings.llm_provider`.
    * Builds the LiteLLM-compatible model string — LiteLLM auto-detects
      the vendor from the `<provider>/<model>` prefix, except for OpenAI
      which takes a bare model name.
    * Extracts the right API key (or the `"ollama"` placeholder for the
      keyless local provider).
    * Validates that the matching key is present and raises with an
      actionable error if not.

    Returns:
        Tuple `(model_string, api_key)`. Examples:
          - `("ollama/mistral", "ollama")`
          - `("gemini/gemini-2.0-flash", "<key>")`
          - `("gpt-4o-mini", "<key>")`

    Raises:
        RuntimeError: provider is selected but its credential is missing.
        ValueError: provider name is not one of `ollama|gemini|openai`.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "ollama":
        # Ollama is keyless, but LiteLLM requires *some* string in the
        # `api_key` field. `"ollama"` is the conventional placeholder.
        return f"ollama/{settings.ollama_llm_model}", "ollama"

    if provider == "gemini":
        if not settings.has_gemini:
            raise RuntimeError(
                "Gemini provider selected but GOOGLE_API_KEY is not set. "
                "Get a free key from: https://aistudio.google.com/app/apikey"
            )
        return (
            f"gemini/{settings.gemini_llm_model}",
            settings.google_api_key.get_secret_value(),
        )

    if provider == "openai":
        if not settings.has_openai:
            raise RuntimeError(
                "OpenAI provider selected but OPENAI_API_KEY is not set. "
                "Get a key from: https://platform.openai.com/api-keys"
            )
        # OpenAI: bare model name (no `openai/` prefix in LiteLLM).
        return settings.openai_llm_model, settings.openai_api_key.get_secret_value()

    raise ValueError(f"Unknown LLM provider: {provider}")


# ─── Cached factories (keyed by provider name) ───────────────────────────

@lru_cache(maxsize=4)
def _cached_llm(name: ProviderName) -> LLMProvider:
    settings = get_settings()
    if name == "ollama":
        from src.providers.ollama import OllamaLLM
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
        )
    if name == "gemini":
        if not settings.has_gemini:
            raise ValueError("GOOGLE_API_KEY not set for Gemini provider")
        from src.providers.gemini import GeminiLLM
        return GeminiLLM(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.gemini_llm_model,
        )
    if name == "openai":
        if not settings.has_openai:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        from src.providers.openai import OpenAILLM
        return OpenAILLM(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_llm_model,
        )
    raise ValueError(f"Unknown LLM provider: {name}")


@lru_cache(maxsize=4)
def _cached_embedding(name: ProviderName) -> EmbeddingProvider:
    settings = get_settings()
    if name == "ollama":
        from src.providers.ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    if name == "gemini":
        if not settings.has_gemini:
            raise ValueError("GOOGLE_API_KEY not set for Gemini provider")
        from src.providers.gemini import GeminiEmbeddings
        return GeminiEmbeddings(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.gemini_embedding_model,
        )
    if name == "openai":
        if not settings.has_openai:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        from src.providers.openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
        )
    raise ValueError(f"Unknown embedding provider: {name}")
