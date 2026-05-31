"""Provider abstraction layer for LLMs and embeddings.

Phase 0 scaffolding: the agent loop still goes through CrewAI's LiteLLM
adapter, but Phase 1's RAG pipeline (embeddings, query expansion) will
construct providers via `src.providers.factory.get_*_provider` so the rest
of the system doesn't need to know which vendor is in use.
"""

from src.providers.base import EmbeddingProvider, LLMProvider
from src.providers.factory import get_embedding_provider, get_llm_provider

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
]
