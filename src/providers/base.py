"""Abstract base classes for LLM and embedding providers.

Why two separate ABCs rather than one combined one?
---------------------------------------------------
Embeddings and chat completions are independent capabilities — a user might
want OpenAI for chat and Ollama for embeddings (or vice versa) to balance
cost against quality. Splitting the ABCs lets `factory.get_llm_provider`
and `factory.get_embedding_provider` pick mixes without coercing every
vendor wrapper to implement both halves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base for LLM providers (Ollama, Gemini, OpenAI, ...)."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the provider with vendor-specific kwargs."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Send a single prompt and return the model's text response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The vendor-specific model identifier (e.g. `mistral`, `gpt-4o-mini`)."""


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the provider with vendor-specific kwargs."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single string. Used for search queries."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Used during document ingestion."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Length of vectors this provider produces. Stable per model."""
