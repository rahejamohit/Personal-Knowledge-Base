"""Ollama LLM and embedding provider.

We talk to Ollama via its HTTP API directly using `requests` rather than
the `langchain-ollama` package — fewer transitive deps, and the wire
format is small enough that the wrapper is a few lines.

The two classes share a `_verify_connection` because a missing local
Ollama server is by far the most common failure mode and we want a clear
error at construction time rather than a cryptic timeout on first use.
"""

from __future__ import annotations

import requests

from src.providers.base import EmbeddingProvider, LLMProvider


def _verify_ollama(base_url: str) -> None:
    """Probe `<base_url>/api/tags`. Raises RuntimeError if unreachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Ollama server not running at {base_url}. Start it with: ollama serve"
        ) from e


class OllamaLLM(LLMProvider):
    """LLM provider using a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
    ) -> None:
        self.base_url = base_url
        self._model = model
        self.generate_url = f"{base_url}/api/generate"
        _verify_ollama(base_url)

    def invoke(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.generate_url,
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            raise RuntimeError(f"Error calling Ollama: {e}") from e

    @property
    def model_name(self) -> str:
        return self._model


class OllamaEmbeddings(EmbeddingProvider):
    """Embedding provider using a local Ollama server.

    Default `nomic-embed-text` returns 768-d vectors. If you swap to a
    different model with a different output size, pass the right
    `_embedding_dimension` — Chroma collections are pinned to a single
    dimension per index.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self.base_url = base_url
        self._model = model
        self.embed_url = f"{base_url}/api/embeddings"  # Ollama: /api/embeddings, singular result
        self._embedding_dimension = 768
        _verify_ollama(base_url)

    def embed_query(self, text: str) -> list[float]:
        try:
            response = requests.post(
                self.embed_url,
                json={"model": self._model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            raise RuntimeError(f"Error calling Ollama embeddings: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Ollama's /api/embeddings is one-at-a-time. Phase 1 can batch in
        # parallel via a thread pool if throughput becomes a problem.
        return [self.embed_query(t) for t in texts]

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension
