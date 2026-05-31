"""OpenAI LLM and embedding provider.

Note: the LangChain class is also called `OpenAIEmbeddings`, which would
shadow our local class if imported directly. We alias the import to
`_LCOpenAIEmbeddings` so `self.embeddings = _LCOpenAIEmbeddings(...)`
unambiguously refers to LangChain's implementation.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as _LCOpenAIEmbeddings

from src.providers.base import EmbeddingProvider, LLMProvider


class OpenAILLM(LLMProvider):
    """LLM provider using OpenAI's chat models."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            self.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=0.7,
            )
            self._model = model
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize OpenAI: {e}") from e

    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling OpenAI: {e}") from e

    @property
    def model_name(self) -> str:
        return self._model


# Dimensions for each text-embedding-3-* model. `text-embedding-ada-002`
# is 1536-d too; the lookup falls back to 1536 for anything unrecognized.
_OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings(EmbeddingProvider):
    """Embedding provider using OpenAI's `text-embedding-3-*` models."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        try:
            self.embeddings = _LCOpenAIEmbeddings(model=model, api_key=api_key)
            self._model = model
            self._embedding_dimension = _OPENAI_EMBEDDING_DIMS.get(model, 1536)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling OpenAI embeddings: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling OpenAI embeddings: {e}") from e

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension
