"""Google Gemini LLM and embedding provider.

Thin wrappers around `langchain-google-genai`. We use LangChain rather
than the raw Google SDK so we get a consistent `.invoke(str) -> message`
shape across providers — this matters for Phase 1's RAG pipeline which
will accept any `EmbeddingProvider`/`LLMProvider`.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.providers.base import EmbeddingProvider, LLMProvider


class GeminiLLM(LLMProvider):
    """LLM provider using Google Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.7,
            )
            self._model = model
        except Exception as e:  # noqa: BLE001 — SDK raises a mix of vendor exceptions
            raise RuntimeError(f"Failed to initialize Gemini: {e}") from e

    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            # ChatGoogleGenerativeAI returns an AIMessage; `.content` is a str
            # for text-only responses (which is all we use today).
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling Gemini: {e}") from e

    @property
    def model_name(self) -> str:
        return self._model


class GeminiEmbeddings(EmbeddingProvider):
    """Embedding provider using Google's embedding-001 (768-d)."""

    def __init__(self, api_key: str, model: str = "models/embedding-001") -> None:
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key,
            )
            self._model = model
            self._embedding_dimension = 768
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize Gemini embeddings: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling Gemini embeddings: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error calling Gemini embeddings: {e}") from e

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension
