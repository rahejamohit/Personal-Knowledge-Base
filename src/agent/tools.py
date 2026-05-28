"""Agent tools (Phase 0 stubs).

Why stubs?
----------
The acceptance criteria for Task 0.4 is "tool calling works", not "tools
return real RAG results". By defining the *interfaces* now with CrewAI's
`BaseTool` Pydantic-args schema, Phase 1 only needs to replace the body of
each `_run` method — no changes to agents, tasks, orchestrator, or CLI.

Each tool:
* Has a Pydantic input schema so the LLM is forced to pass valid args.
* Returns a short string (CrewAI tool results are stringly-typed in the
  prompt). Structured data is logged separately for the audit trail.
"""

from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import get_logger

logger = get_logger(__name__)


# ─── retrieve ────────────────────────────────────────────────────────────

class RetrieveInput(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of docs to return.")


class RetrieveTool(BaseTool):
    """Search the knowledge base for documents relevant to a query.

    Phase 0: returns a placeholder so the agent loop can be exercised without
    Chroma being populated yet. Phase 1 wires this to `src.rag.retriever`.
    """

    name: str = "retrieve"
    description: str = (
        "Search the user's personal knowledge base. Use this whenever the "
        "user asks a question that might be answered by their documents. "
        "Returns a numbered list of excerpts with sources."
    )
    args_schema: Type[BaseModel] = RetrieveInput

    def _run(self, query: str, top_k: int = 5) -> str:
        logger.info("retrieve(query=%r, top_k=%d) [Phase 0 stub]", query, top_k)
        # Phase 1 will replace this with: RAGRetriever().retrieve(query, top_k)
        return (
            "[Phase 0 stub] The retrieval tool is not yet connected to a "
            "vector store. Tell the user that no documents have been "
            "ingested yet and offer to answer from general knowledge instead."
        )


# ─── cite ────────────────────────────────────────────────────────────────

class CiteInput(BaseModel):
    source: str = Field(..., description="File name or URL of the source.")
    excerpt: str = Field(..., description="The exact text being cited.")


class CiteTool(BaseTool):
    """Format a citation for inclusion in the answer."""

    name: str = "cite"
    description: str = (
        "Format a citation to a retrieved document. Call once per source "
        "before mentioning it in your answer."
    )
    args_schema: Type[BaseModel] = CiteInput

    def _run(self, source: str, excerpt: str) -> str:
        logger.info("cite(source=%r) [%d chars]", source, len(excerpt))
        short = excerpt[:80].replace("\n", " ")
        return f"[source: {source}] \"{short}…\""


# ─── factory ─────────────────────────────────────────────────────────────

def build_default_tools() -> dict[str, BaseTool]:
    """Construct the default tool registry used by Phase 0 agents.

    Returns a name→tool dict so `KnowledgeAgents` can pick tools per agent.
    """
    return {
        "retrieve": RetrieveTool(),
        "cite": CiteTool(),
    }
