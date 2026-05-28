"""CrewAI agent definitions.

Architectural decision: Two specialists, one LLM
-------------------------------------------------
Per the architecture doc, we split the work into two CrewAI agents that share
the same Gemini instance:

* **Retrieval Specialist** — decides what to search for, calls `retrieve`.
* **Knowledge Analyst** — synthesizes an answer from retrieved docs, calls
  `cite` for provenance.

Splitting the work has two benefits even though both agents use the same
model:
1. Each agent has a *narrower* system prompt → fewer hallucinations and more
   focused tool use.
2. The two-step pipeline (retrieve → answer) maps cleanly to the two Gemini
   calls described in the token-flow diagram of the architecture doc.

If Phase 3 finds the second hop unnecessary for short queries, we can collapse
them; the data models don't change.
"""

from __future__ import annotations

from typing import Mapping

from crewai import Agent, LLM
from crewai.tools import BaseTool

from src.config import Settings, get_settings
from src.utils import get_logger

logger = get_logger(__name__)


def _build_llm(settings: Settings) -> LLM:
    """Construct the CrewAI LLM wrapper around Gemini.

    CrewAI 0.5x routes LLM calls through LiteLLM, which auto-detects the
    provider from the model prefix (`gemini/...`). We pass the API key
    explicitly rather than relying on env-var sniffing so this stays
    testable.
    """
    return LLM(
        model=settings.pka_llm_model,
        api_key=settings.google_api_key.get_secret_value(),
        # temperature kept low — this is a grounded-QA agent, not a creative one.
        temperature=0.2,
    )


class KnowledgeAgents:
    """Container for the two Phase 0 agents.

    Kept as a plain class (no inheritance) so it's trivial to introspect /
    test. CrewAI's `Agent` is what does the real work.
    """

    def __init__(
        self,
        tools: Mapping[str, BaseTool],
        *,
        settings: Settings | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = _build_llm(self.settings)

        retrieve_tool = tools.get("retrieve")
        cite_tool = tools.get("cite")
        if retrieve_tool is None or cite_tool is None:
            raise ValueError(
                "KnowledgeAgents requires both 'retrieve' and 'cite' tools."
            )

        self.retrieve_agent = Agent(
            role="Document Retrieval Specialist",
            goal=(
                "Find the most relevant passages from the user's personal "
                "knowledge base for the question being asked."
            ),
            backstory=(
                "You are an expert librarian. You read user questions "
                "carefully, decide what to search for, and call the "
                "retrieve tool with focused queries. You return raw "
                "evidence — you do NOT answer the question yourself."
            ),
            tools=[retrieve_tool],
            llm=self._llm,
            verbose=verbose,
            allow_delegation=False,
        )

        self.answer_agent = Agent(
            role="Knowledge Base Analyst",
            goal=(
                "Synthesize a clear, accurate answer from the retrieved "
                "documents, citing each source you draw from."
            ),
            backstory=(
                "You are a careful research analyst. You write answers "
                "grounded ONLY in the evidence you were given. If the "
                "evidence is missing or contradictory, you say so. You "
                "always cite the source of any factual claim."
            ),
            tools=[cite_tool],
            llm=self._llm,
            verbose=verbose,
            allow_delegation=False,
        )

        logger.debug(
            "KnowledgeAgents initialized (model=%s, tools=%s)",
            self.settings.pka_llm_model,
            list(tools.keys()),
        )
