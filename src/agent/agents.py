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
    """Construct the CrewAI LLM wrapper for the configured provider.

    All provider dispatch (model-string assembly, key extraction,
    validation) is delegated to `factory.get_litellm_model_and_key()` so
    this function stays a thin adapter. `settings` is kept in the
    signature for backwards-compat with callers, even though we read it
    indirectly through the factory now.

    Temperature is intentionally low (0.2) — this is a grounded-QA agent,
    not a creative writer. The per-provider classes in `src.providers`
    use 0.7 for general-purpose use.
    """
    # Local import: `src.providers.factory` imports `src.config`, which is
    # safe to import eagerly, but keeping this lazy means a future cycle
    # (e.g. the factory importing from `src.agent`) won't blow up at
    # module load.
    from src.providers.factory import get_litellm_model_and_key

    try:
        model, api_key = get_litellm_model_and_key()
    except RuntimeError as e:
        logger.error("Failed to configure LLM provider: %s", e)
        raise

    return LLM(model=model, api_key=api_key, temperature=0.2)


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
                "Answer the user's question using ONLY facts that appear "
                "verbatim in the retrieved evidence. Never invent, infer, "
                "or recall facts from anywhere else. If the evidence does "
                "not contain the answer, say so explicitly."
            ),
            backstory=(
                "You are a careful research analyst whose single most "
                "important duty is to never state anything the retrieved "
                "evidence does not support.\n\n"
                "YOUR RULES:\n"
                "  1. NEVER make up dates, names, numbers, or facts.\n"
                "  2. NEVER infer or assume information not explicitly "
                "stated in the evidence.\n"
                "  3. NEVER fall back on your own training knowledge to "
                "fill a gap — the evidence is the ONLY source of truth.\n"
                "  4. ALWAYS cite the source of every factual claim.\n"
                "  5. ALWAYS state plainly when the evidence does not "
                "contain the requested information.\n"
                "  6. Copy specific values (dates, numbers, names) exactly "
                "as they appear — do not paraphrase or round them.\n\n"
                "HALLUCINATION EXAMPLE (DO NOT DO THIS):\n"
                "  Q: When did I apply for parental leave?\n"
                "  BAD: 'You applied on March 1, 2022, June 15, 2022, "
                "and August 20, 2022.'\n"
                "       (These dates appear in NO document — they are "
                "invented. This destroys the user's trust.)\n"
                "  GOOD: 'Your application is dated May 1, 2026. "
                "[source: Parental Leave application.pdf]'\n\n"
                "Being honest about what the documents do and do not "
                "contain is far more valuable than a confident guess. A "
                "wrong fact is worse than 'I don't know.'"
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
