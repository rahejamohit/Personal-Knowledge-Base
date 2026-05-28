"""Crew orchestration — assemble agents + tasks and run them.

This is the boundary the `ConversationManager` calls into. Keeping the Crew
construction here (rather than inside `ConversationManager`) means we can
unit-test the agent pipeline in isolation by passing fake tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from crewai import Crew, Process
from crewai.tools import BaseTool

from src.agent.agents import KnowledgeAgents
from src.agent.tasks import KnowledgeTasks
from src.config import Settings, get_settings
from src.models.conversation import TokenUsage
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AgentRunResult:
    """Output of one full crew run.

    `token_usage` is best-effort — CrewAI surfaces usage via the crew's
    `usage_metrics` attribute, but the exact shape varies across versions,
    so we defensively coerce.
    """

    answer: str
    token_usage: TokenUsage


class KnowledgeAgent:
    """High-level orchestrator: one method, `answer_query`."""

    def __init__(
        self,
        tools: Mapping[str, BaseTool],
        *,
        settings: Settings | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.verbose = verbose
        self.agents = KnowledgeAgents(tools, settings=self.settings, verbose=verbose)

    def answer_query(self, query: str, history: str = "") -> AgentRunResult:
        """Run the two-task crew synchronously and return the final answer.

        `history` is a pre-formatted string the `ConversationManager` builds
        from prior turns. We keep this method synchronous because CrewAI's
        `kickoff()` is itself synchronous — pretending it's async would only
        add a fake `to_thread` wrapper.
        """
        retrieval = KnowledgeTasks.retrieval_task(query, history, self.agents)
        answer = KnowledgeTasks.answer_task(query, history, self.agents, depends_on=retrieval)

        crew = Crew(
            agents=[self.agents.retrieve_agent, self.agents.answer_agent],
            tasks=[retrieval, answer],
            process=Process.sequential,
            verbose=self.verbose,
        )

        logger.info("Crew kickoff: query=%r", query[:120])
        result = crew.kickoff()

        # CrewAI returns a `CrewOutput` object in current versions; older
        # versions returned a string. Handle both.
        answer_text = getattr(result, "raw", None) or str(result)

        usage = _extract_usage(crew)
        logger.info("Crew finished: tokens=%s", usage.total_tokens)
        return AgentRunResult(answer=answer_text.strip(), token_usage=usage)


def _extract_usage(crew: Crew) -> TokenUsage:
    """Best-effort extraction of token usage from CrewAI's metrics dict.

    Older CrewAI versions store usage on `crew.usage_metrics` as a flat dict
    with keys like `total_tokens`, `prompt_tokens`, `completion_tokens`.
    Newer versions return a pydantic model. We coerce both.
    """
    raw = getattr(crew, "usage_metrics", None)
    if raw is None:
        return TokenUsage()
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
    )
