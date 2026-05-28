"""CrewAI task definitions.

A `Task` in CrewAI is a single unit of work assigned to one agent. Tasks
chain via `context=[prev_task]` — the second task receives the first task's
output as part of its prompt.

For Phase 0 we have a fixed two-step pipeline. The task descriptions are
deliberately verbose: CrewAI's prompts include them verbatim, so they double
as instructions to the LLM about what *good output* looks like.
"""

from __future__ import annotations

from crewai import Task

from src.agent.agents import KnowledgeAgents


class KnowledgeTasks:
    """Factory of `Task` objects for the conversation pipeline."""

    @staticmethod
    def retrieval_task(query: str, history: str, agents: KnowledgeAgents) -> Task:
        """Step 1: find relevant evidence."""
        return Task(
            description=(
                "You are answering this user question:\n"
                f"  USER QUESTION: {query}\n\n"
                "Recent conversation context (may be empty):\n"
                f"{history or '  (no prior turns in this session)'}\n\n"
                "Use the `retrieve` tool one or more times to gather "
                "relevant passages. Choose your search queries carefully — "
                "rephrase if the first attempt finds nothing useful. "
                "Do NOT attempt to answer the user; just collect evidence."
            ),
            expected_output=(
                "A numbered list of the most relevant passages with their "
                "sources, exactly as returned by the retrieve tool. "
                "Include a one-sentence note about whether the evidence "
                "looks sufficient to answer the question."
            ),
            agent=agents.retrieve_agent,
        )

    @staticmethod
    def answer_task(query: str, history: str, agents: KnowledgeAgents, depends_on: Task) -> Task:
        """Step 2: synthesize the answer from the retrieved evidence."""
        return Task(
            description=(
                f"USER QUESTION: {query}\n\n"
                "Recent conversation context (may be empty):\n"
                f"{history or '  (no prior turns in this session)'}\n\n"
                "Write a clear, complete answer using ONLY the evidence "
                "from the previous task. Rules:\n"
                "  • Cite every factual claim by source (call the `cite` tool).\n"
                "  • If the evidence is insufficient, say so plainly rather "
                "    than guessing.\n"
                "  • Keep the answer concise — no preamble like "
                "    'Based on the documents...'."
            ),
            expected_output=(
                "A direct natural-language answer to the user's question, "
                "with inline citations of the form `[source: filename]`."
            ),
            agent=agents.answer_agent,
            context=[depends_on],
        )
