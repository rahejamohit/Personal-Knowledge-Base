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
                "YOUR JOB:\n"
                "  1. Use the `retrieve` tool to search for relevant documents.\n"
                "  2. Try multiple searches with different keywords if needed.\n"
                "  3. Return EXACTLY what the tool returns:\n"
                "     - If tool returns documents: list them clearly\n"
                "     - If tool returns NOTHING: say 'NO DOCUMENTS FOUND'\n"
                "  4. Do NOT invent or guess what documents might exist.\n"
                "  5. Do NOT reference previous conversation topics as if they're documents.\n"
                "  6. Do NOT say things like 'I found documents about X' unless the tool actually returned them.\n\n"
                "CRITICAL:\n"
                "  Past conversation history is NOT the same as retrieved documents.\n"
                "  Only report what the retrieve tool actually returns.\n"
                "  If retrieve returns nothing, your job is to clearly state that."
            ),
            expected_output=(
                "Either:\n"
                "  A) A numbered list of relevant passages the tool actually returned, OR\n"
                "  B) 'NO DOCUMENTS FOUND' if the retrieve tool returned no results.\n\n"
                "Include one sentence assessing if the evidence is sufficient to answer the question."
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
                "YOUR JOB:\n"
                "  1. Read what the previous task returned about retrieved documents.\n"
                "  2. If it says 'NO DOCUMENTS FOUND':\n"
                "     → Answer ONLY: 'I don't have information about that in your knowledge base.'\n"
                "     → Stop. Do NOT invent documents or infer from conversation.\n"
                "  3. If documents were returned:\n"
                "     → FIRST, silently list every fact in the evidence related to the\n"
                "       question — every date, period, number, name, and condition.\n"
                "     → THEN write an answer that includes ALL of those facts. Do NOT\n"
                "       stop at the first matching fact.\n"
                "     → Answer the question BROADLY, not literally: if the user asks\n"
                "       about 'dates' and the evidence has an application date AND a\n"
                "       leave period AND a weekly-hours figure, report ALL of them and\n"
                "       label what each one is.\n"
                "     → Combine information across documents when needed for a full answer.\n"
                "     → Cite each claim with a source.\n\n"
                "SYNTHESIS IS GOOD (do this):\n"
                "  ✓ Combining multiple facts from the retrieved documents.\n"
                "  ✓ Writing a complete, well-organized, readable answer.\n"
                "  ✓ Including all relevant details the documents contain.\n\n"
                "HALLUCINATION IS BAD (never do this):\n"
                "  ✗ Making up documents that were not retrieved.\n"
                "  ✗ Treating conversation history as if it were a document.\n"
                "  ✗ Stating facts not found in the retrieved documents.\n"
                "  ✗ Claiming 'I found documents about X' when retrieval was empty.\n\n"
                "ANTI-HALLUCINATION RULES:\n"
                "  • Past conversations ≠ documents in knowledge base.\n"
                "  • 'No relevant documents' ≠ 'documents exist but don't answer'.\n"
                "  • Conversation mentions X ≠ X is in your knowledge base.\n\n"
                "EXAMPLES:\n"
                "  GOOD SYNTHESIS (documents were retrieved):\n"
                "    'You applied on May 1, 2026, for leave from June 28 to November 27,\n"
                "     2026, working 32 hours per week. [source: Parental Leave application.pdf]'\n"
                "    (Combined several facts from the same document into one answer.)\n\n"
                "  BAD HALLUCINATION (retrieval was empty):\n"
                "    'I found employment contracts but they don't contain cricket info.'\n"
                "    (Implies documents were found when they weren't.)\n"
                "  GOOD instead:\n"
                "    'I don't have information about cricket matches in your knowledge base.'"
            ),
            expected_output=(
                "Either:\n"
                "  A) A direct, complete answer using ALL retrieved documents, with\n"
                "     inline citations of the form `[source: filename]` and every\n"
                "     relevant detail included, OR\n"
                "  B) 'I don't have information about [topic] in your knowledge base.'\n\n"
                "Do NOT use phrases like 'I found documents about X but they don't contain Y'.\n"
                "Do NOT reference conversation history as a document source."
            ),
            agent=agents.answer_agent,
            context=[depends_on],
        )
