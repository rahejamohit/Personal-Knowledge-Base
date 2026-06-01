"""Markdown rendering of eval results.

Stub for Phase 1.7a. The function contract here is what the eval
harness will call once metrics + cases are wired up — keeping the
signature stable now means harness code can be written first and
the renderer filled in afterward.
"""

from __future__ import annotations

from typing import Any


def generate_eval_report(results: dict[str, Any]) -> str:
    """Render a markdown report from a dict of per-case eval results.

    Expected `results` shape (Phase 1.7a will document this in code):

    ```python
    {
        "cases": [
            {
                "query": "What is RAG?",
                "category": "technical",
                "difficulty": "easy",
                "recall_at_5": 1.0,
                "precision_at_5": 0.2,
                "answer_accuracy": 1.0,
                "citation_accuracy": 1.0,
                "retrieved_docs": ["rag_guide.md:p2", "faq.md:q1", ...],
                "answer": "RAG stands for ...",
                "errors": [],   # populated on crashes / timeouts
            },
            ...
        ],
        "config": {
            "llm_provider": "ollama",
            "embedding_provider": "ollama",
            "top_k": 5,
            "ran_at": "2026-06-01T12:34:56Z",
        },
    }
    ```

    Report structure (Phase 1.7a will produce):
    1. **Summary** — overall recall/precision/accuracy averages, grouped
       by category and difficulty. Render as a small markdown table.
    2. **Per-case breakdown** — one section per test case with the
       query, retrieved docs, answer, and the four metrics.
    3. **Failure analysis** — cases where any metric < 0.5, with the
       likely cause inferred from which metrics failed (low recall →
       retrieval problem; high recall + low accuracy → synthesis
       problem; low citation accuracy → hallucinated citations).
    4. **Recommendations** — actionable next steps (e.g. "bump
       `top_k`", "re-embed with a larger model", "tighten the answer
       agent's system prompt").

    Args:
        results: Output of the eval harness — structure as above.

    Returns:
        A fully-formatted markdown string suitable for dropping into a
        PR description or pasting into a weekly review doc.

    Raises:
        NotImplementedError: until Phase 1.7a.
    """
    raise NotImplementedError("Phase 1.7a implements report generation.")
