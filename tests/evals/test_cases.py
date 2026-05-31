"""Curated test cases for Phase 1 RAG evaluation.

Twelve cases spanning four categories. The design choices:

* **Technical (5 cases)** — definitional queries with one canonical
  source doc. Sanity-checks retrieval can find the obvious answer.
* **FAQ (3 cases)** — Q&A-style questions whose answers live in a
  dedicated FAQ doc. Tests whether the retriever prefers FAQ-formatted
  text over wandering through prose.
* **Multi-doc (2 cases)** — queries whose full answer needs synthesis
  across two or more docs. Stresses recall@5.
* **Edge cases (2 cases)** — queries that should *fail gracefully*
  (no relevant docs, or ambiguous phrasing). Tests that the agent
  acknowledges uncertainty rather than confabulating.

Doc references use `<filename>` or `<filename>:<location>` syntax;
Phase 1.7a parses this into a match predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TestCase:
    """One evaluation case.

    Attributes:
        query: The user's question, exactly as it would be typed.
        relevant_docs: Filenames (and optional `:section`) of the
            documents that should appear in the retriever's top-k.
            Matched as substrings against `RetrievedDoc.source`.
        expected_answer_contains: Lowercased substrings that MUST appear
            in the agent's final answer. Case-insensitive substring
            matching, not full keyword recall.
        category: One of `"technical" | "faq" | "multi_doc" | "edge"`.
            Used to break out per-category metrics in the report.
        difficulty: `"easy" | "medium" | "hard"`. Drives the difficulty
            histogram in the report; does not affect pass/fail.
    """

    query: str
    relevant_docs: list[str]
    expected_answer_contains: list[str]
    category: str
    difficulty: str = "medium"
    notes: str = field(default="")


TEST_CASES: list[TestCase] = [
    # ─── Technical (5) ────────────────────────────────────────────────
    TestCase(
        query="What is RAG?",
        relevant_docs=["rag_guide.md"],
        expected_answer_contains=["retrieval", "augmented", "generation"],
        category="technical",
        difficulty="easy",
    ),
    TestCase(
        query="How do vector embeddings work?",
        relevant_docs=["embeddings_explained.md"],
        expected_answer_contains=["vector", "dimension", "similarity"],
        category="technical",
        difficulty="easy",
    ),
    TestCase(
        query="What is cosine similarity?",
        relevant_docs=["embeddings_explained.md", "vector_db_comparison.md"],
        expected_answer_contains=["cosine", "angle", "between"],
        category="technical",
        difficulty="medium",
    ),
    TestCase(
        query="What problem does RAG solve in LLMs?",
        relevant_docs=["rag_guide.md"],
        expected_answer_contains=["knowledge", "hallucination", "context"],
        category="technical",
        difficulty="medium",
    ),
    TestCase(
        query="Compare Chroma and Pinecone as vector databases",
        relevant_docs=["vector_db_comparison.md"],
        expected_answer_contains=["chroma", "pinecone"],
        category="technical",
        difficulty="hard",
        notes="Tests comparison synthesis across structured doc.",
    ),

    # ─── FAQ (3) ──────────────────────────────────────────────────────
    TestCase(
        query="Do I need an API key to run this locally?",
        relevant_docs=["faq.md"],
        expected_answer_contains=["no", "ollama", "local"],
        category="faq",
        difficulty="easy",
    ),
    TestCase(
        query="How much does it cost to use the agent?",
        relevant_docs=["faq.md"],
        expected_answer_contains=["free", "phase 1", "cost"],
        category="faq",
        difficulty="easy",
    ),
    TestCase(
        query="Where are my documents stored?",
        relevant_docs=["faq.md"],
        expected_answer_contains=["local", "chroma", "disk"],
        category="faq",
        difficulty="medium",
    ),

    # ─── Multi-doc synthesis (2) ──────────────────────────────────────
    TestCase(
        query="How do RAG systems use vector embeddings and which DB stores them?",
        relevant_docs=[
            "rag_guide.md",
            "embeddings_explained.md",
            "vector_db_comparison.md",
        ],
        expected_answer_contains=["embedding", "vector", "retrieval"],
        category="multi_doc",
        difficulty="hard",
    ),
    TestCase(
        query="Explain prompt engineering and how it relates to multi-agent systems",
        relevant_docs=["prompt_engineering.txt", "agent_systems.md"],
        expected_answer_contains=["prompt", "agent"],
        category="multi_doc",
        difficulty="hard",
    ),

    # ─── Edge cases (2) ───────────────────────────────────────────────
    TestCase(
        query="What is the weather today in Tokyo?",
        relevant_docs=[],
        expected_answer_contains=["don't know", "no", "documents"],
        category="edge",
        difficulty="medium",
        notes=(
            "Out-of-corpus question. The agent should explicitly say it "
            "can't answer from the indexed docs rather than confabulating."
        ),
    ),
    TestCase(
        query="thing",
        relevant_docs=[],
        expected_answer_contains=["clarify", "specific", "?"],
        category="edge",
        difficulty="hard",
        notes="Single-word, ambiguous query. Should request clarification.",
    ),
]


def cases_by_category() -> dict[str, list[TestCase]]:
    """Group `TEST_CASES` by `category` — used by `report.generate_eval_report`."""
    grouped: dict[str, list[TestCase]] = {}
    for case in TEST_CASES:
        grouped.setdefault(case.category, []).append(case)
    return grouped
