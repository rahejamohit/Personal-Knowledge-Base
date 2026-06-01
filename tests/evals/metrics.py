"""RAG quality metrics.

Phase 1.7a fills in the bodies — this module defines the function
contracts (signatures + docstrings) so the eval harness in
`tests/evals/report.py` can be written against a stable API now.

Four metrics, each returning a float in `[0.0, 1.0]`:

* `recall@k`    — "of the docs that SHOULD have been retrieved, what
                  fraction landed in the top-k?"
* `precision@k` — "of the docs that DID land in the top-k, what fraction
                  were relevant?"
* `answer_accuracy`   — "what fraction of expected concepts appear in
                        the answer?" (case-insensitive substring)
* `citation_accuracy` — "what fraction of citations point at a doc that
                        was actually retrieved?" (catches hallucinated
                        citations)

These are intentionally simple — Phase 2 can swap in BERT-score / LLM-
graded evals if the substring heuristic starts to under-rate good
answers.
"""

from __future__ import annotations


def compute_retrieval_recall_at_k(
    retrieved_docs: list[str],
    relevant_docs: list[str],
    k: int = 5,
) -> float:
    """Recall@k for retrieval.

    `recall@5 = |relevant ∩ top_5(retrieved)| / |relevant|`

    Args:
        retrieved_docs: Doc identifiers in the order the retriever
            returned them (e.g. `RetrievedDoc.source`). May be longer
            than `k`; we'll truncate.
        relevant_docs: Expected-to-be-relevant doc identifiers, as
            authored in the test case. Substring matching against
            `retrieved_docs` (so `"faq.md"` matches `"faq.md:q2"`).
        k: Number of top results to consider. Defaults to 5 because
            that's the agent's default `top_k`.

    Returns:
        Recall in `[0.0, 1.0]`. Returns `1.0` if `relevant_docs` is
        empty — no relevant docs means trivially "all" were retrieved.

    Raises:
        NotImplementedError: until Phase 1.7a.
        ValueError: if `k < 1`.
    """
    raise NotImplementedError("Phase 1.7a implements recall@k.")


def compute_retrieval_precision_at_k(
    retrieved_docs: list[str],
    relevant_docs: list[str],
    k: int = 5,
) -> float:
    """Precision@k for retrieval.

    `precision@5 = |relevant ∩ top_5(retrieved)| / min(k, |retrieved|)`

    Args:
        retrieved_docs: Same as `compute_retrieval_recall_at_k`.
        relevant_docs: Same as `compute_retrieval_recall_at_k`.
        k: Top-results window.

    Returns:
        Precision in `[0.0, 1.0]`. Returns `0.0` if the retriever
        returned nothing — there's no precision to measure on an
        empty list.

    Raises:
        NotImplementedError: until Phase 1.7a.
        ValueError: if `k < 1`.
    """
    raise NotImplementedError("Phase 1.7a implements precision@k.")


def compute_answer_accuracy(
    answer: str,
    expected_content: list[str],
) -> float:
    """Fraction of `expected_content` substrings that appear in `answer`.

    Case-insensitive substring matching. Deliberately permissive — we
    want to credit answers that hit the right concept even if the LLM
    paraphrases. Phase 2 can upgrade to embedding-similarity matching
    when the substring heuristic starts failing the right cases.

    Args:
        answer: The agent's full response text.
        expected_content: Lowercased substrings (concept tags) that
            should appear. Empty list → returns `1.0` (nothing to miss).

    Returns:
        Fraction matched, in `[0.0, 1.0]`.

    Raises:
        NotImplementedError: until Phase 1.7a.
    """
    raise NotImplementedError("Phase 1.7a implements answer accuracy.")


def compute_citation_accuracy(
    citations: list[str],
    retrieved_docs: list[str],
) -> float:
    """Fraction of citations that point at a doc the retriever surfaced.

    Catches the most common hallucination failure mode: the agent
    invents citations to docs that were never retrieved.

    Args:
        citations: Citation strings extracted from the answer (e.g.
            `"[source: faq.md]"`). Phase 1.7a parses these out of the
            agent response.
        retrieved_docs: Doc identifiers the retriever returned. A
            citation is "valid" if any retrieved doc identifier appears
            as a substring of the citation string.

    Returns:
        Fraction valid, in `[0.0, 1.0]`. Returns `1.0` if `citations`
        is empty (no citations → nothing to fail).

    Raises:
        NotImplementedError: until Phase 1.7a.
    """
    raise NotImplementedError("Phase 1.7a implements citation accuracy.")
