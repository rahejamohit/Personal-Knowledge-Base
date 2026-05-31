"""Phase 1 evaluation harness.

Each module in this package owns one piece of the RAG-quality
measurement pipeline:

* `test_cases` — curated `(query, expected docs, expected concepts)`
  triples spanning the categories the agent should handle.
* `metrics` — recall@k, precision@k, answer-content accuracy,
  citation validity.
* `report` — markdown rendering of a metrics dict, suitable for
  dropping into a PR description or weekly review.

Phase 1.7a wires these together into a runnable eval that:
1. Ingests the sample docs into a fresh Chroma index.
2. Runs the agent over each test case.
3. Computes metrics + writes the report.
"""
