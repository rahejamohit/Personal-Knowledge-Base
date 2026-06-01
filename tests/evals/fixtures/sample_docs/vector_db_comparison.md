# Choosing a Vector Database

The vector-database market in 2026 has settled into roughly four
players. Each has different trade-offs around cost, scale, and
operational overhead. This guide compares the four most common
choices for personal-knowledge and small-team RAG applications.

## Chroma

Chroma is an open-source, in-process vector database. It runs entirely
inside your Python process and persists to SQLite on disk. No server,
no network, no API keys.

* **Strengths:** Zero setup, $0 cost, fast for under ~1M vectors,
  ships embedded so dev/test/CI are trivial.
* **Weaknesses:** Single-process — can't be shared across multiple
  app servers without an external service in front of it. Indexing
  performance degrades past ~5M vectors. No managed scaling.
* **Best for:** Local development, single-user apps, side projects,
  RAG prototypes.

Chroma uses HNSW (Hierarchical Navigable Small World) for its
approximate-nearest-neighbor index, with cosine similarity as the
default metric.

## Pinecone

Pinecone is a managed cloud vector database. You hit a REST API; they
handle the infrastructure.

* **Strengths:** Scales to billions of vectors without operational
  work, generous free tier (1M vectors), strong metadata filtering,
  hybrid sparse-dense search built in.
* **Weaknesses:** Monthly cost grows with index size, locked into a
  cloud provider, network latency adds to every query (typically
  50–150 ms).
* **Best for:** Production multi-tenant SaaS, applications that need
  to scale without an ops team.

Pricing as of 2026: free tier covers 1M vectors at 1536-d; paid plans
start around $25/month for the starter index.

## Qdrant

Qdrant is open-source and can be self-hosted via Docker, or used as a
managed service. Sits between Chroma (no ops) and Pinecone (full ops).

* **Strengths:** Open source, strong filtering and payload support,
  Rust-based engine is fast and memory-efficient.
* **Weaknesses:** Requires operating a service if self-hosted,
  smaller community than the big two.
* **Best for:** Teams that want control without writing it themselves.

## Weaviate

Weaviate is the most full-featured of the four — it has built-in
modules for embedding, hybrid search, generative completions, and
multi-tenancy.

* **Strengths:** Batteries-included, GraphQL API, strong multi-modal
  support.
* **Weaknesses:** Steeper learning curve, heavier operational
  footprint than Qdrant.

## Migration Path

For a personal-knowledge agent: start with Chroma. The query API is
nearly identical across all four (top-k similarity search with
metadata filters), so switching later is mostly a ~20-line change in
the storage abstraction. The architecture doc recommends sticking with
Chroma through Phase 1 and revisiting in Phase 2 only if scale demands
it.
