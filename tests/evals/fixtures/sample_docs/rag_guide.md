# A Technical Guide to Retrieval-Augmented Generation (RAG)

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines two
distinct components: a *retrieval system* that fetches relevant
documents from a knowledge base, and a *generative language model*
that produces a final answer conditioned on those documents. The
retrieval step grounds the LLM's response in real source material,
turning a generic chat model into a domain expert without the cost of
fine-tuning.

The technique was formalized in the 2020 paper "Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. Since
then, RAG has become the default pattern for production LLM
applications that need to answer questions about evolving or
proprietary information.

## What Problem Does RAG Solve?

Pure generative models have three well-known weaknesses when used as
question-answering systems:

1. **Knowledge cutoff.** Once a model is trained, it doesn't know
   about anything that happened after its training data was frozen.
   For a 2024-trained model, asking "what happened last week?" is
   guaranteed to fail.

2. **Hallucination.** When a model doesn't know an answer, it tends
   to invent one that sounds plausible. The model has no internal
   signal for "I don't know" — it generates the most likely
   continuation of the prompt.

3. **Lack of provenance.** Even when the answer is correct, the user
   has no way to verify it. There's no "source" to click.

RAG addresses all three by inserting a retrieval step between the
question and the answer. The retriever surfaces a handful of relevant
passages from a curated corpus; the generator is then prompted to
answer *using only those passages* and to cite them. The result is an
answer that is fresh (the corpus can be updated independently of the
model), grounded (the model has actual reference text to draw from),
and verifiable (citations point at the exact source).

## How RAG Works (End to End)

A canonical RAG pipeline has five stages:

1. **Ingestion.** Documents are read, split into chunks of a few
   hundred tokens each, and embedded into vectors using an embedding
   model.

2. **Indexing.** The vectors are stored in a vector database keyed by
   chunk ID, with the source text and metadata kept alongside.

3. **Query.** When a user asks a question, the query is embedded with
   the same model, and the vector database returns the top-K most
   similar chunks (typically 3–10).

4. **Augmentation.** The retrieved chunks are inserted into the LLM's
   prompt as context, along with the original query.

5. **Generation.** The LLM generates an answer using the retrieved
   context, ideally citing each chunk it draws from.

The art of building a good RAG system lies almost entirely in stages
1–3: getting the chunking, embeddings, and retrieval right. Stage 5
is mostly prompt engineering once the retriever is solid.

## Common Failure Modes

* **Retrieval miss.** The right chunk exists in the index but the
  retriever ranks irrelevant chunks above it. Fixes: better embeddings,
  reranking, query expansion.
* **Chunk boundaries.** The answer spans two chunks, and neither is
  individually high-scoring. Fixes: larger chunks, overlap between
  adjacent chunks.
* **Stale corpus.** The retriever finds an old doc that's been
  superseded. Fixes: timestamped metadata, recency boosting.
