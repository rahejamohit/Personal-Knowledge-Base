# Vector Embeddings, Explained

## What Is a Vector Embedding?

A vector embedding is a numerical representation of a piece of text
(or image, or audio) as a fixed-length list of floating-point numbers.
A typical sentence becomes something like a 768- or 1536-dimensional
vector — a point in a very high-dimensional space.

The key property of an embedding is that semantically related
inputs land near each other in this space, while unrelated inputs land
far apart. "The cat sat on the mat" and "A feline rested on the rug"
will have nearly identical embeddings even though they share almost no
words; "the cat sat on the mat" and "the stock market crashed" will
not.

Embeddings are produced by a neural network that's been trained on a
large corpus to make this property true. The network sees a string of
text and emits the vector; conceptually, it has internalized "what
this text is about" into the vector's coordinates.

## Why High Dimensions?

A single dimension can only encode one axis of variation — say, "is
this about animals?" To distinguish "cats" from "dogs" from "fish"
from "is this even about animals?", you need many dimensions, each
capturing a different concept. The embedding model's training process
discovers these dimensions automatically; humans rarely interpret what
any single dimension means.

Common sizes:

* **384-d** — small, fast models like `all-MiniLM-L6-v2`.
* **768-d** — mid-range; `nomic-embed-text` and many BERT-family
  models live here.
* **1536-d** — OpenAI's `text-embedding-3-small`.
* **3072-d** — OpenAI's `text-embedding-3-large`.

Higher dimensions generally mean better quality on hard retrieval
tasks, but cost more to store and search. For most personal-knowledge
applications, 768- or 1536-d is the sweet spot.

## Measuring Similarity

Once two pieces of text are embedded, you compare them with a
*distance metric*. The two most common are:

* **Cosine similarity** — the cosine of the angle between the two
  vectors. Ranges from `-1` (opposite) to `1` (identical direction).
  Doesn't care about vector magnitude, only direction; this is the
  right choice when your embeddings encode "what about" rather than
  "how much of".
* **Euclidean distance** — the straight-line distance between the two
  points. Useful when both magnitude and direction matter, which is
  rare for text embeddings.

Cosine similarity is the default for almost all production text-RAG
systems. Most vector databases compute it natively; Chroma's
`hnsw:space = "cosine"` setting is what configures this.

## The Embedding-Retrieval Loop

In a RAG system, you embed all your document chunks once (during
ingestion) and store the vectors in a vector database. At query time,
you embed the user's question with the *same model* — using a
different model would put the query in a different space and break
similarity scoring. The database then returns the chunks whose
embeddings are closest to the query embedding.

The crucial rule: **embedding consistency**. Once a corpus is indexed
with one embedding model, every subsequent query must use the same
model. Switching models means re-embedding the entire corpus.
