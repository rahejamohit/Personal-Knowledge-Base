# Personal Knowledge Agent — Frequently Asked Questions

## Setup & costs

**Q: Do I need an API key to run this locally?**

No. The default configuration uses Ollama for both chat and
embeddings, which runs entirely on your laptop. You only need API
keys if you flip `LLM_PROVIDER` or `EMBEDDING_PROVIDER` to `gemini`
or `openai` in your `.env` file.

**Q: How much does it cost to use the agent?**

In Phase 1 with the default Ollama configuration, the cost is $0.
Everything runs locally — no API calls, no cloud storage, no
managed services. You only pay if you switch to Gemini or OpenAI
for higher-quality responses, and even then the per-query cost is
typically under a tenth of a cent.

**Q: What do I need to install?**

Three things: Python 3.11+, `uv` (the Python package manager), and
Ollama. After that, `uv sync` pulls in everything else, and
`ollama pull mistral && ollama pull nomic-embed-text` downloads the
two local models.

## Data & privacy

**Q: Where are my documents stored?**

All of your documents and conversation history live on your local
disk — never sent to any external service in the default
configuration. The vector index lives in `.chroma/` (a Chroma-managed
directory of SQLite files) and your conversation history lives in
`.data/sessions.db`. Both are gitignored.

**Q: Will my data leave my machine?**

Only if you opt in. Switching to a cloud LLM (Gemini or OpenAI) sends
your queries and the retrieved document excerpts to that provider so
they can generate an answer. The embeddings provider also sees the
text it embeds. If you stay on Ollama, nothing leaves your laptop.

**Q: How do I delete my data?**

Two `rm -rf` commands: `rm -rf .chroma/` removes the document index,
`rm -rf .data/` removes conversation history. Both directories are
recreated cleanly on next startup.

## Behavior

**Q: How many turns of conversation does the agent remember?**

By default, the agent keeps the last 10 turns in context verbatim.
Older turns will be summarized in Phase 2. You can tune the limit
with `PKA_HISTORY_TURNS` in your `.env` file.

**Q: Can the agent see documents I haven't ingested?**

No. The retrieval tool only searches the local Chroma index. If
the agent can't find what you're asking about, it should say so
rather than guess.

**Q: Why does the agent sometimes say "I don't know"?**

That's a feature, not a bug. The agent is explicitly prompted to
admit uncertainty rather than confabulate. If you're seeing
"I don't know" on questions you think *are* covered by your docs,
your retrieval is probably underperforming — check the retrieved
docs in the agent's debug output.
