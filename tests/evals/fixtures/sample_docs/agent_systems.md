# Multi-Agent Systems for LLM Applications

## Why More Than One Agent?

A single LLM agent can do a lot — answer questions, write code, call
tools. So why split work across multiple agents?

The answer is **focus**. Each agent in a multi-agent system gets a
narrower system prompt, a smaller tool registry, and a clearer goal.
Narrower prompts mean fewer mistakes; smaller tool registries mean
the agent picks the right tool more reliably; clearer goals mean
the agent stays on task.

This isn't unique to LLMs. The same logic drives microservices and
single-responsibility classes — a system of focused specialists tends
to outperform a system of one generalist.

## The Retrieve-Then-Synthesize Pattern

The most common multi-agent pattern for RAG is two agents in series:

1. **Retrieval Specialist.** Gets the user's question. Its only tool
   is the document search. Its goal: collect the most relevant
   passages. It does NOT try to answer; it just gathers evidence.
2. **Answer Specialist.** Gets the question + the retrieved
   passages. Its only tool is citation formatting. Its goal: write a
   grounded, cited answer. It does NOT search; it works from the
   evidence the retriever surfaced.

Splitting the work this way gives each agent a focused prompt and
forces an explicit handoff. A common failure mode in single-agent
RAG is that the agent skips retrieval and answers from memory; the
two-agent pattern makes that mistake structurally impossible.

## Costs and Trade-offs

Multi-agent systems aren't free:

* **More LLM calls.** Two agents means at least two model calls per
  user turn. For RAG, that's typically a small number of additional
  tokens (the retrieval agent's output is short), so the cost is
  rarely the blocker.
* **More orchestration code.** Frameworks like CrewAI, LangGraph, and
  AutoGen take this off your plate, but learning curve applies.
* **Harder to debug.** When the answer is wrong, you have to trace
  the failure back through two agents instead of one.

For most RAG applications under heavy traffic, the trade-off is worth
it. For prototype-scale work, a single agent with a careful prompt
is often enough.

## Agent Frameworks in 2026

* **CrewAI** — multi-agent native, role-based abstractions, small
  surface area, easy to learn. Used by this project.
* **LangGraph** — graph-of-agents, more flexible than CrewAI but
  steeper learning curve.
* **AutoGen** — conversational multi-agent, focuses on group chat
  patterns.

The right choice depends on how you think about your problem. If
your agents naturally fit "specialist roles", CrewAI is a good fit.
If they fit "nodes in a state machine", LangGraph is better.
