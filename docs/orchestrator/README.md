# Orchestrator Agent

**Source file:** [`src/minecraft_ai_helper/agents/orchestrator.py`](../../src/minecraft_ai_helper/agents/orchestrator.py)

## Purpose

The orchestrator is the **final stage of every query**. It receives the results from all invoked agents, ranks them by source authority, deduplicates overlapping sources, and synthesises a single cohesive response — including the full answer, the short HUD line, and follow-up question hints.

It is the only agent that sees all other agents' outputs at once.

## Input / Output

**Input:** player question + `IntentResult` (from intent classifier)

**Output:** `QueryResponse` — the final response returned to the client
```python
{
  "full_answer": "A diamond pickaxe is crafted by placing 3 diamonds...",
  "hud_answer": "3 diamonds (top row) + 2 sticks = diamond pickaxe.",
  "follow_up_hints": [
    "What can a diamond pickaxe mine?",
    "How do I enchant a pickaxe?",
    "Is netherite pickaxe better?"
  ],
  "intent": "crafting",
  "sources": [...]
}
```

## How It Works

```
IntentResult (intent + agents_to_invoke)
     │
     ▼
asyncio.gather()     ← fan out: wiki_rag, official_docs, community run concurrently
     │
     ▼
Filter skipped/empty results
Sort by: authority_weight × confidence  (descending)
     │
     ▼
Build context string with agent labels and weights
     │
     ▼
llm.complete()       ← final synthesis, use_thinking=True
     │
     ▼
Parse JSON → QueryResponse
Deduplicate sources across agents
```

## Source Authority Weights

| Agent | Authority weight |
|---|---|
| `official_docs` | 1.0 — highest; Mojang/wiki technical pages |
| `wiki_rag` | 0.8 — high; local snapshot of the same wiki |
| `community` | 0.6 — lower; Reddit/fan content, potentially outdated |

The effective ranking score is `authority × confidence`. A high-confidence wiki RAG result (0.8 × 0.95 = 0.76) can outrank a low-confidence official docs result (1.0 × 0.5 = 0.50).

## Synthesis Rules (LLM System Prompt)

The orchestrator instructs the LLM to:
1. Prefer higher-authority sources when information conflicts
2. Explicitly cite the authority source when reconciling a conflict
3. Write the full answer in clear prose
4. Write the HUD answer as 1–2 sentences, max ~120 chars, no bullet points
5. Generate up to 3 natural follow-up questions

## Fallback

If all agents are skipped or return empty answers, the orchestrator returns a static fallback response without calling the LLM:
```
"Sorry, I couldn't find relevant information for that question."
```
