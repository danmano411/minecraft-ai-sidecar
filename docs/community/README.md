# Community Agent

**Source file:** [`src/minecraft_ai_helper/agents/community.py`](../../src/minecraft_ai_helper/agents/community.py)

## Purpose

The community agent uses **Tavily web search** to pull in real-time content from Reddit, fan wikis, YouTube guides, and patch notes. It handles the things the local wiki snapshot can't: recent updates, current meta strategies, tips from experienced players, and newly discovered mechanics.

It is invoked for intents where player experience and recency matter: `combat`, `farming`, `building`, and any question containing words like "best", "meta", "efficient", "tips", or "strategy".

## Input / Output

**Input:** player question + refined search query

**Output:** `AgentResult`
```python
{
  "agent": "community",
  "answer": "The most efficient gold farm in 1.21 involves...",
  "sources": [{"title": "r/Minecraft — Best gold farm 2024", "url": "...", ...}],
  "confidence": 0.6,
  "skipped": False
}
```

Returns `skipped=True` if `TAVILY_API_KEY` is not set, or if the search returns no results.

## Search Behaviour

Queries Tavily with `search_depth="advanced"` and `max_results=5`, restricted to these domains:

- `reddit.com`
- `minecraft.fandom.com`
- `minecraftforum.net`
- `youtube.com`
- `minecraft.net`
- `bugs.mojang.com`

The search query is prefixed with `"Minecraft "` to keep results on-topic.

## How It Works

```
search_query
     │
     ▼
AsyncTavilyClient.search()   ← Tavily API (web search)
     │
     ▼
Extract title + url + content snippet (first 800 chars per result)
     │
     ▼
llm.complete()               ← synthesis with use_thinking=True
     │
     ▼
AgentResult (confidence hardcoded at 0.6 — community source, lower trust)
```

## Authority and Trust

The orchestrator weights this agent at **authority 0.6** — the lowest of the three agents. Community content is valuable for recency and strategy, but is more likely to contain outdated, version-specific, or opinionated information.

The LLM system prompt instructs the agent to note when advice is version-specific or likely to change between patches.

## Optional Agent

This agent is entirely optional — if `TAVILY_API_KEY` is not set in `.env`, it silently skips and the orchestrator proceeds with the results from the other agents.
