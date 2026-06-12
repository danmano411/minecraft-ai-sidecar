# Wiki RAG Agent

**Source file:** [`src/minecraft_ai_helper/agents/wiki_rag.py`](../../src/minecraft_ai_helper/agents/wiki_rag.py)

## Purpose

The Wiki RAG agent is the **primary factual source** — it is always invoked regardless of intent. It searches the local ChromaDB vector database (built from scraped minecraft.wiki pages) and synthesises an answer from the most relevant chunks.

Because it answers from a local database rather than making web requests, it is the fastest agent and the most reliable for established game content.

## Input / Output

**Input:** player question + refined search query (from intent classifier)

**Output:** `AgentResult`
```python
{
  "agent": "wiki_rag",
  "answer": "A diamond pickaxe requires 3 diamonds arranged in the top row...",
  "sources": [{"title": "Pickaxe", "url": "https://minecraft.wiki/w/Pickaxe", ...}],
  "confidence": 0.87,
  "skipped": False
}
```

## How It Works

```
search_query
     │
     ▼  (asyncio.to_thread — non-blocking)
embed_query()          ← OpenAI / Ollama embedding API
     │
     ▼
query_collection()     ← ChromaDB cosine similarity search, returns top-k chunks
     │
     ▼
Build context string   ← numbered list of [page — section] + text
     │
     ▼
llm.complete()         ← LLM synthesis with use_thinking=True
     │
     ▼
AgentResult
```

The `embed_query` and `query_collection` calls are wrapped in `asyncio.to_thread` because both are synchronous (synchronous OpenAI SDK + ChromaDB). This prevents blocking the event loop while other agents are running concurrently.

## Confidence Score

Computed as the average cosine similarity of the top-3 retrieved chunks. A score near 1.0 means the stored wiki content closely matched the query; near 0.0 means a poor match.

If ChromaDB is empty (knowledge base not built yet), the agent returns `skipped=True` with `confidence=0.0`.

## Limitations

- Only knows what was scraped at build time — will miss very recent additions to the wiki
- Quality is bounded by the coverage of the scraped categories
- Recipe tables survive chunking as pipe-separated text, which is readable but not structured
