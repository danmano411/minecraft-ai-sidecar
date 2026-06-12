# Official Docs Agent

**Source file:** [`src/minecraft_ai_helper/agents/official_docs.py`](../../src/minecraft_ai_helper/agents/official_docs.py)

## Purpose

The official docs agent fetches **live, curated Minecraft Wiki technical pages** and synthesises an answer from them. It is invoked only for intents where authoritative technical accuracy matters most: `crafting`, `enchanting`, `mechanic`, `redstone`, and `combat`.

Unlike the Wiki RAG agent (which searches a static local snapshot), this agent makes a real HTTP request at query time, so it always reflects the current state of the wiki.

## Input / Output

**Input:** player question + intent + refined search query

**Output:** `AgentResult`
```python
{
  "agent": "official_docs",
  "answer": "Enchanting mechanics work as follows...",
  "sources": [{"title": "Enchanting mechanics", "url": "https://minecraft.wiki/w/Enchanting_mechanics", ...}],
  "confidence": 0.85,
  "skipped": False
}
```

Returns `skipped=True` if the intent has no mapped pages, or if all page fetches fail.

## Curated Page Map

Each intent maps to up to 2 authoritative wiki pages fetched at query time:

| Intent | Pages fetched |
|---|---|
| `crafting` | Crafting |
| `enchanting` | Enchanting mechanics, Enchanting |
| `redstone` | Redstone circuit, Redstone components |
| `mechanic` | Gameplay, Hunger |
| `combat` | Combat |

## How It Works

```
intent → look up page list
     │
     ▼
httpx.AsyncClient    ← GET minecraft.wiki/w/api.php?action=parse&page=...
     │
     ▼
BeautifulSoup        ← strip scripts/styles, extract plain text (first 6000 chars)
     │
     ▼
llm.complete()       ← synthesis with use_thinking=True
     │
     ▼
AgentResult (confidence hardcoded at 0.85 — official source, high trust)
```

## Extending the Page Map

To add coverage for a new intent, add an entry to `_TOPIC_PAGES` in `official_docs.py`:

```python
"farming": [
    ("Farming", "https://minecraft.wiki/w/Farming"),
    ("Crop farming", "https://minecraft.wiki/w/Crop_farming"),
],
```

The orchestrator weights this agent at **authority 1.0** — the highest of any agent — so its answers take priority when there is a conflict with other sources.
