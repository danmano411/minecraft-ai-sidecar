# Minecraft AI Helper

A RAG-powered Minecraft assistant that answers in-game questions using the official Minecraft Wiki. Built as a Python sidecar that a Fabric mod (work-in-progress) calls over HTTP to display answers in a game HUD overlay.

---

## Architecture

```
Player question
      │
      ▼
Intent Classifier  ──────────────────────────────────────────┐
  (LLM call)                                                 │
      │ intent + search_query + agents_to_invoke             │
      ▼                                                       │
Orchestrator  ─── fan-out ──────────────────────────────────┤
  │                                                          │
  ├─► wiki_rag agent        (ChromaDB cosine search → LLM)  │
  ├─► official_docs agent   (Tavily web search)              │
  └─► community agent       (Tavily community search)        │
                                                             │
      Results ranked by authority × confidence              │
      Final synthesis pass (LLM)                            │
      │                                                      │
      ▼                                                      │
 QueryResponse                                              │
   full_answer   (detailed prose)                           │
   hud_answer    (≤2 sentences, in-game overlay)            │
   follow_up_hints                                          │
   sources                                                  │
```

**Source authority weights:** `official_docs` (1.0) > `wiki_rag` (0.8) > `community` (0.6)

---

## Knowledge Base

The wiki_rag agent is backed by **26,830 document chunks** embedded from the official Minecraft Wiki across 10 categories:

| Category | Description |
|---|---|
| Items | Tools, food, materials |
| Blocks | All placeable blocks |
| Mobs | Creatures and enemies |
| Biomes | World environments |
| Enchantments | Armor and weapon enchants |
| Status_effects | Buffs and debuffs |
| Game_mechanics | Core game systems |
| Crafting | Crafting recipes |
| Brewing | Potion brewing |
| Structures | Generated structures |

Embeddings are stored locally in ChromaDB at `data/chroma/`. The build pipeline deduplicates automatically — re-running `minecraft-ai build` only fetches pages not already in the DB.

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) running locally (for local LLM + embedding backends)
  - `ollama pull llama3.2:3b` — default LLM
  - `ollama pull nomic-embed-text` — default embedding model

Or, configure cloud backends via `.env` (Anthropic, OpenAI, Groq).

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd "Minecraft AI Helper"
pip install -e ".[dev]"
```

---

## Configuration

Create a `.env` file in the project root. All fields are optional — defaults work with Ollama running locally.

```env
# LLM backend: "anthropic" | "ollama" | "groq"
LLM_BACKEND=ollama
LLM_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434/v1

# Embedding backend: "openai" | "ollama"
EMBEDDING_BACKEND=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768

# API keys (only needed for the backends you use)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
TAVILY_API_KEY=tvly-...
GROQ_API_KEY=

# ChromaDB storage location
CHROMA_PATH=./data/chroma
CHROMA_COLLECTION=minecraft_wiki

# FastAPI sidecar
HOST=127.0.0.1
PORT=8765

# RAG thresholds
RAG_TOP_K=8
RAG_MIN_RETRIEVAL_SCORE=0.30
RAG_MIN_ANSWER_CONFIDENCE=0.45
```

---

## CLI

### Build the knowledge base

```bash
# Full build — scrapes all 10 wiki categories (~1,287 pages, ~26,830 chunks)
minecraft-ai build

# Smoke test — only fetches 50 new pages
minecraft-ai build --test

# Specific categories only
minecraft-ai build --categories "Items,Crafting,Enchantments"
```

The build pipeline has four phases with a live progress bar:

1. **Collect** — fast API discovery of all page titles per category
2. **Fetch** — downloads HTML for new pages only (already-embedded pages are skipped)
3. **Chunk** — splits HTML into overlapping text chunks
4. **Embed** — embeds and upserts batches into ChromaDB

### Start the sidecar server

```bash
minecraft-ai serve
# Listening on http://127.0.0.1:8765
```

**Endpoints:**
- `GET  /health` — liveness check
- `POST /query`  — accepts `{"question": "..."}`, returns `QueryResponse`

### One-shot query

```bash
minecraft-ai query "How do I craft a diamond sword?"
minecraft-ai query "What does the Mending enchantment do?"
minecraft-ai query "Where do blazes spawn?"
```

---

## Project Structure

```
src/minecraft_ai_helper/
├── __main__.py              # CLI entry point (build / serve / query)
├── config.py                # Pydantic settings — reads .env
├── llm.py                   # LLM backend abstraction (Anthropic / Ollama / Groq)
│
├── pipeline/
│   ├── scraper.py           # MediaWiki scraper — two-phase: collect_all_titles + fetch_pages
│   ├── chunker.py           # HTML → Chunk objects (title, section, text, url)
│   └── ingestor.py          # Embed chunks + upsert into ChromaDB
│
├── agents/
│   ├── intent_classifier.py # Classifies question → intent + agents_to_invoke
│   ├── orchestrator.py      # Fan-out, rank by authority × confidence, synthesise
│   ├── wiki_rag.py          # ChromaDB cosine search + LLM answer with two anti-hallucination gates
│   ├── official_docs.py     # Tavily web search (official Minecraft docs)
│   └── community.py         # Tavily community search (Reddit, forums)
│
└── server/
    ├── app.py               # FastAPI app — /health and /query endpoints
    └── models.py            # Pydantic models: QueryRequest, QueryResponse, AgentResult, Source
```

---

## Anti-Hallucination Design (wiki_rag)

The wiki_rag agent uses two gates to avoid confabulation:

**Gate 1 — Retrieval floor (pre-LLM):**
If the best cosine similarity score across all retrieved chunks is below `RAG_MIN_RETRIEVAL_SCORE` (default 0.30), the query is considered off-topic. The LLM is never called; the result is returned as `skipped=True` with `confidence=0.0`.

**Gate 2 — LLM self-signal (post-LLM):**
The system prompt instructs the model to begin its response with `INSUFFICIENT DATA:` when the provided excerpts don't support a reliable answer. If this prefix is detected, `confidence` is halved and a disclaimer is prepended. Confidence is always derived from retrieval cosine similarity — never from LLM self-report — to stay deterministic regardless of model size.

---

## Running Tests

```bash
# All tests
pytest

# Connectivity tests only (API reachability, LLM ping)
pytest tests/connectivity/

# Agent tests only (wiki_rag end-to-end)
pytest tests/agents/

# With verbose live output
pytest -s tests/agents/test_wiki_rag.py
```

**Test layout:**

| Directory | What it tests |
|---|---|
| `tests/connectivity/` | wiki API reachability, Ollama LLM/embedding, Tavily API key |
| `tests/agents/` | wiki_rag agent end-to-end: on-topic answers, Gate 1, Gate 2, off-topic rejection |

Tavily tests auto-skip when `TAVILY_API_KEY` is not set. Some wiki_rag tests are marked `xfail(strict=False)` for behavior that requires a larger model than llama3.2:3b.

---

## Roadmap

- [ ] Fabric mod (Java, Minecraft 1.21.x) — renders HUD overlay, calls sidecar over HTTP
- [ ] Tavily API key — activates `official_docs` and `community` agents
- [ ] End-to-end pipeline test (`minecraft-ai serve` + live query)
- [ ] Upgrade LLM to a larger model (llama3.1:8b or Claude) for better extraction quality
- [ ] Streaming response endpoint for lower perceived latency
