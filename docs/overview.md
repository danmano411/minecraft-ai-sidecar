# Minecraft AI Helper — Project Overview

## What This Project Does

A Minecraft AI assistant that answers player questions **natively in-game** without alt-tabbing. The player presses a keybind, types a question, and gets an answer rendered as a HUD overlay — powered by a local RAG pipeline and a multi-agent LLM backend.

## Architecture (3 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3 — Fabric Mod  (in-game, Java)       [deferred] │
│  Keybind → text input → HTTP → render HUD panel         │
└────────────────────────┬────────────────────────────────┘
                         │ POST /query  localhost:8765
┌────────────────────────▼────────────────────────────────┐
│  Layer 2 — Python Sidecar  (FastAPI server)             │
│                                                         │
│   Intent Classifier                                     │
│        │                                                │
│        ├──▶ Wiki RAG Agent   (ChromaDB)                 │
│        ├──▶ Official Docs Agent  (live wiki fetch)      │
│        └──▶ Community Agent  (Tavily web search)        │
│                        │                                │
│                   Orchestrator                          │
│            (rank, deduplicate, synthesise)              │
└────────────────────────┬────────────────────────────────┘
                         │ reads from
┌────────────────────────▼────────────────────────────────┐
│  Layer 1 — Knowledge Pipeline  (one-time setup)         │
│  Scrape wiki → chunk by section → embed → ChromaDB      │
└─────────────────────────────────────────────────────────┘
```

## Response Shape

Every query returns:

| Field | Description |
|---|---|
| `full_answer` | Detailed prose answer |
| `hud_answer` | 1–2 sentence summary for the in-game overlay |
| `follow_up_hints` | Up to 3 suggested follow-up questions |
| `intent` | Classified query type |
| `sources` | Which pages/sites were used |

## Stack

| Concern | Tool |
|---|---|
| LLM (agents) | Ollama `llama3.2:3b` (testing) / Claude `claude-opus-4-8` (production) |
| Embeddings | Ollama `nomic-embed-text` (testing) / OpenAI `text-embedding-3-small` (production) |
| Vector store | ChromaDB (local persistent, `./data/chroma`) |
| Web search | Tavily |
| Server | FastAPI + uvicorn on `localhost:8765` |
| Package manager | uv (Python 3.12+) |
| Game mod | Fabric (Java, Minecraft 1.21.x) — deferred |

## CLI Commands

```bash
minecraft-ai build          # one-time: scrape wiki, embed, store
minecraft-ai build --test   # same but capped at 50 pages
minecraft-ai serve          # start the sidecar on localhost:8765
minecraft-ai query "..."    # test a single question end-to-end
```

## Docs Index

- [embeddings/](embeddings/) — knowledge pipeline: scraping, chunking, embedding, ChromaDB
- [intent_classifier/](intent_classifier/) — query classification and agent routing
- [wiki_rag/](wiki_rag/) — local RAG retrieval over the embedded wiki
- [official_docs/](official_docs/) — live fetch of curated technical wiki pages
- [community/](community/) — Tavily web search for real-time community content
- [orchestrator/](orchestrator/) — result ranking, deduplication, final synthesis
