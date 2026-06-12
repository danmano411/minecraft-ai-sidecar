# Embeddings — Knowledge Pipeline

**Source files:**
- [`src/minecraft_ai_helper/pipeline/scraper.py`](../../src/minecraft_ai_helper/pipeline/scraper.py)
- [`src/minecraft_ai_helper/pipeline/chunker.py`](../../src/minecraft_ai_helper/pipeline/chunker.py)
- [`src/minecraft_ai_helper/pipeline/ingestor.py`](../../src/minecraft_ai_helper/pipeline/ingestor.py)

## Purpose

The knowledge pipeline is a **one-time setup step** that builds the local vector database the Wiki RAG agent searches at query time. It has three stages:

```
minecraft.wiki (MediaWiki API)
        │
        ▼
   scraper.py       — fetch page HTML per category
        │
        ▼
   chunker.py       — split each page into section-level chunks
        │
        ▼
   ingestor.py      — embed each chunk, upsert into ChromaDB
        │
        ▼
   ./data/chroma    — persistent local vector store
```

## Stage 1 — Scraper (`scraper.py`)

Hits `https://minecraft.wiki/w/api.php` using the MediaWiki API:
- `action=query&list=categorymembers` to list pages in a category
- `action=parse&page=TITLE&prop=text` to get rendered HTML per page

Default categories scraped:
`Items`, `Blocks`, `Mobs`, `Biomes`, `Enchantments`, `Status_effects`, `Game_mechanics`, `Crafting`, `Brewing`, `Structures`

Rate-limited by `WIKI_REQUEST_DELAY` (default 0.5 s) to avoid hammering the wiki.
Returns a list of `RawPage(title, url, html)` objects.

## Stage 2 — Chunker (`chunker.py`)

Splits rendered wiki HTML into section-level `Chunk` objects using BeautifulSoup:
- Splits at `<h2>` and `<h3>` boundaries
- Tables are converted to pipe-separated text (preserves crafting recipe data)
- Sections shorter than 80 chars are discarded
- Sections longer than 2000 chars are split into numbered sub-chunks

Each `Chunk` carries: `page_title`, `section_title`, `text`, `url`, `chunk_id`.

## Stage 3 — Ingestor (`ingestor.py`)

Embeds chunks in batches of 100 and upserts into ChromaDB:
- Uses the configured embedding backend (Ollama or OpenAI)
- Upsert semantics — safe to re-run without duplicating data
- Collection uses cosine similarity (`hnsw:space: cosine`)

Also exposes two query-time helpers used by the Wiki RAG agent:
- `embed_query(text)` — embed a single string for retrieval
- `query_collection(embedding, top_k)` — return the top-k most similar chunks

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `EMBEDDING_BACKEND` | `openai` | `ollama` for free local embeddings |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `nomic-embed-text` for Ollama |
| `EMBEDDING_DIMENSIONS` | `1536` | `768` for nomic-embed-text |
| `CHROMA_PATH` | `./data/chroma` | Where the vector DB is stored on disk |
| `WIKI_REQUEST_DELAY` | `0.5` | Seconds between wiki API requests |

> **Important:** if you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` after building, delete `./data/chroma/` and rebuild — the stored vectors and query vectors must have matching dimensions.
