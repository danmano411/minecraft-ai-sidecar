"""
Embeds chunks and stores them in ChromaDB.

Embedding backends:
    openai  →  text-embedding-3-small via OpenAI API  (paid)
    ollama  →  nomic-embed-text (or similar) via local Ollama  (free)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from openai import OpenAI

from minecraft_ai_helper.config import settings
from minecraft_ai_helper.pipeline.chunker import Chunk

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

EMBED_BATCH_SIZE = 100


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path_resolved))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _make_openai_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured embedding backend."""
    if settings.embedding_backend == "ollama":
        return OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=settings.openai_api_key)


def _embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


# ── public helpers ─────────────────────────────────────────────────────────────

def get_existing_page_titles() -> set[str]:
    """Return the set of page titles already stored in ChromaDB."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()
    results = collection.get(include=["metadatas"])
    return {m["page_title"] for m in (results["metadatas"] or [])}


def ingest_chunks(
    chunks: list[Chunk],
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> None:
    """
    Embed and upsert chunks into ChromaDB. Safe to re-run (upsert semantics).

    When progress and task_id are supplied the caller's rich Progress bar is
    advanced by EMBED_BATCH_SIZE after each batch, giving real-time feedback
    inside the build command's live display.  When omitted a simple fallback
    print is used instead.
    """
    if not chunks:
        if progress is None:
            print("No chunks to ingest.")
        return

    collection = _get_collection()
    embed_client = _make_openai_client()

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]
        embeddings = _embed_texts(embed_client, texts)

        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=[
                {
                    "page_title": c.page_title,
                    "section_title": c.section_title,
                    "url": c.url,
                }
                for c in batch
            ],
        )

        if progress is not None and task_id is not None:
            progress.advance(task_id, advance=len(batch))
        else:
            done = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
            print(f"  Embedded {done}/{len(chunks)} chunks", end="\r", flush=True)

    if progress is None:
        print()  # newline after \r progress


def query_collection(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    """Return top-k most similar chunks for a given embedding."""
    k = top_k or settings.rag_top_k
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    chunks_out: list[dict] = []
    docs = (results["documents"] or [[]])[0]
    metas = (results["metadatas"] or [[]])[0]
    dists = (results["distances"] or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        chunks_out.append({
            "text": doc,
            "page_title": meta.get("page_title", ""),
            "section_title": meta.get("section_title", ""),
            "url": meta.get("url", ""),
            "score": 1.0 - dist,
        })
    return chunks_out


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    embed_client = _make_openai_client()
    response = embed_client.embeddings.create(
        input=[query],
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding
