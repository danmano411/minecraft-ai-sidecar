"""
Wiki RAG agent — retrieves relevant chunks from ChromaDB and synthesizes an answer.

Two gates prevent hallucination:

  Gate 1 — Retrieval floor (pre-LLM):
    If the best chunk cosine similarity < rag_min_retrieval_score the query is
    off-topic for the current knowledge base. The LLM is never called; the
    result is returned as skipped=True with confidence=0.

  Gate 2 — LLM self-signal (post-LLM):
    The LLM is instructed to begin its response with "INSUFFICIENT DATA:" when
    the provided excerpts do not support a reliable answer. This is a simple
    prefix check — no JSON parsing — so it works reliably with small models.
    When triggered, confidence is halved and a disclaimer is prepended; the
    retrieved sources are still returned so the caller can inspect what was found.

Confidence is always derived from retrieval cosine similarity, not from LLM
self-report, to stay deterministic regardless of model size.
"""

import asyncio
import logging
import time

from minecraft_ai_helper import llm
from minecraft_ai_helper.config import settings
from minecraft_ai_helper.pipeline.ingestor import embed_query, query_collection
from minecraft_ai_helper.server.models import AgentResult, Source

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a Minecraft wiki assistant. Answer the player's question using ONLY the \
provided wiki excerpts. Do not invent facts or draw on knowledge outside the excerpts.

If the excerpts do not contain enough information to answer reliably, begin your \
response with exactly "INSUFFICIENT DATA:" followed by a brief explanation of what \
is missing.

Otherwise, answer concisely in plain prose suitable for a game HUD.
"""

_RETRIEVAL_FLOOR: float = settings.rag_min_retrieval_score


def _build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['page_title']} — {c['section_title']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


async def run(question: str, search_query: str) -> AgentResult:
    log.info("wiki_rag | embedding query: %r", search_query[:60])
    t_start = time.perf_counter()

    embedding = await asyncio.to_thread(embed_query, search_query)

    # Fetch a wider pool then cap per page to maximise source diversity.
    # e.g. for "sword enchantments" this surfaces Sword + Sharpness + Mending +
    # Unbreaking + Fire Aspect instead of 8 chunks all from the top-scoring page.
    _POOL = 24
    _MAX_PER_PAGE = 3
    raw_chunks = await asyncio.to_thread(query_collection, embedding, top_k=_POOL)
    page_counts: dict[str, int] = {}
    chunks: list[dict] = []
    for c in raw_chunks:
        pt = c["page_title"]
        if page_counts.get(pt, 0) < _MAX_PER_PAGE:
            chunks.append(c)
            page_counts[pt] = page_counts.get(pt, 0) + 1

    log.info("wiki_rag | embed+retrieve: %.1fs", time.perf_counter() - t_start)

    if not chunks:
        log.info("wiki_rag | no chunks returned — skipping")
        return AgentResult(
            agent="wiki_rag",
            answer="No relevant wiki pages found for this query.",
            confidence=0.0,
            skipped=True,
        )

    # ── Gate 1: retrieval floor ───────────────────────────────────────────────
    best_score = max(c["score"] for c in chunks)
    top_titles = ", ".join(dict.fromkeys(c["page_title"] for c in chunks[:6]))
    log.info(
        "wiki_rag | %d chunks — best=%.3f  top pages: %s",
        len(chunks), best_score, top_titles,
    )

    if best_score < _RETRIEVAL_FLOOR:
        log.info("wiki_rag | Gate 1 fired (%.3f < %.3f) — off-topic, skipping LLM", best_score, _RETRIEVAL_FLOOR)
        return AgentResult(
            agent="wiki_rag",
            answer=(
                f"This query does not appear to be covered by the current knowledge base "
                f"(best match score: {best_score:.2f}, "
                f"minimum required: {_RETRIEVAL_FLOOR:.2f})."
            ),
            confidence=0.0,
            skipped=True,
        )

    # Sources always come from retrieval — independent of what the LLM says.
    sources: list[Source] = [
        Source(
            title=c["page_title"],
            url=c["url"] or None,
            agent="wiki_rag",
            confidence=round(c["score"], 3),
        )
        for c in chunks[:3]
    ]
    retrieval_avg = sum(c["score"] for c in chunks[:3]) / max(len(chunks[:3]), 1)

    # ── LLM call ──────────────────────────────────────────────────────────────
    context = _build_context(chunks)
    user_msg = f"Question: {question}\n\nWiki excerpts:\n{context}"

    log.info("wiki_rag | calling LLM (%s, max_tokens=512)...", settings.llm_model)
    t_llm = time.perf_counter()

    answer = await llm.complete(
        system=_SYSTEM,
        user=user_msg,
        max_tokens=512,
        use_thinking=False,
    )

    # ── Gate 2: LLM self-signal ───────────────────────────────────────────────
    gate2 = "INSUFFICIENT DATA" if answer.strip().startswith("INSUFFICIENT DATA:") else "OK"
    log.info(
        "wiki_rag | LLM done in %.1fs — %d chars, gate2=%s, confidence=%.3f",
        time.perf_counter() - t_llm, len(answer), gate2, retrieval_avg,
    )

    if gate2 == "INSUFFICIENT DATA":
        return AgentResult(
            agent="wiki_rag",
            answer=f"[Low confidence — insufficient data] {answer.strip()}",
            sources=sources,
            confidence=round(retrieval_avg * 0.5, 3),
            skipped=False,
        )

    return AgentResult(
        agent="wiki_rag",
        answer=answer,
        sources=sources,
        confidence=round(retrieval_avg, 3),
    )
