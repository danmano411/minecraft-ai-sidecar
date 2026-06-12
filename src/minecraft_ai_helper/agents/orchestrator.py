"""
Orchestrator — fans out to selected agents, ranks results by source authority,
and synthesises the final full answer + 1-2 line HUD version.

Source authority weights:  official_docs (1.0) > wiki_rag (0.8) > community (0.6)
"""

import asyncio
import json

from minecraft_ai_helper import llm
from minecraft_ai_helper.server.models import (
    AgentName,
    AgentResult,
    IntentResult,
    QueryResponse,
    Source,
)
from . import community, official_docs, wiki_rag

_AUTHORITY: dict[AgentName, float] = {
    "official_docs": 1.0,
    "wiki_rag": 0.8,
    "community": 0.6,
}

_SYSTEM = """\
You are the final synthesiser for a Minecraft AI assistant. You receive answers
from multiple knowledge agents and must produce a single, cohesive response.

Rules:
1. Prefer information from higher-authority sources (official docs > wiki > community).
2. Reconcile conflicts by citing the authoritative source explicitly.
3. Write the full answer in clear prose — detailed enough to actually help.
4. Write the HUD answer as 1–2 tight sentences (max ~120 chars) for an in-game
   text overlay. No formatting, no bullet points in the HUD answer.
5. Generate up to 3 natural follow-up questions the player might ask next.

Respond with ONLY valid JSON — no prose, no markdown fences:
{"full_answer": "...", "hud_answer": "...", "follow_up_hints": ["...", "...", "..."]}
"""


async def _dispatch(intent_result: IntentResult, question: str) -> list[AgentResult]:
    tasks: list[asyncio.Task] = []
    for agent_name in intent_result.agents_to_invoke:
        if agent_name == "wiki_rag":
            tasks.append(asyncio.create_task(
                wiki_rag.run(question, intent_result.search_query), name="wiki_rag"
            ))
        elif agent_name == "official_docs":
            tasks.append(asyncio.create_task(
                official_docs.run(question, intent_result.intent, intent_result.search_query),
                name="official_docs",
            ))
        elif agent_name == "community":
            tasks.append(asyncio.create_task(
                community.run(question, intent_result.search_query), name="community"
            ))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, AgentResult)]


def _build_context(results: list[AgentResult]) -> str:
    parts: list[str] = []
    for r in results:
        if r.skipped or not r.answer.strip():
            continue
        weight = _AUTHORITY.get(r.agent, 0.5)
        parts.append(
            f"[{r.agent.upper()} | authority={weight:.1f} | confidence={r.confidence:.2f}]\n"
            f"{r.answer}"
        )
    return "\n\n---\n\n".join(parts)


async def run(question: str, intent_result: IntentResult) -> QueryResponse:
    agent_results = await _dispatch(intent_result, question)
    usable = [r for r in agent_results if not r.skipped and r.answer.strip()]

    if not usable:
        return QueryResponse(
            full_answer="Sorry, I couldn't find relevant information for that question.",
            hud_answer="No info found. Try rephrasing your question.",
            follow_up_hints=[],
            intent=intent_result.intent,
            sources=[],
        )

    usable.sort(key=lambda r: _AUTHORITY.get(r.agent, 0.5) * r.confidence, reverse=True)

    context = _build_context(usable)
    raw = await llm.complete(
        system=_SYSTEM,
        user=f"Player question: {question}\n\nAgent answers (highest authority first):\n{context}",
        max_tokens=2048,
        use_thinking=True,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"full_answer": raw, "hud_answer": raw[:120], "follow_up_hints": []}

    all_sources: list[Source] = []
    seen: set[str] = set()
    for r in usable:
        for s in r.sources:
            key = s.url or s.title
            if key not in seen:
                seen.add(key)
                all_sources.append(s)

    return QueryResponse(
        full_answer=data.get("full_answer", ""),
        hud_answer=data.get("hud_answer", "")[:200],
        follow_up_hints=data.get("follow_up_hints", [])[:3],
        intent=intent_result.intent,
        sources=all_sources,
    )
