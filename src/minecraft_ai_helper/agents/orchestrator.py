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
3. Write the full_answer in clear, complete prose. If the context contains a list
   (enchantments on a weapon, mob drops, brewing steps, biome features) cover ALL
   items — do not stop at the first or most prominent one and go into depth on only
   that. Be thorough: enumerate everything the context provides.
4. Write the hud_answer as 3–5 natural sentences (~300–400 chars). Translate any
   raw numbers or game values into plain English (e.g. "restores 2 hearts" not
   "heals 4 HP"). No bullet points, no markdown, no JSON — plain prose only.
   Do not start the hud_answer with a JSON brace.
5. Generate up to 3 natural follow-up questions the player might ask next.

CRITICAL — output format: your ENTIRE response must be one valid JSON object and
nothing else. No text before it, no text after it, no markdown fences.
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

    def _try_parse(s: str) -> dict:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}

    data = _try_parse(raw)
    if not data:
        # Model added preamble — extract the outermost {...} block
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            data = _try_parse(raw[start:end + 1])
    if not data:
        data = {"full_answer": raw, "hud_answer": raw[:500], "follow_up_hints": []}

    # If full_answer itself is a JSON string (model double-nested), unwrap it
    fa = data.get("full_answer", "")
    if isinstance(fa, str) and fa.strip().startswith("{"):
        inner = _try_parse(fa.strip())
        if inner.get("full_answer"):
            data = inner

    # If hud_answer is still missing or empty, derive from full_answer
    if not data.get("hud_answer", "").strip():
        data["hud_answer"] = data.get("full_answer", "")[:500]

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
        hud_answer=data.get("hud_answer", "")[:600],
        follow_up_hints=data.get("follow_up_hints", [])[:3],
        intent=intent_result.intent,
        sources=all_sources,
    )
