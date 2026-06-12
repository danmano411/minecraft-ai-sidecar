"""
Intent classifier — determines the query type and which agents to invoke.
Uses llm.complete() so it works with any configured backend.
"""

import json
from typing import cast

from minecraft_ai_helper import llm
from minecraft_ai_helper.server.models import AgentName, IntentResult, IntentType

_SYSTEM = """\
You are a Minecraft query classifier. Given a player's question, you must:
1. Classify the intent into exactly one category.
2. Decide which knowledge agents to invoke.
3. Produce a refined search query optimised for semantic retrieval.

Intent categories:
- crafting      : recipes, crafting tables, ingredients, anvil upgrades
- combat        : fighting mobs, weapons, armour, PvP tactics, boss strategies
- redstone      : circuits, contraptions, pistons, observers, hoppers
- biome         : biome locations, structures, generation, climate
- enchanting    : enchantment tables, books, anvils, enchant IDs/levels
- farming       : crop farms, animal breeding, mob farms, XP farms
- building      : building tips, material choices, structural blocks
- mechanic      : game systems (hunger, sleep, respawn, weather, time, XP)
- lore          : story, history, lore, Easter eggs, developer notes
- general       : anything that doesn't clearly fit the above

Agent routing rules:
- wiki_rag     : always include (primary factual source)
- official_docs: include for crafting, enchanting, mechanic, redstone
- community    : include for combat, farming, building, and questions with
                 "best", "strategy", "meta", "efficient", "farm", "tips"

Respond with ONLY valid JSON — no prose, no markdown fences:
{"intent": "<intent_type>", "agents_to_invoke": ["wiki_rag", ...], "search_query": "<refined query>"}
"""

_INTENT_VALUES: set[str] = {
    "crafting", "combat", "redstone", "biome", "enchanting",
    "farming", "building", "mechanic", "lore", "general",
}
_AGENT_VALUES: set[str] = {"wiki_rag", "official_docs", "community"}


async def classify_intent(question: str) -> IntentResult:
    raw = await llm.complete(
        system=_SYSTEM,
        user=question,
        max_tokens=256,
        use_thinking=False,
    )
    raw = raw.strip()

    # Strip markdown code fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return IntentResult(intent="general", agents_to_invoke=["wiki_rag"], search_query=question)

    intent: IntentType = data.get("intent", "general")
    if intent not in _INTENT_VALUES:
        intent = "general"

    raw_agents: list[str] = data.get("agents_to_invoke", ["wiki_rag"])
    agents: list[AgentName] = [cast(AgentName, a) for a in raw_agents if a in _AGENT_VALUES]
    if not agents:
        agents = cast(list[AgentName], ["wiki_rag"])

    return IntentResult(
        intent=intent,
        agents_to_invoke=agents,
        search_query=data.get("search_query") or question,
    )
