"""
Pipeline integration tests — wiki_rag wired through the full orchestrator.

Scope
-----
Complements tests/agents/test_wiki_rag.py (which calls the wiki_rag agent
directly) by testing the agent when given proper orchestrator inputs:
    IntentResult (wiki_rag forced) → orchestrator.run() → QueryResponse

Verifies that wiki_rag produces correct QueryResponse output (full_answer,
hud_answer, sources) when driven through the synthesis pipeline, not just
that the agent returns an AgentResult on its own.

The official_docs and community agents are NOT exercised here — both require
a TAVILY_API_KEY and will be covered in a separate test file once implemented.

Bypasses the LLM intent classifier (tested separately) by constructing
IntentResult directly with agents_to_invoke=["wiki_rag"].

Requires the full knowledge base: minecraft-ai build

Progress is visible in real-time via log_cli=true in pyproject.toml.
A markdown results table is logged at the end of the suite run.
"""

import logging
import time
from typing import cast

import pytest

from minecraft_ai_helper.server.models import AgentName, IntentResult, IntentType, QueryResponse

log = logging.getLogger(__name__)

# ── query corpus — one per embedded category ─────────────────────────────────

_QUERIES: list[dict] = [
    {
        "category": "Items",
        "question": "What does a Blaze Rod do in Minecraft?",
        "search_query": "blaze rod use fuel brewing",
        "intent": "general",
    },
    {
        "category": "Blocks",
        "question": "What is obsidian and how do I obtain it?",
        "search_query": "obsidian block formation water lava mining",
        "intent": "general",
    },
    {
        "category": "Mobs",
        "question": "What does an Enderman drop when killed?",
        "search_query": "enderman drops ender pearl",
        "intent": "combat",
    },
    {
        "category": "Mobs",
        "question": "How do I stop a Creeper from exploding?",
        "search_query": "creeper explosion prevent attack",
        "intent": "combat",
    },
    {
        "category": "Biomes",
        "question": "What is special about the mushroom fields biome?",
        "search_query": "mushroom island biome mooshroom mycelium",
        "intent": "biome",
    },
    {
        "category": "Enchantments",
        "question": "What does the Mending enchantment do?",
        "search_query": "mending enchantment repair xp orbs",
        "intent": "enchanting",
    },
    {
        "category": "Enchantments",
        "question": "What enchantments can be applied to a sword?",
        "search_query": "sharpness enchantment damage increase melee sword axe",
        "intent": "enchanting",
    },
    {
        "category": "Status Effects",
        "question": "What does the Regeneration status effect do?",
        "search_query": "regeneration effect health restore potion",
        "intent": "general",
    },
    {
        "category": "Game Mechanics",
        "question": "How does the hunger system work in Minecraft?",
        "search_query": "hunger bar food points saturation exhaustion",
        "intent": "mechanic",
    },
    {
        "category": "Crafting",
        "question": "How do I craft a bow in Minecraft?",
        "search_query": "bow weapon ranged crafting recipe",
        "intent": "crafting",
    },
    {
        "category": "Brewing",
        "question": "How do I brew a Potion of Healing?",
        "search_query": "potion healing brewing stand glistering melon",
        "intent": "general",
    },
    {
        "category": "Structures",
        "question": "Where can I find a Nether Fortress and what spawns there?",
        "search_query": "nether fortress location blaze wither skeleton spawn",
        "intent": "biome",
    },
    {
        "category": "Game Mechanics",
        "question": "How does experience and levelling work in Minecraft?",
        "search_query": "experience XP orbs levels enchanting",
        "intent": "mechanic",
    },
]


def _wiki_only_intent(q: dict) -> IntentResult:
    """Build an IntentResult that routes exclusively to wiki_rag."""
    return IntentResult(
        intent=cast(IntentType, q["intent"]),
        agents_to_invoke=cast(list[AgentName], ["wiki_rag"]),
        search_query=q["search_query"],
    )


# ── result collector (module-level, populated by parametrized test) ───────────

_RESULTS: list[dict] = []


# ── parametrized E2E test ─────────────────────────────────────────────────────

@pytest.mark.parametrize("q", _QUERIES, ids=[q["question"][:50] for q in _QUERIES])
async def test_pipeline_query(q: dict) -> None:
    """Full pipeline: retrieval + LLM synthesis → valid QueryResponse."""
    from minecraft_ai_helper.agents import orchestrator

    intent = _wiki_only_intent(q)
    log.info("→ [%s] %r", q["category"], q["question"])
    t0 = time.perf_counter()

    response: QueryResponse = await orchestrator.run(q["question"], intent)
    elapsed = time.perf_counter() - t0

    log.info(
        "← %.1fs | hud: %r",
        elapsed,
        response.hud_answer[:80],
    )

    # Collect for the summary table
    _RESULTS.append({
        "category": q["category"],
        "question": q["question"],
        "hud_answer": response.hud_answer,
        "full_answer": response.full_answer,
        "sources": [s.title for s in response.sources],
        "elapsed": elapsed,
    })

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(response, QueryResponse)
    assert len(response.full_answer.strip()) > 0, "full_answer must not be empty"
    assert len(response.hud_answer.strip()) > 0, "hud_answer must not be empty"
    assert response.intent in {
        "crafting", "combat", "redstone", "biome", "enchanting",
        "farming", "building", "mechanic", "lore", "general",
    }


# ── summary table printed after all parametrized cases finish ─────────────────

@pytest.fixture(scope="module", autouse=True)
def _print_results_table(request):
    """After all tests in this module complete, log a markdown results table."""
    yield
    if not _RESULTS:
        return

    log.info("\n\n## E2E Pipeline Results\n")
    log.info("| # | Category | Question | HUD Answer | Sources | Time |")
    log.info("|---|----------|----------|------------|---------|------|")
    for i, r in enumerate(_RESULTS, 1):
        q    = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        hud  = r["hud_answer"][:70] + ("…" if len(r["hud_answer"]) > 70 else "")
        srcs = ", ".join(r["sources"][:2]) if r["sources"] else "—"
        log.info("| %d | %s | %s | %s | %s | %.1fs |", i, r["category"], q, hud, srcs, r["elapsed"])
