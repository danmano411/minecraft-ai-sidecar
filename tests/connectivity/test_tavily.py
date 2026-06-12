"""
Connectivity tests — Tavily web search API.

All tests are automatically skipped when TAVILY_API_KEY is not set in .env —
just add a real key (starting with tvly-) and re-run to activate them.
"""

import logging
import time

import pytest

from minecraft_ai_helper.config import settings

log = logging.getLogger(__name__)

_key = settings.tavily_api_key
_key_missing = not _key or _key.startswith("your_") or "placeholder" in _key.lower()

pytestmark = pytest.mark.skipif(
    _key_missing,
    reason="TAVILY_API_KEY not configured — skipping Tavily tests",
)


async def test_tavily_basic_search():
    """Search returns a results list with at least one hit."""
    log.info("→ Tavily basic search")
    t0 = time.perf_counter()
    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    results = await client.search(
        query="Minecraft diamond pickaxe crafting recipe",
        search_depth="basic",
        max_results=3,
    )
    log.info("← %.1fs — %d results", time.perf_counter() - t0, len(results.get("results", [])))
    assert "results" in results
    assert len(results["results"]) > 0


async def test_tavily_result_has_required_fields():
    """Each result contains title, url, and content fields."""
    log.info("→ Tavily result field check")
    t0 = time.perf_counter()
    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    results = await client.search(
        query="Minecraft combat tips 1.21",
        search_depth="basic",
        max_results=2,
    )
    log.info("← %.1fs — checking %d results", time.perf_counter() - t0, len(results["results"]))
    for hit in results["results"]:
        assert "title" in hit, "Missing 'title' field"
        assert "url" in hit, "Missing 'url' field"
        assert "content" in hit, "Missing 'content' field"
        assert isinstance(hit["content"], str)
        assert len(hit["content"]) > 0


async def test_tavily_domain_filter():
    """Domain filtering restricts results to the specified sites."""
    log.info("→ Tavily domain filter check")
    t0 = time.perf_counter()
    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    results = await client.search(
        query="Minecraft best farming strategies",
        search_depth="basic",
        max_results=5,
        include_domains=["reddit.com", "minecraft.fandom.com"],
    )
    hits = results.get("results", [])
    log.info("← %.1fs — %d results", time.perf_counter() - t0, len(hits))
    for hit in hits:
        url: str = hit.get("url", "")
        assert "reddit.com" in url or "minecraft.fandom.com" in url or url == "", (
            f"Unexpected domain in result: {url}"
        )


async def test_tavily_community_agent_integration():
    """The community agent runs end-to-end without raising when key is set."""
    log.info("→ community agent integration test")
    t0 = time.perf_counter()
    from minecraft_ai_helper.agents.community import run
    result = await run(
        question="What is the best sword enchantment in Minecraft?",
        search_query="best sword enchantment Minecraft",
    )
    log.info("← %.1fs — skipped=%s, answer_len=%d", time.perf_counter() - t0, result.skipped, len(result.answer))
    assert not result.skipped
    assert isinstance(result.answer, str)
    assert len(result.answer.strip()) > 0
