"""
Connectivity tests — Minecraft Wiki MediaWiki API.

Verifies that the API endpoint is reachable and returns well-formed responses
for the operations the scraper and official_docs agent depend on.
"""

import logging

import pytest
import httpx

from minecraft_ai_helper.config import settings

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "MinecraftAIHelper/1.0 (test suite)"}


@pytest.fixture
async def wiki_client():
    async with httpx.AsyncClient(
        headers=HEADERS, follow_redirects=True, timeout=15
    ) as client:
        yield client


async def test_api_reachable(wiki_client):
    """API endpoint responds with HTTP 200."""
    log.info("→ wiki API reachability check (%s)", settings.wiki_api_url)
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={"action": "query", "meta": "siteinfo", "format": "json"},
    )
    log.info("← HTTP %d", r.status_code)
    assert r.status_code == 200


async def test_siteinfo_correct_wiki(wiki_client):
    """Siteinfo confirms we are hitting the Minecraft Wiki, not some other MediaWiki."""
    log.info("→ siteinfo check")
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={"action": "query", "meta": "siteinfo", "siprop": "general", "format": "json"},
    )
    info = r.json()["query"]["general"]
    log.info("← sitename=%r  generator=%r", info["sitename"], info["generator"])
    assert info["sitename"] == "Minecraft Wiki"
    assert "MediaWiki" in info["generator"]


async def test_category_members_items(wiki_client):
    """Category:Items returns at least one page title."""
    log.info("→ category members — Items")
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Items",
            "cmlimit": "5",
            "cmtype": "page",
            "format": "json",
        },
    )
    members = r.json()["query"]["categorymembers"]
    log.info("← %d members returned", len(members))
    assert r.status_code == 200
    assert len(members) > 0
    assert all("title" in m for m in members)


async def test_category_members_blocks(wiki_client):
    """Category:Blocks returns pages (used by build pipeline)."""
    log.info("→ category members — Blocks")
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Blocks",
            "cmlimit": "5",
            "cmtype": "page",
            "format": "json",
        },
    )
    members = r.json()["query"]["categorymembers"]
    log.info("← %d members returned", len(members))
    assert r.status_code == 200
    assert len(members) > 0


async def test_page_parse_returns_html(wiki_client):
    """action=parse for the Diamond page returns the correct article with non-empty HTML."""
    log.info("→ page parse — Diamond")
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={
            "action": "parse",
            "page": "Diamond",
            "prop": "text",
            "disablelimitreport": "1",
            "format": "json",
        },
    )
    assert r.status_code == 200
    data = r.json()["parse"]
    log.info("← title=%r  html_len=%d", data["title"], len(data["text"]["*"]))
    assert data["title"] == "Diamond"
    html = data["text"]["*"]
    assert len(html) > 5_000
    assert "diamond" in html.lower()


async def test_page_parse_missing_page(wiki_client):
    """Parsing a non-existent page returns an error key, not a 500."""
    log.info("→ page parse — non-existent page")
    r = await wiki_client.get(
        settings.wiki_api_url,
        params={
            "action": "parse",
            "page": "ThisPageDefinitelyDoesNotExist_XYZ_999",
            "prop": "text",
            "format": "json",
        },
    )
    data = r.json()
    log.info("← HTTP %d, has_error=%s", r.status_code, "error" in data)
    assert r.status_code == 200
    assert "error" in data or "parse" not in data
