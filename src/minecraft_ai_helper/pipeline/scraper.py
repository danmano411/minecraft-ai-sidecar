"""
MediaWiki scraper for minecraft.wiki.

Fetches pages by category, parses their HTML into RawPage objects.
Respects the configured per-request delay so we don't hammer the wiki.

Two-phase design lets callers show accurate progress bars:
  1. collect_all_titles()  — fast API queries only, no HTML downloads
  2. fetch_pages()         — downloads HTML for a pre-filtered title list
"""

import asyncio
from dataclasses import dataclass
from typing import Callable

import httpx

from minecraft_ai_helper.config import settings

SCRAPE_CATEGORIES: list[str] = [
    "Items",
    "Blocks",
    "Mobs",
    "Biomes",
    "Enchantments",
    "Status_effects",
    "Game_mechanics",
    "Crafting",
    "Brewing",
    "Structures",
]


@dataclass
class RawPage:
    title: str
    url: str
    html: str


async def _fetch_json(client: httpx.AsyncClient, params: dict) -> dict:
    resp = await client.get(settings.wiki_api_url, params={**params, "format": "json"})
    resp.raise_for_status()
    return resp.json()


async def _get_category_members(client: httpx.AsyncClient, category: str) -> list[str]:
    """Return all page titles in a wiki category (handles continuation)."""
    titles: list[str] = []
    params: dict = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "cmtype": "page",
    }
    while True:
        data = await _fetch_json(client, params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        await asyncio.sleep(settings.wiki_request_delay)
    return titles


async def _fetch_page_html(client: httpx.AsyncClient, title: str) -> str | None:
    """Fetch rendered HTML for a single wiki page."""
    try:
        data = await _fetch_json(client, {
            "action": "parse",
            "page": title,
            "prop": "text",
            "disablelimitreport": "1",
            "disableeditsection": "1",
        })
        return data.get("parse", {}).get("text", {}).get("*")
    except httpx.HTTPStatusError:
        return None


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "MinecraftAIHelper/1.0 (educational RAG project)"},
        timeout=30.0,
    )


# ── Phase 1: fast title discovery ─────────────────────────────────────────────

async def collect_all_titles(
    categories: list[str] | None = None,
    on_category_done: Callable[[str, int], None] | None = None,
) -> dict[str, list[str]]:
    """
    Return {category_name: [page_titles]} for every requested category.
    Makes only lightweight API calls — no HTML is fetched yet.

    on_category_done(category, title_count) is called after each category.
    """
    cats = categories or SCRAPE_CATEGORIES
    result: dict[str, list[str]] = {}

    async with _make_client() as client:
        for cat in cats:
            try:
                titles = await _get_category_members(client, cat)
            except Exception:
                titles = []
            result[cat] = titles
            if on_category_done:
                on_category_done(cat, len(titles))

    return result


# ── Phase 2: HTML fetch ────────────────────────────────────────────────────────

async def fetch_pages(
    titles: list[str],
    on_page_done: Callable[[RawPage | None], None] | None = None,
) -> list[RawPage]:
    """
    Download HTML for each title in order.

    on_page_done(page_or_None) is called after every attempt — pass None if
    the fetch failed or the page had no content.  Use this to drive a progress
    bar in the caller.
    """
    pages: list[RawPage] = []

    async with _make_client() as client:
        for title in titles:
            await asyncio.sleep(settings.wiki_request_delay)
            html = await _fetch_page_html(client, title)
            if html:
                url = f"https://minecraft.wiki/w/{title.replace(' ', '_')}"
                page = RawPage(title=title, url=url, html=html)
                pages.append(page)
                if on_page_done:
                    on_page_done(page)
            else:
                if on_page_done:
                    on_page_done(None)

    return pages


# ── Legacy helper (used by tests and old callers) ─────────────────────────────

async def scrape_categories(
    categories: list[str] | None = None,
    max_pages: int | None = None,
    exclude_titles: set[str] | None = None,
) -> list[RawPage]:
    """
    Convenience wrapper: collect titles then fetch HTML in one call.

    exclude_titles: page titles to skip (already in the DB).
    max_pages: hard cap on pages fetched (useful for smoke tests).
    """
    cat_map = await collect_all_titles(categories)

    seen: set[str] = set()
    titles_to_fetch: list[str] = []
    for titles in cat_map.values():
        for t in titles:
            if t in seen:
                continue
            seen.add(t)
            if exclude_titles and t in exclude_titles:
                continue
            titles_to_fetch.append(t)
            if max_pages and len(titles_to_fetch) >= max_pages:
                break
        if max_pages and len(titles_to_fetch) >= max_pages:
            break

    return await fetch_pages(titles_to_fetch)
