"""
Minecraft Wiki page discovery — full category crawl.

Queries the minecraft.wiki MediaWiki API to collect every article title
across all gameplay-relevant categories. Output is written to:
  docs/wiki_coverage/wiki_pages_discovered.md   — all titles organised by category
  docs/wiki_coverage/database_comparison.md     — DB vs wiki gap analysis

Run from project root:
    uv run python docs/wiki_coverage/collect_wiki_pages.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import chromadb
import httpx

API = "https://minecraft.wiki/api.php"
HEADERS = {"User-Agent": "MinecraftAIHelper/1.0 (educational RAG project)"}
DELAY = 0.35  # seconds between requests — polite crawling

# ── Category map ──────────────────────────────────────────────────────────────
# Each entry: "Section label" -> ["Category_name", ...]
# Multiple categories per section are union-merged and deduplicated.
CATEGORY_MAP: dict[str, list[str]] = {
    "Mobs — Hostile":       ["Hostile_mobs"],
    "Mobs — Passive":       ["Passive_mobs"],
    "Mobs — Neutral":       ["Neutral_mobs"],
    "Mobs — Boss":          ["Boss_mobs"],
    "Mobs — Undead":        ["Undead_mobs"],
    "Mobs — Aquatic":       ["Aquatic_mobs"],
    "Mobs — Nether":        ["Nether_mobs"],
    "Mobs — End":           ["End_mobs"],
    "Blocks":               ["Blocks"],
    "Items":                ["Items"],
    "Combat Items":         ["Combat"],
    "Enchantments":         ["Enchantments"],
    "Status Effects":       ["Effects"],
    "Potions & Brewing":    ["Potions"],
    "Biomes — Overworld":   ["Overworld_biomes"],
    "Biomes — Nether":      ["Nether_biomes"],
    "Biomes — End":         ["End_biomes"],
    "Structures":           ["Generated_structures", "Illager_structures", "Villager_structures"],
    "Gameplay Mechanics":   ["Gameplay"],
    "Dimensions":           ["Dimensions"],
    "Redstone":             ["Redstone_mechanics"],
}

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _get(client: httpx.Client, params: dict) -> dict:
    resp = client.get(API, params={**params, "format": "json"}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_all_members(client: httpx.Client, category: str) -> list[str]:
    """Return every page title in a category, following continuation tokens."""
    titles: list[str] = []
    params: dict = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "cmtype": "page",
    }
    while True:
        data = _get(client, params)
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        time.sleep(DELAY)
    return titles


def category_exists(client: httpx.Client, category: str) -> bool:
    data = _get(client, {
        "action": "query", "prop": "categoryinfo",
        "titles": f"Category:{category}",
    })
    for p in data.get("query", {}).get("pages", {}).values():
        return "categoryinfo" in p
    return False


def is_article(title: str) -> bool:
    """Filter to main-namespace articles only (no User:, Template:, etc.)."""
    skip_prefixes = (
        "User:", "Talk:", "User talk:", "Template:", "File:", "MediaWiki:",
        "Help:", "Category:", "Module:", "Minecraft Wiki:",
    )
    if any(title.startswith(p) for p in skip_prefixes):
        return False
    # Skip sub-pages that are archive/history cuts (contain "/" but aren't biome subcats)
    if "/" in title and not any(keep in title for keep in ["Before ", "Java Edition", "Bedrock Edition"]):
        return False
    return True


# ── DB helper ─────────────────────────────────────────────────────────────────

def get_db_titles() -> set[str]:
    from minecraft_ai_helper.config import settings
    client = chromadb.PersistentClient(path=str(settings.chroma_path_resolved))
    try:
        collection = client.get_collection(settings.chroma_collection)
    except Exception:
        return set()
    if collection.count() == 0:
        return set()
    results = collection.get(include=["metadatas"])
    return {m["page_title"] for m in (results["metadatas"] or [])}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path(__file__).parent

    print("Collecting wiki pages...")
    section_titles: dict[str, list[str]] = {}
    all_titles: set[str] = set()

    with httpx.Client(headers=HEADERS) as client:
        for section, categories in CATEGORY_MAP.items():
            section_set: set[str] = set()
            for cat in categories:
                print(f"  [{section}] Category:{cat} ...", end=" ", flush=True)
                time.sleep(DELAY)
                if not category_exists(client, cat):
                    print("(not found)")
                    continue
                members = fetch_all_members(client, cat)
                filtered = [t for t in members if is_article(t)]
                section_set.update(filtered)
                print(f"{len(filtered)} pages")
                time.sleep(DELAY)
            section_titles[section] = sorted(section_set)
            all_titles.update(section_set)

    print(f"\nTotal unique wiki articles discovered: {len(all_titles)}")

    # ── Write wiki_pages_discovered.md ────────────────────────────────────────
    lines: list[str] = [
        "# Minecraft Wiki — Discovered Pages",
        "",
        "> Auto-generated by `docs/wiki_coverage/collect_wiki_pages.py`  ",
        "> Source: minecraft.wiki MediaWiki API  ",
        f"> Total unique articles: **{len(all_titles)}**",
        "",
        "---",
        "",
    ]

    for section, titles in section_titles.items():
        lines.append(f"## {section}  ({len(titles)} pages)")
        lines.append("")
        for t in titles:
            lines.append(f"- {t}")
        lines.append("")

    wiki_doc = out_dir / "wiki_pages_discovered.md"
    wiki_doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {wiki_doc}")

    # ── Cross-reference with ChromaDB ─────────────────────────────────────────
    print("\nLoading ChromaDB titles...")
    db_titles = get_db_titles()
    print(f"Pages in DB: {len(db_titles)}")

    in_both   = all_titles & db_titles
    missing   = all_titles - db_titles
    db_only   = db_titles - all_titles  # in DB but not in our category crawl

    # Organise missing by section
    missing_by_section: dict[str, list[str]] = {}
    for section, titles in section_titles.items():
        m = sorted(t for t in titles if t in missing)
        if m:
            missing_by_section[section] = m

    # ── Write database_comparison.md ──────────────────────────────────────────
    cmp_lines: list[str] = [
        "# Database vs Wiki — Coverage Gap Analysis",
        "",
        "> Auto-generated by `docs/wiki_coverage/collect_wiki_pages.py`",
        "",
        "## Summary",
        "",
        f"| | Count |",
        f"|---|---|",
        f"| Wiki articles discovered | {len(all_titles)} |",
        f"| Pages in ChromaDB | {len(db_titles)} |",
        f"| Already covered (in both) | {len(in_both)} |",
        f"| **Missing from DB** | **{len(missing)}** |",
        f"| In DB but not in wiki crawl | {len(db_only)} |",
        "",
        "---",
        "",
        "## Missing from Database (by section)",
        "",
    ]

    for section, titles in missing_by_section.items():
        cmp_lines.append(f"### {section}  ({len(titles)} missing)")
        cmp_lines.append("")
        for t in titles:
            cmp_lines.append(f"- {t}")
        cmp_lines.append("")

    cmp_lines += [
        "---",
        "",
        "## In DB but not in wiki category crawl",
        "",
        "> These may be valid pages from categories not covered by this crawl,",
        "> or stale/renamed pages.",
        "",
    ]
    for t in sorted(db_only)[:200]:
        cmp_lines.append(f"- {t}")
    if len(db_only) > 200:
        cmp_lines.append(f"- *(and {len(db_only)-200} more...)*")

    cmp_doc = out_dir / "database_comparison.md"
    cmp_doc.write_text("\n".join(cmp_lines), encoding="utf-8")
    print(f"Written: {cmp_doc}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Wiki articles discovered : {len(all_titles):>5}")
    print(f"Pages in ChromaDB        : {len(db_titles):>5}")
    print(f"Covered                  : {len(in_both):>5}")
    print(f"MISSING from DB          : {len(missing):>5}")
    print(f"DB-only (not in crawl)   : {len(db_only):>5}")
    print("="*60)
    print("\nTop missing sections:")
    for section, titles in sorted(missing_by_section.items(), key=lambda x: -len(x[1]))[:8]:
        print(f"  {len(titles):>4}  {section}")


if __name__ == "__main__":
    main()
