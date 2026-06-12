"""
Official docs agent — fetches curated wiki technical pages and synthesises
authoritative answers for crafting, commands, and game mechanics.
"""

import httpx

from minecraft_ai_helper import llm
from minecraft_ai_helper.server.models import AgentResult, Source

_SYSTEM = """\
You are a Minecraft technical reference expert. You have been given content from
official Minecraft documentation or the Minecraft Wiki's technical pages.
Answer the question accurately using only the provided content.
Cite the version (Java/Bedrock, version number) when relevant.
Be concise but technically precise.
"""

_TOPIC_PAGES: dict[str, list[tuple[str, str]]] = {
    "crafting": [
        ("Crafting", "https://minecraft.wiki/w/Crafting"),
    ],
    "enchanting": [
        ("Enchanting mechanics", "https://minecraft.wiki/w/Enchanting_mechanics"),
        ("Enchanting", "https://minecraft.wiki/w/Enchanting"),
    ],
    "redstone": [
        ("Redstone circuit", "https://minecraft.wiki/w/Redstone_circuit"),
        ("Redstone components", "https://minecraft.wiki/w/Redstone_components"),
    ],
    "mechanic": [
        ("Gameplay", "https://minecraft.wiki/w/Gameplay"),
        ("Hunger", "https://minecraft.wiki/w/Hunger"),
    ],
    "combat": [
        ("Combat", "https://minecraft.wiki/w/Combat"),
    ],
}

_WIKI_API = "https://minecraft.wiki/api.php"


async def _fetch_wiki_text(client: httpx.AsyncClient, title: str) -> str:
    try:
        resp = await client.get(
            _WIKI_API,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "disablelimitreport": "1",
                "disableeditsection": "1",
                "format": "json",
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        html = data.get("parse", {}).get("text", {}).get("*", "")
        if not html:
            return ""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "sup", "span"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)[:6000]
    except Exception:
        return ""


async def run(question: str, intent: str, search_query: str) -> AgentResult:
    pages = _TOPIC_PAGES.get(intent, [])
    if not pages:
        return AgentResult(agent="official_docs", answer="", confidence=0.0, skipped=True)

    headers = {"User-Agent": "MinecraftAIHelper/1.0 (educational RAG project)"}
    async with httpx.AsyncClient(headers=headers) as client:
        fetched: list[tuple[str, str, str]] = []
        for title, url in pages[:2]:
            text = await _fetch_wiki_text(client, title)
            if text:
                fetched.append((title, url, text))

    if not fetched:
        return AgentResult(agent="official_docs", answer="", confidence=0.0, skipped=True)

    context = "\n\n---\n\n".join(f"[{t}]\n{txt}" for t, _, txt in fetched)
    answer = await llm.complete(
        system=_SYSTEM,
        user=f"Question: {question}\n\nOfficial documentation:\n{context}",
        max_tokens=1024,
        use_thinking=True,
    )

    sources = [
        Source(title=t, url=u, agent="official_docs", confidence=0.9)
        for t, u, _ in fetched
    ]
    return AgentResult(agent="official_docs", answer=answer, sources=sources, confidence=0.85)
