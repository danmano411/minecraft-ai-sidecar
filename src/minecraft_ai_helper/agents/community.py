"""
Community / real-time agent — uses Tavily to search Reddit, fan wikis,
YouTube guides, and patch notes for current meta and recent updates.
"""

from tavily import AsyncTavilyClient

from minecraft_ai_helper import llm
from minecraft_ai_helper.config import settings
from minecraft_ai_helper.server.models import AgentResult, Source

_SYSTEM = """\
You are a Minecraft community advisor. You have been given search results from
Reddit, fan wikis, YouTube guides, and patch notes.
Answer the question using only the provided search results.
Note if advice is version-specific or may change between patches.
Prioritise practical, actionable tips.
"""

_INCLUDE_DOMAINS = [
    "reddit.com",
    "minecraft.fandom.com",
    "minecraftforum.net",
    "youtube.com",
    "minecraft.net",
    "bugs.mojang.com",
]


async def run(question: str, search_query: str) -> AgentResult:
    if not settings.tavily_api_key:
        return AgentResult(
            agent="community",
            answer="Tavily API key not configured — community search skipped.",
            confidence=0.0,
            skipped=True,
        )

    tavily = AsyncTavilyClient(api_key=settings.tavily_api_key)
    try:
        results = await tavily.search(
            query=f"Minecraft {search_query}",
            search_depth="advanced",
            max_results=5,
            include_domains=_INCLUDE_DOMAINS,
        )
    except Exception as exc:
        return AgentResult(
            agent="community",
            answer=f"Web search unavailable: {exc}",
            confidence=0.0,
            skipped=True,
        )

    hits = results.get("results", [])
    if not hits:
        return AgentResult(agent="community", answer="", confidence=0.0, skipped=True)

    context_parts: list[str] = []
    sources: list[Source] = []
    for hit in hits:
        title = hit.get("title", "")
        url = hit.get("url", "")
        content = hit.get("content", "")
        if content:
            context_parts.append(f"[{title}] ({url})\n{content[:800]}")
            sources.append(Source(title=title, url=url, agent="community", confidence=0.6))

    context = "\n\n---\n\n".join(context_parts)
    answer = await llm.complete(
        system=_SYSTEM,
        user=f"Question: {question}\n\nCommunity search results:\n{context}",
        max_tokens=1024,
        use_thinking=True,
    )
    return AgentResult(agent="community", answer=answer, sources=sources, confidence=0.6)
