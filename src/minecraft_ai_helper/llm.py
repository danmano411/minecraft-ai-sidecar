"""
Unified LLM client — routes calls to the backend set in LLM_BACKEND.

    anthropic  →  claude-opus-4-8 with adaptive thinking + streaming  (paid)
    ollama     →  any model served by a local Ollama instance           (free)
    groq       →  any model on Groq's free-tier OpenAI-compatible API  (free)

All three expose the same public coroutine:

    answer = await complete(system, user, max_tokens=1024, use_thinking=False)
"""

import anthropic
from openai import AsyncOpenAI

from minecraft_ai_helper.config import settings


async def complete(
    system: str,
    user: str,
    max_tokens: int = 1024,
    use_thinking: bool = False,
) -> str:
    backend = settings.llm_backend
    if backend == "anthropic":
        return await _anthropic_complete(system, user, max_tokens, use_thinking)
    if backend == "ollama":
        return await _openai_compat_complete(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            model=settings.llm_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
    if backend == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        return await _openai_compat_complete(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
    raise ValueError(
        f"Unknown LLM_BACKEND={backend!r}. Valid options: anthropic, ollama, groq"
    )


async def _anthropic_complete(
    system: str, user: str, max_tokens: int, use_thinking: bool
) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    kwargs: dict = {}
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    async with client.messages.stream(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    ) as stream:
        msg = await stream.get_final_message()
    return next((b.text for b in msg.content if b.type == "text"), "")


async def _openai_compat_complete(
    base_url: str, api_key: str, model: str,
    system: str, user: str, max_tokens: int,
) -> str:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
