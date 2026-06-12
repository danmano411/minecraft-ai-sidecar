"""
Shared pytest fixtures.
"""

import pytest
from openai import AsyncOpenAI
from minecraft_ai_helper.config import settings


@pytest.fixture
def ollama() -> AsyncOpenAI:
    """OpenAI-compatible client pointed at the local Ollama server."""
    return AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
