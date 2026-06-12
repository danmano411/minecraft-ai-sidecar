from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── API keys (optional — only the backend you choose needs its key) ──────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    groq_api_key: str = ""

    # ── LLM backend ───────────────────────────────────────────────────────────
    llm_backend: Literal["anthropic", "ollama", "groq"] = "anthropic"
    llm_model: str = "claude-opus-4-8"
    ollama_base_url: str = "http://localhost:11434/v1"

    # ── Embedding backend ─────────────────────────────────────────────────────
    embedding_backend: Literal["openai", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "minecraft_wiki"

    # ── Wiki scraper ──────────────────────────────────────────────────────────
    wiki_api_url: str = "https://minecraft.wiki/api.php"
    wiki_request_delay: float = 0.5

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    rag_top_k: int = 8
    # Minimum cosine similarity for the best retrieved chunk. Queries whose
    # best match falls below this are considered off-topic and skipped before
    # the LLM is called.
    rag_min_retrieval_score: float = 0.30
    # Minimum confidence the LLM must self-report for an answer to be
    # considered credible. Below this a disclaimer is prepended and confidence
    # is capped at this value minus a small margin.
    rag_min_answer_confidence: float = 0.45

    @property
    def chroma_path_resolved(self) -> Path:
        return Path(self.chroma_path).resolve()


settings = Settings()
