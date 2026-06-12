from pydantic import BaseModel, Field
from typing import Literal


IntentType = Literal[
    "crafting", "combat", "redstone", "biome", "enchanting",
    "farming", "building", "mechanic", "lore", "general",
]

AgentName = Literal["wiki_rag", "official_docs", "community"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    game_version: str | None = Field(None, examples=["1.21.4"])


class Source(BaseModel):
    title: str
    url: str | None = None
    agent: AgentName
    confidence: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    full_answer: str
    hud_answer: str = Field(..., description="1–2 line summary for the in-game HUD")
    follow_up_hints: list[str] = Field(default_factory=list, max_length=3)
    intent: IntentType
    sources: list[Source] = Field(default_factory=list)


class AgentResult(BaseModel):
    agent: AgentName
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    skipped: bool = False


class IntentResult(BaseModel):
    intent: IntentType
    agents_to_invoke: list[AgentName]
    search_query: str
