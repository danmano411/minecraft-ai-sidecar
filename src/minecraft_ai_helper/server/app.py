"""
FastAPI sidecar server — listens on localhost:8765 for player queries.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from minecraft_ai_helper.agents import intent_classifier, orchestrator
from minecraft_ai_helper.config import settings
from minecraft_ai_helper.server.models import QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Minecraft AI Helper",
    version="1.0.0",
    description="RAG-powered Minecraft assistant sidecar",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    intent_result = await intent_classifier.classify_intent(request.question)
    response = await orchestrator.run(request.question, intent_result)
    return response


def serve() -> None:
    uvicorn.run(
        "minecraft_ai_helper.server.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
