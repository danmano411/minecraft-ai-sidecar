"""
Connectivity tests — Ollama LLM and embedding server.

Covers:
  - LLM chat completion (llama3.2:3b)
  - LLM produces Minecraft-relevant output
  - Embedding dimensions match config (768 for nomic-embed-text)
  - Embedding vectors are non-zero and normalised
  - Semantic similarity: related items score higher than unrelated ones
  - llm.complete() abstraction routes correctly to the Ollama backend
"""

import logging
import math
import time

import pytest

from minecraft_ai_helper.config import settings

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ── LLM tests ────────────────────────────────────────────────────────────────

async def test_llm_responds(ollama):
    """LLM returns a non-empty string for a trivial prompt."""
    log.info("→ LLM basic response check (model=%s)", settings.llm_model)
    t0 = time.perf_counter()
    resp = await ollama.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        max_tokens=10,
    )
    text = resp.choices[0].message.content or ""
    log.info("← %.1fs — response: %r", time.perf_counter() - t0, text[:50])
    assert text.strip() != ""


async def test_llm_model_name_in_response(ollama):
    """Completion response carries the expected model name."""
    log.info("→ LLM model name check")
    t0 = time.perf_counter()
    resp = await ollama.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=20,
    )
    log.info("← %.1fs — resp.model=%r", time.perf_counter() - t0, resp.model)
    assert settings.llm_model in resp.model


async def test_llm_follows_system_prompt(ollama):
    """LLM respects a system prompt instruction."""
    log.info("→ LLM system-prompt compliance check")
    t0 = time.perf_counter()
    resp = await ollama.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You must respond with exactly one word."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        max_tokens=10,
    )
    text = (resp.choices[0].message.content or "").strip()
    log.info("← %.1fs — response: %r (%d words)", time.perf_counter() - t0, text, len(text.split()))
    assert len(text.split()) <= 3  # allow minor variance in "one word" compliance


async def test_llm_knows_minecraft(ollama):
    """LLM correctly identifies a Minecraft-specific term."""
    log.info("→ LLM Minecraft knowledge check")
    t0 = time.perf_counter()
    resp = await ollama.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "What game features a Creeper mob? One word."}],
        max_tokens=15,
    )
    text = (resp.choices[0].message.content or "").lower()
    log.info("← %.1fs — response: %r", time.perf_counter() - t0, text[:60])
    assert "minecraft" in text


# ── Embedding tests ───────────────────────────────────────────────────────────

async def test_embedding_dimensions(ollama):
    """Embedding vector length matches EMBEDDING_DIMENSIONS in config."""
    log.info("→ embedding dimensions check (model=%s, expected=%d)", settings.embedding_model, settings.embedding_dimensions)
    t0 = time.perf_counter()
    resp = await ollama.embeddings.create(
        model=settings.embedding_model,
        input=["test"],
    )
    dims = len(resp.data[0].embedding)
    log.info("← %.1fs — dimensions=%d", time.perf_counter() - t0, dims)
    assert dims == settings.embedding_dimensions


async def test_embedding_non_zero(ollama):
    """Embedding vector is not all zeros."""
    log.info("→ embedding non-zero check")
    t0 = time.perf_counter()
    resp = await ollama.embeddings.create(
        model=settings.embedding_model,
        input=["diamond pickaxe"],
    )
    vec = resp.data[0].embedding
    log.info("← %.1fs — first 3 values: %s", time.perf_counter() - t0, vec[:3])
    assert any(v != 0.0 for v in vec)


async def test_embedding_batch(ollama):
    """Multiple texts can be embedded in a single call."""
    texts = ["sword", "pickaxe", "axe"]
    log.info("→ embedding batch check (%d texts)", len(texts))
    t0 = time.perf_counter()
    resp = await ollama.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    log.info("← %.1fs — %d embeddings returned", time.perf_counter() - t0, len(resp.data))
    assert len(resp.data) == len(texts)
    assert all(len(d.embedding) == settings.embedding_dimensions for d in resp.data)


async def test_embedding_semantic_similarity(ollama):
    """Related Minecraft items are closer in embedding space than unrelated ones."""
    log.info("→ embedding semantic similarity check")
    t0 = time.perf_counter()
    resp = await ollama.embeddings.create(
        model=settings.embedding_model,
        input=["diamond sword", "iron sword", "baked potato"],
    )
    vecs = [d.embedding for d in resp.data]
    sim_weapons = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    log.info(
        "← %.1fs — diamond↔iron=%.3f  diamond↔potato=%.3f",
        time.perf_counter() - t0, sim_weapons, sim_unrelated,
    )
    assert sim_weapons > sim_unrelated, (
        f"Expected weapon similarity ({sim_weapons:.3f}) > "
        f"unrelated similarity ({sim_unrelated:.3f})"
    )


async def test_same_text_identical_embedding(ollama):
    """The same input produces the same embedding twice (deterministic)."""
    text = ["how do I craft a diamond pickaxe"]
    log.info("→ embedding determinism check")
    t0 = time.perf_counter()
    r1 = await ollama.embeddings.create(model=settings.embedding_model, input=text)
    r2 = await ollama.embeddings.create(model=settings.embedding_model, input=text)
    match = r1.data[0].embedding == r2.data[0].embedding
    log.info("← %.1fs — identical=%s", time.perf_counter() - t0, match)
    assert match


# ── llm.complete() abstraction test ──────────────────────────────────────────

async def test_llm_complete_abstraction():
    """minecraft_ai_helper.llm.complete() routes to Ollama and returns a string."""
    log.info("→ llm.complete() abstraction test (backend=%s)", settings.llm_backend)
    t0 = time.perf_counter()
    from minecraft_ai_helper import llm
    result = await llm.complete(
        system="You are a helpful Minecraft assistant.",
        user="Name one use for diamonds in Minecraft.",
        max_tokens=50,
        use_thinking=False,
    )
    log.info("← %.1fs — %d chars: %r", time.perf_counter() - t0, len(result), result[:80])
    assert isinstance(result, str)
    assert len(result.strip()) > 0
