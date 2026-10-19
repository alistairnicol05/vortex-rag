"""FastAPI service for Cloud Run. Rebuilds an in-memory Chroma from data/index at startup (no API calls),
embeds only the incoming question, and has simple cost guards (length cap, global rate limit, token cap)."""
from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .embed_store import open_from_index
from .embeddings import GeminiEmbedder
from .llm import GeminiLLM
from .query import RagPipeline
from .retrieve import BM25

MAX_REQ_PER_MIN = int(os.getenv("MAX_REQ_PER_MIN", "20"))
_recent: deque[float] = deque()
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    store, chunks, meta = open_from_index(s.index_dir, s.collection)
    if meta["embed_model"] != s.embed_model:
        raise RuntimeError(f"Index built with {meta['embed_model']} but EMBED_MODEL={s.embed_model}")
    bm25 = BM25(chunks) if s.hybrid else None
    state["pipe"] = RagPipeline(s, GeminiEmbedder(s), store, GeminiLLM(s, max_output_tokens=600), bm25, trace_path=None)
    print(f"index loaded: {store.count()} chunks")  # stdout goes to Cloud Logging
    yield


app = FastAPI(title="Flow-acoustics RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    now = time.time()
    while _recent and now - _recent[0] > 60:
        _recent.popleft()
    if len(_recent) >= MAX_REQ_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit; try again in a minute.")
    _recent.append(now)
    a = state["pipe"].ask(req.question, k=req.k)
    print({k: a.trace[k] for k in ("request_id", "total_s", "input_tokens", "output_tokens", "refused", "cost_usd")})
    return {
        "answer": a.text,
        "refused": a.refused,
        "sources": [{"n": i, "source": h.source, "page": h.page} for i, h in enumerate(a.hits, start=1)],
        "cited": a.cited,
        "usage": {"input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "cost_usd": a.cost_usd},
        "latency_s": a.trace["total_s"],
    }
