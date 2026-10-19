"""Stretch goal: the retriever as a tool for a tiny Google ADK agent.

Run (in a SEPARATE venv, see the guide):
    $env:PYTHONPATH = "src"
    adk run vortex_agent          # terminal chat
    adk web --port 8000           # dev UI (development only)
Needs GOOGLE_API_KEY in vortex_agent/.env (same key as GEMINI_API_KEY).
"""
from __future__ import annotations

from google.adk.agents.llm_agent import Agent

from rag.config import get_settings
from rag.embed_store import open_persistent
from rag.embeddings import GeminiEmbedder
from rag.query import RagPipeline
from rag.llm import GeminiLLM

_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        s = get_settings()
        _pipeline = RagPipeline(s, GeminiEmbedder(s), open_persistent(s), GeminiLLM(s))
    return _pipeline


def search_papers(query: str, k: int = 5) -> dict:
    """Search the indexed flow-acoustics papers and return the most relevant passages.

    Args:
        query: A self-contained search query (rephrase the user's question to include key terms).
        k: Number of passages to return (1-8).

    Returns:
        A dict with status and a list of passages, each with source file, page and text.
    """
    k = max(1, min(int(k), 8))
    hits = _get_pipeline().retrieve(query, k)
    return {
        "status": "success",
        "passages": [{"n": i, "source": h.source, "page": h.page, "text": h.text} for i, h in enumerate(hits, 1)],
    }


root_agent = Agent(
    model=get_settings().gen_model,
    name="paper_research_agent",
    description="Answers questions about vortex whistles, flow-induced tones and flow sensing using an indexed paper set.",
    instruction=(
        "You answer questions using ONLY passages returned by the search_papers tool. "
        "For comparisons, call the tool once per topic with a focused query. "
        "Cite as (source, page). If the passages do not contain the answer, say you cannot find it in the papers."
    ),
    tools=[search_papers],
)
