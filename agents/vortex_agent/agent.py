"""ADK agent that uses the existing RAG pipeline as a tool.

Works in two places:
  - locally with `adk run agents/vortex_agent` (builds its own pipeline), and
  - inside the Cloud Run service (reuses the pipeline that main.py already loaded, so the index is in memory once).
"""
from __future__ import annotations

from google.adk.agents.llm_agent import Agent

from rag.config import get_settings

_local_pipeline = None


def _get_pipeline():
    # Inside the deployed service, rag.app.state["pipe"] is filled at startup.
    from rag.app import state
    if state.get("pipe") is not None:
        return state["pipe"]
    # Local `adk run`: build the same pipeline the API uses, from the prebuilt index.
    global _local_pipeline
    if _local_pipeline is None:
        from rag.embed_store import open_from_index
        from rag.embeddings import GeminiEmbedder
        from rag.llm import GeminiLLM
        from rag.query import RagPipeline
        from rag.retrieve import BM25
        s = get_settings()
        store, chunks, _ = open_from_index(s.index_dir, s.collection)
        _local_pipeline = RagPipeline(s, GeminiEmbedder(s), store, GeminiLLM(s), BM25(chunks) if s.hybrid else None, trace_path=None)
    return _local_pipeline


def search_papers(query: str, k: int = 5) -> dict:
    """Search the indexed research papers and return the most relevant passages.

    Args:
        query: A self-contained search query. Rephrase the user's question to include the key terms.
        k: Number of passages to return (1 to 8).

    Returns:
        A dict with a status and a list of passages, each with its source file, page number and text.
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
    description="Answers questions about vortex whistles, flow-induced tones and flow sensing using an indexed set of research papers.",
    instruction=(
        "Answer using ONLY passages returned by the search_papers tool. "
        "For comparisons, call the tool once per topic with a focused query. "
        "Cite every claim as (source, page). "
        "If the passages do not contain the answer, say you cannot find it in the papers."
    ),
    tools=[search_papers],
)
