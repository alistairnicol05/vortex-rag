"""Central settings. Every tunable lives here so evaluation sweeps are one-line changes."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

REFUSAL_TEXT = "I cannot find this in the provided documents."


@dataclass(frozen=True)
class Settings:
    # --- models (verified against ai.google.dev docs on 2026-10-19; re-check before use) ---
    gen_model: str = os.getenv("GEN_MODEL", "gemini-3.1-flash-lite")
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    # --- chunking ---
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_chars: int = 200  # drop tiny fragments (page numbers, stray headers)

    # --- retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))
    hybrid: bool = os.getenv("HYBRID", "0") == "1"
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # --- paths ---
    papers_dir: Path = field(default_factory=lambda: ROOT / "data" / "papers")
    chroma_dir: Path = field(default_factory=lambda: ROOT / "chroma_db")
    index_dir: Path = field(default_factory=lambda: ROOT / "data" / "index")
    trace_path: Path = field(default_factory=lambda: ROOT / "traces" / "requests.jsonl")
    collection: str = os.getenv("COLLECTION", "papers")

    @property
    def api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def get_settings() -> Settings:
    return Settings()
