"""Central settings. Every tunable lives here so evaluation sweeps are one-line changes."""
from __future__ import annotations  # lets type hints like "str | None" work everywhere

import os  # reads environment variables
from dataclasses import dataclass, field  # quick way to define a class that just holds values
from pathlib import Path  # file paths that work on Windows and Linux

from dotenv import load_dotenv

load_dotenv()  # copy the lines in .env (e.g. GEMINI_API_KEY) into environment variables

ROOT = Path(__file__).resolve().parents[2]  # project root: this file is src/rag/config.py, so go up 2 folders

REFUSAL_TEXT = "I cannot find this in the provided documents."  # reply when the papers lack the answer


# Each setting: os.getenv("NAME", "default") uses the env var / .env value if set, else the default.
@dataclass(frozen=True)  # frozen = settings can't be changed by accident once created
class Settings:
    # --- models (verified against ai.google.dev docs on 2026-10-19; re-check before use) ---
    gen_model: str = os.getenv("GEN_MODEL", "gemini-3.1-flash-lite")  # writes the answers
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")  # grades the answers (a different model)
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")  # turns text into vectors
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))  # numbers per vector

    # --- chunking ---
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1200"))  # max characters per chunk
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters shared by neighbouring chunks
    min_chunk_chars: int = 200  # drop tiny fragments (page numbers, stray headers)

    # --- retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))  # how many chunks are given to the model per question
    hybrid: bool = os.getenv("HYBRID", "0") == "1"  # True only if HYBRID=1: adds keyword (BM25) search
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # --- paths ---
    # field(default_factory=...) builds the path when Settings() is created
    papers_dir: Path = field(default_factory=lambda: ROOT / "data" / "papers")  # downloaded PDFs
    chroma_dir: Path = field(default_factory=lambda: ROOT / "chroma_db")  # local vector database
    index_dir: Path = field(default_factory=lambda: ROOT / "data" / "index")  # exported chunks + vectors for deploy
    trace_path: Path = field(default_factory=lambda: ROOT / "traces" / "requests.jsonl")  # one log line per question
    collection: str = os.getenv("COLLECTION", "papers")  # table name inside Chroma

    @property  # read like a value (settings.api_key), but looked up fresh each time
    def api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # None if neither is set


def get_settings() -> Settings:
    return Settings()  # the rest of the code calls this to get the settings
