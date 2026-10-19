"""Embedders behind one interface: GeminiEmbedder (real) and FakeEmbedder (offline tests)."""
from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class GeminiEmbedder:
    """Google GenAI SDK embeddings.

    gemini-embedding-001: pass task_type (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY); one vector per string.
    gemini-embedding-2:   task_type is NOT supported; put the task in the text prefix, and wrap each
                          string in its own Content object, otherwise the API returns ONE aggregated vector.
    Truncated dimensions are normalised here (required for 001, harmless for 2).
    """

    def __init__(self, settings: Settings, batch_size: int = 64, pause_s: float = 0.0) -> None:
        from google import genai  # imported lazily so tests do not need the SDK configured

        if not settings.api_key:
            raise RuntimeError("Set GEMINI_API_KEY (see .env.example).")
        self._client = genai.Client(api_key=settings.api_key)
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.batch_size = batch_size
        self.pause_s = pause_s
        self._is_v2 = self.model.startswith("gemini-embedding-2")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _call(self, batch: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        if self._is_v2:
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
            cfg = types.EmbedContentConfig(output_dimensionality=self.dim)
        else:
            contents = batch
            cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        res = self._client.models.embed_content(model=self.model, contents=contents, config=cfg)
        vecs = [list(e.values) for e in res.embeddings]
        if len(vecs) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(vecs)}")
        return vecs

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i : i + self.batch_size], task_type))
            if self.pause_s:
                time.sleep(self.pause_s)
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._is_v2:
            texts = [f"title: none | text: {t}" for t in texts]
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_v2:
            text = f"task: question answering | query: {text}"
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class FakeEmbedder:
    """Deterministic hashed bag-of-words. Not semantic, but similar texts land close together,
    so retrieval tests are meaningful. Zero cost, zero network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return np.log1p(v)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vec(text)[None, :])[0]
