"""Embedders behind one interface: GeminiEmbedder (real) and FakeEmbedder (offline tests)."""
from __future__ import annotations

import hashlib # used by FakeEmbedder to hash tokens into a fixed-size vector
import re # used by FakeEmbedder to split text into tokens
import time
from typing import Protocol # used to define the Embedder interface

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential  # automatic retry on failed API calls

from .config import Settings


# Any class with these three things counts as an Embedder, so the rest of the code
# can use GeminiEmbedder or FakeEmbedder without caring which one it got.
class Embedder(Protocol):
    dim: int  # numbers per vector

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # many chunks -> one row per chunk

    def embed_query(self, text: str) -> np.ndarray: ...  # one question -> one vector


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so similarity depends only on direction (meaning), not size."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)  # length of each row
    return m / np.clip(norms, 1e-12, None)  # clip avoids dividing by zero for an all-zero row


class GeminiEmbedder:
    """Google GenAI SDK embeddings.

    gemini-embedding-001: pass task_type (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY); one vector per string.
    gemini-embedding-2:   task_type is NOT supported; put the task in the text prefix, and wrap each
                          string in its own Content object, otherwise the API returns ONE aggregated vector.
    Truncated dimensions are normalised here (required for 001, harmless for 2).
    """

    def __init__(self, settings: Settings, batch_size: int = 64, pause_s: float = 0.0) -> None:
        from google import genai  # imported lazily so tests do not need the SDK configured

        # Check if the API key is set
        if not settings.api_key:
            raise RuntimeError("Set GEMINI_API_KEY (see .env.example).")
        self._client = genai.Client(api_key=settings.api_key)  # connection to Google's API
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.batch_size = batch_size  # texts sent per API call
        self.pause_s = pause_s  # optional wait between calls, to stay under rate limits
        self._is_v2 = self.model.startswith("gemini-embedding-2")  # v2 needs a different request format

    # On error: try up to 5 times, waiting longer each time (2s, 4s, 8s, ... max 60s).
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _call(self, batch: list[str], task_type: str) -> list[list[float]]:
        """One API call: a batch of texts in, one vector per text out."""
        from google.genai import types

        if self._is_v2:
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
            cfg = types.EmbedContentConfig(output_dimensionality=self.dim)
        else:
            contents = batch
            cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        res = self._client.models.embed_content(model=self.model, contents=contents, config=cfg)  # the network call
        vecs = [list(e.values) for e in res.embeddings]
        if len(vecs) != len(batch):  # safety check: one vector per text, no more, no less
            raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(vecs)}")
        return vecs

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        """Split texts into batches, embed each batch, return a normalised (n_texts, dim) array."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):  # i = 0, 64, 128, ...
            out.extend(self._call(texts[i : i + self.batch_size], task_type))
            if self.pause_s:
                time.sleep(self.pause_s)
        return _l2_normalize(np.asarray(out, dtype=np.float32))  # float32 halves memory vs float64

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed paper chunks (tuned for 'text to be searched')."""
        if self._is_v2:
            texts = [f"title: none | text: {t}" for t in texts]
        return self._embed(texts, "RETRIEVAL_DOCUMENT") # task_type is ignored for v2, but required for v1

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question (tuned for 'text doing the searching')."""
        if self._is_v2:
            text = f"task: question answering | query: {text}"
        return self._embed([text], "RETRIEVAL_QUERY")[0]  # [0]: one question -> one vector


class FakeEmbedder:
    """Deterministic hashed bag-of-words. Not semantic, but similar texts land close together,
    so retrieval tests are meaningful. Zero cost, zero network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):  # split into lowercase words
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # same word -> same number, every run
            v[h % self.dim] += 1.0  # count the word in one of `dim` buckets
        return np.log1p(v)  # log(1 + count): stops very common words dominating

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))  # stack rows into one array

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vec(text)[None, :])[0]  # [None, :] makes it a 1-row array for normalising
