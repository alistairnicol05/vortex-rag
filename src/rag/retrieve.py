"""Optional hybrid retrieval: dense (Chroma) + BM25, fused with reciprocal rank fusion (RRF)."""
from __future__ import annotations

import math
import re
from collections import Counter

from .embed_store import Hit, VectorStore
from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores: list[tuple[int, float]] = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                if t in tf:
                    denom = tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / denom
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])[:k]


def hybrid_search(
    store: VectorStore, bm25: BM25, qvec, qtext: str, k: int, pool: int = 20, rrf_k: int = 60
) -> list[Hit]:
    dense = store.query(qvec, max(pool, k))
    sparse = bm25.top(qtext, pool)
    fused: dict[str, float] = {}
    by_id: dict[str, Hit] = {h.id: h for h in dense}
    for rank, h in enumerate(dense, start=1):
        fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (i, _) in enumerate(sparse, start=1):
        c = bm25.chunks[i]
        fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (rrf_k + rank)
        by_id.setdefault(c.id, Hit(c.id, c.source, c.page, c.text, 0.0))
    ranked = sorted(fused, key=lambda i: -fused[i])[:k]
    return [Hit(by_id[i].id, by_id[i].source, by_id[i].page, by_id[i].text, fused[i]) for i in ranked]
