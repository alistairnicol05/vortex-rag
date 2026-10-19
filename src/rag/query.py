"""Retrieve -> prompt -> generate -> trace.

Usage:
    python -m rag.query "How does vortex whistle frequency scale with flow rate?"
    python -m rag.query "..." --k 8 --hybrid
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import REFUSAL_TEXT, Settings, get_settings
from .embed_store import Hit, VectorStore, open_persistent
from .embeddings import Embedder, GeminiEmbedder
from .ingest import read_chunks
from .llm import LLM, GeminiLLM
from .pricing import llm_cost_usd
from .retrieve import BM25, hybrid_search
from .trace import Trace

SYSTEM_PROMPT = (
    "You answer questions about scientific papers using ONLY the numbered context passages provided.\n"
    "Rules:\n"
    "1. Cite every factual sentence with the passage numbers that support it, like [S1] or [S2][S3].\n"
    f'2. If the passages do not contain the answer, reply with exactly: "{REFUSAL_TEXT}" and nothing else.\n'
    "3. Never use outside knowledge, even if you know the answer.\n"
    "4. If passages disagree, say so and cite both.\n"
    "5. Be concise: at most 4 sentences unless the question needs more."
)

_CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[S{i}] (source: {h.source}, page {h.page})\n{h.text}" for i, h in enumerate(hits, start=1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def parse_citations(answer: str) -> list[int]:
    nums: list[int] = []
    for group in _CITE.findall(answer):
        nums.extend(int(n) for n in re.findall(r"S(\d+)", group))
    return sorted(set(nums))


def is_refusal(answer: str) -> bool:
    return REFUSAL_TEXT.lower().rstrip(".") in answer.lower()


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int]  # 1-indexed passage numbers the answer cites
    invalid_citations: list[int]  # cited numbers that do not exist in the context
    refused: bool
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    trace: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        bm25: BM25 | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.s = settings
        self.embedder, self.store, self.llm, self.bm25 = embedder, store, llm, bm25
        self.trace_path = trace_path if trace_path is not None else settings.trace_path

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_k
        qvec = self.embedder.embed_query(question)
        if self.s.hybrid and self.bm25 is not None:
            return hybrid_search(self.store, self.bm25, qvec, question, k, rrf_k=self.s.rrf_k)
        return self.store.query(qvec, k)

    def ask(self, question: str, k: int | None = None, **trace_fields) -> Answer:
        k = k or self.s.top_k
        tr = Trace(self.trace_path, question=question, k=k, hybrid=self.s.hybrid, **trace_fields)
        with tr.stage("retrieve"):  # includes the query-embedding API call
            hits = self.retrieve(question, k)
        prompt = build_prompt(question, hits)
        with tr.stage("generate"):
            res = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        cited = parse_citations(res.text)
        invalid = [n for n in cited if n < 1 or n > len(hits)]
        refused = is_refusal(res.text)
        cost = llm_cost_usd(res.model, res.input_tokens, res.output_tokens)
        tr.set(
            model=res.model,
            retrieved=[{"id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)} for h in hits],
            answer=res.text,
            cited=cited,
            invalid_citations=invalid,
            refused=refused,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            ttft_s=round(res.ttft_s, 4),
            gen_latency_s=round(res.latency_s, 4),
            decode_tokens_per_s=round(res.decode_tokens_per_s, 2) if res.decode_tokens_per_s else None,
            cost_usd=cost,
        )
        record = tr.write()
        return Answer(question, res.text, hits, cited, invalid, refused, res.model, res.input_tokens,
                      res.output_tokens, cost, record)


def build_default_pipeline(settings: Settings | None = None) -> RagPipeline:
    s = settings or get_settings()
    store = open_persistent(s)
    bm25 = BM25(read_chunks(s.index_dir / "chunks.jsonl")) if s.hybrid else None
    return RagPipeline(s, GeminiEmbedder(s), store, GeminiLLM(s), bm25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--hybrid", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    if args.hybrid:
        s = Settings(**{**s.__dict__, "hybrid": True})
    ans = build_default_pipeline(s).ask(args.question, args.k)
    print(ans.text)
    print("\nSources:")
    for i, h in enumerate(ans.hits, start=1):
        mark = "*" if i in ans.cited else " "
        print(f" {mark}[S{i}] {h.source} p.{h.page}  (cos={h.score:.3f})")
    cost = f"${ans.cost_usd:.6f}" if ans.cost_usd is not None else "n/a"
    print(f"\ntokens in/out: {ans.input_tokens}/{ans.output_tokens}   cost(paid rate): {cost}")


if __name__ == "__main__":
    main()
