"""Evaluation harness: retrieval (hit@k, MRR), answers (LLM judge), faithfulness, refusal, latency, tokens, cost.

Usage:
    python -m rag.evaluate --questions eval/questions.jsonl --tag baseline
    python -m rag.evaluate --tag k8 --k 8 --spotcheck 10
    python -m rag.evaluate --tag hybrid --hybrid --plot
    python -m rag.evaluate --agreement results/baseline_spotcheck.csv     # after you fill human_correctness

questions.jsonl, one object per line:
    {"id": "q01", "type": "single" | "multi" | "unanswerable", "question": "...",
     "gold_answer": "...", "gold": [{"source": "paper.pdf", "pages": [3, 4]}]}
    Unanswerable questions have "gold": [] and gold_answer "".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .embed_store import Hit
from .judge import judge_correctness, judge_faithfulness
from .llm import LLM, GeminiLLM
from .query import RagPipeline, build_default_pipeline

K_SWEEP = (1, 3, 5, 10)


# ---------- pure metric functions (unit-tested) ----------
def relevant_keys(gold: list[dict], page_tolerance: int = 0) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for g in gold:
        for p in g["pages"]:
            for d in range(-page_tolerance, page_tolerance + 1):
                keys.add((g["source"], p + d))
    return keys


def reciprocal_rank(hits: list[Hit], rel: set[tuple[str, int]], k: int) -> float:
    for rank, h in enumerate(hits[:k], start=1):
        if (h.source, h.page) in rel:
            return 1.0 / rank
    return 0.0


def hit_at_k(hits: list[Hit], rel: set[tuple[str, int]], k: int) -> int:
    return int(any((h.source, h.page) in rel for h in hits[:k]))


def source_recall(hits: list[Hit], gold: list[dict], k: int) -> float:
    """Fraction of gold documents with at least one retrieved chunk in the top-k (for multi-doc questions)."""
    if not gold:
        return float("nan")
    got = {h.source for h in hits[:k]}
    return sum(g["source"] in got for g in gold) / len(gold)


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    for q in qs:
        assert q["type"] in {"single", "multi", "unanswerable"}, f"bad type in {q['id']}"
        assert (q["type"] == "unanswerable") == (not q["gold"]), f"gold/type mismatch in {q['id']}"
    return qs


# ---------- evaluation loop ----------
@dataclass
class EvalConfig:
    k: int
    page_tolerance: int = 0
    judge: LLM | None = None


def evaluate_one(pipe: RagPipeline, q: dict, cfg: EvalConfig) -> dict[str, Any]:
    kmax = max(max(K_SWEEP), cfg.k)
    hits_all = pipe.retrieve(q["question"], kmax)
    ans = pipe.ask(q["question"], k=cfg.k, qid=q["id"], qtype=q["type"])
    row: dict[str, Any] = {
        "id": q["id"], "type": q["type"], "question": q["question"], "gold_answer": q["gold_answer"],
        "answer": ans.text, "refused": ans.refused, "cited": ans.cited, "invalid_citations": len(ans.invalid_citations),
        "input_tokens": ans.input_tokens, "output_tokens": ans.output_tokens, "cost_usd": ans.cost_usd,
        "total_s": ans.trace["total_s"], "ttft_s": ans.trace["ttft_s"],
        "decode_tps": ans.trace["decode_tokens_per_s"], "retrieved": [f"{h.source}:p{h.page}" for h in ans.hits],
    }
    if q["type"] != "unanswerable":
        rel = relevant_keys(q["gold"], cfg.page_tolerance)
        for k in K_SWEEP:
            row[f"hit@{k}"] = hit_at_k(hits_all, rel, k)
        row["rr"] = reciprocal_rank(hits_all, rel, kmax)
        row["source_recall"] = source_recall(hits_all, q["gold"], cfg.k)
    if cfg.judge is not None:
        judge_costs: list[float | None] = []  # one entry per judge API call
        if q["type"] == "unanswerable":
            row["correctness"] = 2 if ans.refused else 0  # correct behaviour = refuse
            row["faithful"] = None
        else:
            if ans.refused:
                row["correctness"], row["correctness_note"] = 0, "refused"
            else:
                row["correctness"], row["correctness_note"], c = judge_correctness(
                    cfg.judge, q["question"], q["gold_answer"], ans.text
                )
                judge_costs.append(c)
            if not ans.refused:
                cited_hits = [ans.hits[n - 1] for n in ans.cited if 1 <= n <= len(ans.hits)]
                row["faithful"], row["faithful_note"], c = judge_faithfulness(cfg.judge, ans.text, cited_hits)
                judge_costs.append(c)
            else:
                row["faithful"] = None
        # None if any call's model is unpriced, so a partial sum is never reported as the total.
        row["judge_cost_usd"] = None if None in judge_costs else float(sum(judge_costs))
    return row


def summarize(rows: list[dict], k: int) -> dict[str, Any]:
    ans_rows = [r for r in rows if r["type"] != "unanswerable"]
    un_rows = [r for r in rows if r["type"] == "unanswerable"]
    multi = [r for r in rows if r["type"] == "multi"]
    s: dict[str, Any] = {"n": len(rows), "n_answerable": len(ans_rows), "n_unanswerable": len(un_rows), "k": k}
    for kk in K_SWEEP:
        s[f"hit@{kk}"] = mean([r[f"hit@{kk}"] for r in ans_rows])
    s["mrr"] = mean([r["rr"] for r in ans_rows])
    s["multi_source_recall"] = mean([r["source_recall"] for r in multi])
    if rows and "correctness" in rows[0]:
        s["correctness_mean_0to2"] = mean([r["correctness"] for r in ans_rows])
        s["fully_correct_rate"] = mean([float(r["correctness"] == 2) for r in ans_rows])
        s["faithful_rate"] = mean([r["faithful"] for r in ans_rows if r.get("faithful") is not None])
    s["refusal_rate_unanswerable"] = mean([float(r["refused"]) for r in un_rows])
    s["false_refusal_rate_answerable"] = mean([float(r["refused"]) for r in ans_rows])
    s["invalid_citation_rate"] = mean([float(r["invalid_citations"] > 0) for r in rows if not r["refused"]])
    lat = [r["total_s"] for r in rows]
    s["latency_p50_s"], s["latency_p95_s"] = percentile(lat, 50), percentile(lat, 95)
    s["ttft_p50_s"] = percentile([r["ttft_s"] for r in rows], 50)
    s["decode_tokens_per_s_median"] = percentile([r["decode_tps"] for r in rows if r["decode_tps"]], 50)
    s["input_tokens_mean"] = mean([r["input_tokens"] for r in rows])
    s["output_tokens_mean"] = mean([r["output_tokens"] for r in rows])
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    s["cost_per_request_usd"] = mean(costs)
    s["cost_total_usd"] = float(sum(costs)) if costs else float("nan")  # answer calls only
    if rows and "judge_cost_usd" in rows[0]:
        jc = [r["judge_cost_usd"] for r in rows]
        s["judge_cost_total_usd"] = float("nan") if None in jc else float(sum(jc))
    else:
        s["judge_cost_total_usd"] = 0.0  # --no-judge
    # Whole eval run (answers + judge). Excludes embedding the questions.
    s["eval_cost_total_usd"] = s["cost_total_usd"] + s["judge_cost_total_usd"]
    return s


def markdown_table(tag: str, s: dict[str, Any], config: dict[str, Any]) -> str:
    def f(x: Any, nd: int = 3) -> str:
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    lines = [
        f"### Results: `{tag}`", "",
        "| Config | " + " | ".join(f"{k}={v}" for k, v in config.items()) + " |", "|---|---|", "",
        "| Metric | Value |", "|---|---|",
        f"| Questions (answerable / unanswerable) | {s['n_answerable']} / {s['n_unanswerable']} |",
        f"| Hit@1 / Hit@3 / Hit@5 / Hit@10 | {f(s['hit@1'])} / {f(s['hit@3'])} / {f(s['hit@5'])} / {f(s['hit@10'])} |",
        f"| MRR | {f(s['mrr'])} |",
        f"| Multi-doc source recall@k | {f(s['multi_source_recall'])} |",
    ]
    if "correctness_mean_0to2" in s:
        lines += [
            f"| Answer correctness (0-2, LLM judge) | {f(s['correctness_mean_0to2'], 2)} |",
            f"| Fully correct rate | {f(s['fully_correct_rate'], 2)} |",
            f"| Citation faithfulness (LLM judge) | {f(s['faithful_rate'], 2)} |",
        ]
    lines += [
        f"| Correct refusal rate (unanswerable) | {f(s['refusal_rate_unanswerable'], 2)} |",
        f"| False refusal rate (answerable) | {f(s['false_refusal_rate_answerable'], 2)} |",
        f"| Invalid-citation rate | {f(s['invalid_citation_rate'], 2)} |",
        f"| Latency p50 / p95 (s) | {f(s['latency_p50_s'], 2)} / {f(s['latency_p95_s'], 2)} |",
        f"| Time to first token p50 (s) | {f(s['ttft_p50_s'], 2)} |",
        f"| Decode tokens/s (median) | {f(s['decode_tokens_per_s_median'], 1)} |",
        f"| Tokens in / out (mean) | {f(s['input_tokens_mean'], 0)} / {f(s['output_tokens_mean'], 0)} |",
        f"| Cost per request, paid rate (USD) | {f(s['cost_per_request_usd'], 6)} |",
        f"| Eval run total: answers + judge (USD) | {f(s['cost_total_usd'], 4)} + {f(s['judge_cost_total_usd'], 4)}"
        f" = {f(s['eval_cost_total_usd'], 4)} |",
    ]
    return "\n".join(lines) + "\n"


def write_spotcheck(rows: list[dict], n: int, path: Path, seed: int = 0) -> None:
    pool = [r for r in rows if r["type"] != "unanswerable" and not r["refused"]]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold_answer", "answer", "judge_correctness", "human_correctness_0to2", "notes"])
        for r in sample:
            w.writerow([r["id"], r["question"], r["gold_answer"], r["answer"], r.get("correctness", ""), "", ""])


def agreement(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness_0to2"].strip() != ""]
    if not rows:
        raise SystemExit("Fill in human_correctness_0to2 first.")
    j = np.array([int(r["judge_correctness"]) for r in rows])
    h = np.array([int(r["human_correctness_0to2"]) for r in rows])
    return {"n": len(rows), "exact_agreement": float((j == h).mean()), "within_1": float((abs(j - h) <= 1).mean())}


def plot_results(rows: list[dict], summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    phi = (1 + 5 ** 0.5) / 2
    w = 7.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * w, w / phi))
    axes[0].plot(K_SWEEP, [summary[f"hit@{k}"] for k in K_SWEEP], marker="o")
    axes[0].set(xlabel="k", ylabel="hit@k", title="Retrieval hit@k", ylim=(0, 1.02), xticks=K_SWEEP)
    axes[1].hist([r["total_s"] for r in rows], bins=15)
    axes[1].set(xlabel="end-to-end latency (s)", ylabel="requests", title="Latency distribution")
    fig.tight_layout()
    plt.show()


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--k", type=int, default=s.top_k)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--collection", default=s.collection)
    ap.add_argument("--page-tolerance", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--spotcheck", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--agreement", type=Path, default=None)
    args = ap.parse_args()

    if args.agreement:
        print(json.dumps(agreement(args.agreement), indent=2))
        return

    settings = Settings(**{**s.__dict__, "hybrid": args.hybrid or s.hybrid, "collection": args.collection, "top_k": args.k})
    pipe = build_default_pipeline(settings)
    judge = None if args.no_judge else GeminiLLM(settings, model=settings.judge_model, max_output_tokens=512)
    cfg = EvalConfig(k=args.k, page_tolerance=args.page_tolerance, judge=judge)

    rows = []
    for q in load_questions(args.questions):
        rows.append(evaluate_one(pipe, q, cfg))
        print(f"[{q['id']}] {q['type']:<12} refused={rows[-1]['refused']}  {rows[-1]['total_s']:.2f}s")

    summary = summarize(rows, args.k)
    config = {"gen": settings.gen_model, "embed": settings.embed_model, "chunk": settings.chunk_chars,
              "overlap": settings.chunk_overlap, "k": args.k, "hybrid": settings.hybrid}
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps({"config": config, "summary": summary, "rows": rows}, indent=2, default=str))
    md = markdown_table(args.tag, summary, config)
    (out / f"{args.tag}.md").write_text(md, encoding="utf-8")
    if args.spotcheck:
        write_spotcheck(rows, args.spotcheck, out / f"{args.tag}_spotcheck.csv")
    print("\n" + md)
    if args.plot:
        plot_results(rows, summary)


if __name__ == "__main__":
    main()
