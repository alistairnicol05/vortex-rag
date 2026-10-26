"""LLM-as-judge with written rubrics. Judge model differs from the generator to limit self-preference bias.

Two separate judgements:
  correctness  (0/1/2) compares the answer with the gold answer.
  faithfulness (0/1)   checks that every claim is supported by the passages the answer cites.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embed_store import Hit
from .llm import LLM, parse_json
from .pricing import llm_cost_usd

CORRECTNESS_RUBRIC = """You are grading an answer to a question about scientific papers.
Compare the CANDIDATE answer with the GOLD answer. Ignore style, length and citation markers like [S1].

Score:
2 = Fully correct: contains all key facts of the gold answer, no contradicting statements.
1 = Partially correct: contains some key facts but misses one, or adds a minor error.
0 = Incorrect, contradicts the gold answer, or is a refusal ("I cannot find this...").

Return JSON only: {"correctness": 0|1|2, "reasoning": "<one sentence>"}"""

FAITHFULNESS_RUBRIC = """You are checking whether an answer is supported by the passages it cites.
You get the CANDIDATE answer and the CITED PASSAGES (only those the answer cites).

Score:
1 = Every factual claim in the answer is supported by the cited passages.
0 = At least one factual claim is NOT supported by the cited passages (even if it is true in general),
    or the answer cites nothing while making factual claims.

Return JSON only: {"faithful": 0|1, "reasoning": "<one sentence>"}"""


@dataclass
class JudgeScores:
    correctness: int | None
    faithful: int | None
    correctness_note: str
    faithful_note: str


def _score(text: str, key: str) -> tuple[int | None, str]:
    """(score, reasoning) from the judge's JSON reply. A reply that cannot be read (e.g. cut off
    mid-string) gives score None, so one bad reply skips that score instead of crashing the run."""
    try:
        d = parse_json(text)
        return int(d[key]), str(d.get("reasoning", ""))
    except (ValueError, KeyError, TypeError):  # JSONDecodeError is a ValueError
        return None, f"judge reply unreadable: {text[:200]!r}"


def judge_correctness(judge: LLM, question: str, gold: str, candidate: str) -> tuple[int | None, str, float | None]:
    """Returns (score, reasoning, cost_usd). score is None if the reply is unreadable;
    cost_usd is None if the judge model has no price."""
    prompt = f"{CORRECTNESS_RUBRIC}\n\nQUESTION: {question}\nGOLD: {gold}\nCANDIDATE: {candidate}"
    res = judge.generate(prompt, json_mode=True)
    cost = llm_cost_usd(res.model, res.input_tokens, res.output_tokens)  # the call is billed even if unreadable
    return (*_score(res.text, "correctness"), cost)


def judge_faithfulness(judge: LLM, candidate: str, cited_hits: list[Hit]) -> tuple[int | None, str, float | None]:
    """Returns (score, reasoning, cost_usd). No API call (cost 0) when nothing is cited."""
    if not cited_hits:
        return 0, "no valid citations", 0.0
    ctx = "\n\n".join(f"[{h.source} p.{h.page}]\n{h.text}" for h in cited_hits)
    prompt = f"{FAITHFULNESS_RUBRIC}\n\nCANDIDATE: {candidate}\n\nCITED PASSAGES:\n{ctx}"
    res = judge.generate(prompt, json_mode=True)
    cost = llm_cost_usd(res.model, res.input_tokens, res.output_tokens)
    return (*_score(res.text, "faithful"), cost)
