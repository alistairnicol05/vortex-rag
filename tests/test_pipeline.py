"""Offline end-to-end tests: tiny generated PDFs, FakeEmbedder, StubLLM. No network, no API key."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import chromadb
import numpy as np
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag.config import REFUSAL_TEXT, get_settings
from rag.embed_store import VectorStore, load_index, open_from_index, save_index
from rag.embeddings import FakeEmbedder
from rag.evaluate import (EvalConfig, agreement, evaluate_one, hit_at_k, load_questions, markdown_table,
                          percentile, reciprocal_rank, relevant_keys, source_recall, summarize, write_spotcheck)
from rag.ingest import ingest_dir, split_text
from rag.llm import StubLLM
from rag.pricing import llm_cost_usd
from rag.query import RagPipeline, is_refusal, parse_citations
from rag.retrieve import BM25, hybrid_search
from rag.trace import read_traces

PAGES = {
    "whistle.pdf": [
        "The vortex whistle produces a pure tone whose frequency increases linearly with the volumetric flow rate. "
        "Swirling flow enters tangentially and exits through a narrow outlet tube. " * 6,
        "Acoustic power of the vortex whistle scales with the sixth power of frequency in a reverberant chamber. " * 8,
    ],
    "das.pdf": [
        "Distributed acoustic sensing uses optical fibre along the wellbore to record vibration at every metre. " * 8,
    ],
    "shedding.pdf": [
        "A vortex shedding flowmeter relates the shedding frequency behind a bluff body to velocity via the Strouhal number. " * 8,
    ],
}


def make_pdf(path: Path, pages: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    for text in pages:
        y, line = 740, ""
        for word in text.split():
            if len(line) + len(word) > 90:
                c.drawString(40, y, line)
                y, line = y - 14, ""
            line += word + " "
        c.drawString(40, y, line)
        c.showPage()
    c.save()


@pytest.fixture()
def env(tmp_path: Path):
    papers = tmp_path / "papers"
    papers.mkdir()
    for name, pages in PAGES.items():
        make_pdf(papers / name, pages)
    s = replace(get_settings(), papers_dir=papers, chroma_dir=tmp_path / "chroma", index_dir=tmp_path / "index",
                trace_path=tmp_path / "traces.jsonl", top_k=3)
    chunks = ingest_dir(papers, s.chunk_chars, s.chunk_overlap, s.min_chunk_chars)
    emb = FakeEmbedder()
    vecs = emb.embed_documents([c.text for c in chunks])
    save_index(s.index_dir, chunks, vecs, "fake")
    store = VectorStore(chromadb.EphemeralClient(), f"t{tmp_path.name[-8:]}")
    store.add(chunks, vecs)
    return s, chunks, emb, store


def test_split_text_overlap_and_bounds():
    text = ("Sentence number one is here. " * 100).strip()
    parts = split_text(text, 300, 50)
    assert all(len(p) <= 300 for p in parts)
    assert len(parts) > 5
    with pytest.raises(ValueError):
        split_text(text, 100, 100)


def test_ingest_keeps_page_metadata(env):
    _, chunks, _, _ = env
    assert {c.source for c in chunks} == set(PAGES)
    assert {c.page for c in chunks if c.source == "whistle.pdf"} == {1, 2}
    assert all(c.id.startswith(f"{c.source}:p{c.page}") for c in chunks)


def test_index_roundtrip_and_retrieval(env):
    s, chunks, emb, store = env
    ch2, vecs, meta = load_index(s.index_dir)
    assert len(ch2) == len(chunks) == vecs.shape[0] and meta["embed_model"] == "fake"
    hits = store.query(emb.embed_query("distributed acoustic sensing optical fibre wellbore"), 3)
    assert hits[0].source == "das.pdf" and hits[0].page == 1
    assert hits[0].score >= hits[-1].score
    store2, _, _ = open_from_index(s.index_dir, "again")
    assert store2.count() == len(chunks)


def test_hybrid_returns_k(env):
    _, chunks, emb, store = env
    q = "Strouhal number bluff body"
    hits = hybrid_search(store, BM25(chunks), emb.embed_query(q), q, 3)
    assert len(hits) == 3 and hits[0].source == "shedding.pdf"


def test_pipeline_answer_citations_and_trace(env):
    s, chunks, emb, store = env
    pipe = RagPipeline(s, emb, store, StubLLM())
    a = pipe.ask("How does vortex whistle frequency depend on flow rate?")
    assert not a.refused and a.cited == [1] and not a.invalid_citations
    assert a.hits[0].source == "whistle.pdf"
    traces = read_traces(s.trace_path)
    assert len(traces) == 1
    t = traces[0]
    assert {"retrieve", "generate"} <= set(t["stages_s"]) and t["input_tokens"] > 0 and t["retrieved"][0]["page"] >= 1


def test_refusal_detection_and_citation_parsing():
    assert is_refusal(REFUSAL_TEXT) and not is_refusal("The tone rises with flow [S1].")
    assert parse_citations("A [S1] and B [S2, S3] and C [S9][S1].") == [1, 2, 3, 9]


def test_metrics_pure_functions(env):
    _, chunks, emb, store = env
    hits = store.query(emb.embed_query("distributed acoustic sensing fibre"), 5)
    rel = relevant_keys([{"source": "das.pdf", "pages": [1]}])
    assert hit_at_k(hits, rel, 1) == 1 and reciprocal_rank(hits, rel, 5) == 1.0
    miss = relevant_keys([{"source": "nope.pdf", "pages": [1]}])
    assert hit_at_k(hits, miss, 5) == 0 and reciprocal_rank(hits, miss, 5) == 0.0
    assert source_recall(hits, [{"source": "das.pdf", "pages": [1]}, {"source": "nope.pdf", "pages": [1]}], 5) == 0.5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert llm_cost_usd("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(1.75)
    assert llm_cost_usd("unknown-model", 1, 1) is None


def test_full_evaluation_run(env, tmp_path):
    s, chunks, emb, store = env
    qs = [
        {"id": "q1", "type": "single", "question": "How does vortex whistle frequency depend on flow rate?",
         "gold_answer": "linearly", "gold": [{"source": "whistle.pdf", "pages": [1]}]},
        {"id": "q2", "type": "multi", "question": "Compare fibre acoustic sensing with Strouhal shedding flowmeter",
         "gold_answer": "x", "gold": [{"source": "das.pdf", "pages": [1]}, {"source": "shedding.pdf", "pages": [1]}]},
        {"id": "q3", "type": "unanswerable", "question": "REFUSE_ME What is the capital of Utah?",
         "gold_answer": "", "gold": []},
    ]
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    loaded = load_questions(qpath)
    pipe = RagPipeline(s, emb, store, StubLLM())
    cfg = EvalConfig(k=3, judge=StubLLM())
    rows = [evaluate_one(pipe, q, cfg) for q in loaded]
    summ = summarize(rows, 3)
    assert summ["refusal_rate_unanswerable"] == 1.0 and summ["hit@5"] == 1.0 and summ["mrr"] > 0.5
    assert summ["correctness_mean_0to2"] == 2.0 and summ["faithful_rate"] == 1.0
    md = markdown_table("t", summ, {"k": 3})
    assert "MRR" in md and "Cost per request" in md
    sc = tmp_path / "spot.csv"
    write_spotcheck(rows, 2, sc)
    text = sc.read_text().replace(",,", ",2,", 1)  # fake a human label
    sc.write_text(text)
    assert agreement(sc)["n"] >= 1


def test_token_counts_matches_real_sdk_type():
    from google.genai import types

    from rag.llm import token_counts

    u = types.UsageMetadata(prompt_token_count=100, response_token_count=20, thoughts_token_count=30, total_token_count=150)
    assert token_counts(u) == (100, 50)  # thinking tokens count as output
    assert token_counts(types.UsageMetadata(prompt_token_count=10, total_token_count=25)) == (10, 15)
    assert token_counts(None) == (0, 0)


def test_load_questions_rejects_bad_gold(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "type": "unanswerable", "question": "q", "gold_answer": "", "gold": [{"source": "a", "pages": [1]}]}))
    with pytest.raises(AssertionError):
        load_questions(p)
