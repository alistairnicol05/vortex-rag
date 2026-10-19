# Vortex-RAG: cited Q&A over flow-induced acoustics papers

A small retrieval-augmented generation (RAG) system over ~20 open-access papers on vortex whistles,
flow-induced tones and acoustic flow sensing. Answers carry `[S#]` citations back to source file and page,
and the system refuses when the papers do not contain the answer.

Context: built to learn RAG end to end while working on a capstone (passive vortex whistle for
geothermal brine flow sensing). It is a learning project, not a production system.

## Architecture

```
PDFs -> ingest.py (PyMuPDF, page-bounded chunks) -> chunks.jsonl
     -> embed_store.py (gemini-embedding-001, 768-d) -> Chroma (cosine) + data/index export
query -> embed -> top-k (optional BM25 + RRF hybrid) -> cited prompt -> Gemini -> answer + [S#]
every request -> traces/requests.jsonl (stage timings, tokens, cost)
```

## Quickstart (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"            # needed in every new terminal
Copy-Item .env.example .env        # then paste your GEMINI_API_KEY
python scripts\download_papers.py
python -m rag.ingest               # optional: writes data\index\chunks.jsonl so you can inspect the chunks
python -m rag.embed_store          # re-chunks, embeds (paid API calls) and fills Chroma + data\index
python -m rag.query "How does vortex whistle frequency depend on flow velocity?"
```

## Evaluation

`eval/questions.jsonl`: __N__ questions (__n__ single-document, __n__ multi-document, __n__ unanswerable),
gold answers and gold pages written by hand.

Run: `python -m rag.evaluate --tag baseline` (add `--hybrid`, `--k 3`, etc. for variants).

### Results (fill from `results/*.md`; do not edit numbers by hand)

| Config | hit@1 | hit@5 | MRR | Judge correctness (0-2) | Faithfulness | Refusal acc. | p50 latency (s) | p95 latency (s) | Tokens in/out | Cost / req (USD) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (dense, k=5) | __ | __ | __ | __ | __ | __ | __ | __ | __ / __ | __ |
| hybrid (BM25 + RRF) | __ | __ | __ | __ | __ | __ | __ | __ | __ / __ | __ |

Manual spot-check of 10 judged answers: judge agreed with me on __ / 10.

### Failure analysis

| # | Question id | Failure type (retrieval / generation / citation / refusal) | What happened | Fix tried | Result |
|---|---|---|---|---|---|
| 1 | | | | | |

## Deployment

Cloud Run (private by default), index baked into the image, API key from Secret Manager.
See `docs/deploy.md`. Live URL: __ (or: "kept private to respect paper copyright").

## Limitations

- Text-only parsing: tables and equations inside PDFs are extracted as plain text and often garbled.
- __n__-question evaluation set written by one person; numbers are indicative, not statistically strong.
- LLM judge is itself a model; agreement with my manual check is reported above.
- Papers are not redistributed here; `scripts/download_papers.py` fetches them from their open-access hosts.

## Layout

```
src/rag/       config, ingest, embeddings, embed_store, retrieve, llm, query, judge, evaluate, app, trace, pricing
scripts/       download_papers.py
data/          papers.csv (committed), papers/ (ignored), index/ (built)
eval/          questions.jsonl
tests/         offline tests (fake embedder + stub LLM)
vortex_agent/  optional Google ADK agent (separate venv)
```
