# Vortex-RAG: cited Q&A over flow-induced acoustics papers

A small retrieval-augmented generation (RAG) system over 18 open-access papers on vortex whistles,
swirling flows, flow-induced tones and downhole flow sensing. You ask a question in plain English; it answers
**only** from the papers, cites every claim back to a paper and page (`[S1]`, `[S2]`, ...), and replies
"I cannot find this in the provided documents." when the papers do not contain the answer.

Context: built to learn RAG end to end while working on a capstone (a passive vortex whistle for
geothermal brine flow sensing). It is a learning project, not a production system.

## Results at a glance

- **Retrieval:** with hybrid search (meaning + keyword), the page containing the answer reached the top 5 for
  **all 26 answerable questions** (Hit@5 = 1.00, up from 0.885 with meaning-only search).
- **Answers:** about 1.5–1.6 out of 2 on average (LLM judge); about 2 in 3 answers fully correct.
- **Honesty:** all 6 questions with no answer in the papers were correctly refused in every configuration,
  and the model never cited a passage that does not exist.
- **Judge check:** my own grading agreed with the automatic judge on **9 of 10** sampled answers
  (**10 of 10 within one point**).
- **Cost:** about **$0.0005 per question** (half a tenth of a cent). Each full 32-question evaluation run,
  including the judge, cost about $0.08–0.13.
- **Main finding:** once hybrid search fixed retrieval, answer quality did not rise with it. The remaining
  errors come mostly from the answer step (incomplete answers, refusals) and from page-level metrics that
  hide chunk-level misses (see the failure analysis).

## Why RAG?

| Approach | Problem |
|---|---|
| Ask a model directly, no papers | It may not know these niche papers, can invent plausible formulas, gives no page to check, and never says "I don't know". |
| Paste all 18 papers into every question | ~200,000 input tokens per question instead of ~1,700 (over 100× the cost), slower, details in the middle of long inputs get missed, citations are vaguer, and it does not scale to more papers. |
| Keyword search (Ctrl+F) | Only finds exact words: "pitch vs. flow speed" will not find "Strouhal number relates frequency to inlet velocity". |
| **RAG (this project)** | Finds passages by meaning, sends only the 5 most relevant to the model, forces citations to exact pages, and refuses when the answer is not there. New papers are added by re-running the embedding step, with no retraining. |

## How it works

```
PDFs -> ingest.py (PyMuPDF, page-bounded chunks) -> chunks.jsonl
     -> embed_store.py (gemini-embedding-2, 768-d) -> Chroma (cosine) + data/index export
query -> embed -> top-k (optional BM25 + RRF hybrid) -> cited prompt -> Gemini -> answer + [S#]
every request -> traces/requests.jsonl (stage timings, tokens, cost)
```

**Once, before any questions (`python -m rag.embed_store`):**

1. **Chunking.** Each PDF page's text is cut into pieces of at most 1,200 characters, each overlapping the
   previous one by 200 characters, so a sentence cut at a boundary still appears whole in one piece.
   Chunks never cross a page boundary, so every chunk's (paper, page) label is exact. Fragments under 200
   characters (page numbers, stray headers) are dropped. 18 papers → **827 chunks**.
2. **Embedding.** Each chunk is turned into 768 numbers by `gemini-embedding-2`, a "meaning fingerprint":
   chunks about similar things get similar numbers, even with different words.
3. The vectors are stored in a Chroma database (`./chroma_db`) and exported to `data/index/`
   (`chunks.jsonl`, `embeddings.npy`, `meta.json`) so the index can be rebuilt without any API calls.

**For every question (`python -m rag.query "..."`):**

1. The question is embedded the same way.
2. It is compared with all 827 chunks (cosine similarity), and the chunks are ranked.
   With `--hybrid`, a keyword search (BM25) also ranks them, and the two rankings are merged (see below).
3. The top **k = 5** chunks are put in the prompt as `[S1]`–`[S5]`. Chunks ranked 6th or lower are never
   seen by the answer model, so ranking quality matters more than anything else in retrieval.
4. The answer model is instructed to use only those passages, cite every sentence, never use outside
   knowledge, say so if passages disagree, and refuse with an exact sentence if the answer is not there.

**Hybrid search.** Meaning search is good with paraphrases; keyword search (BM25) is good with exact rare
terms like "Strouhal", "Rankine" or "IDDP-2" (rare words score more than common ones). Each produces its own
top 20, and they are merged with reciprocal rank fusion (RRF): each chunk scores
`1/(60 + rank in meaning list) + 1/(60 + rank in keyword list)`. Chunks that both searches like rise to the top.
Keyword search runs locally, so it adds no API cost.

## Models and settings

| Role | Model / value | Why |
|---|---|---|
| Embeddings | `gemini-embedding-2`, 768 dimensions | Current Gemini embedding model; vectors L2-normalised for cosine search |
| Answer model | `gemini-3.1-flash-lite` | Cheap and fast (~1.3 s per answer); temperature 0 for repeatable runs |
| Judge model | `gemini-3.6-flash` | A different, stronger model than the answer model, to limit a model grading its own work. Default in code is `gemini-3.8-flash`, which was repeatedly overloaded (503 errors), so the judge is set with `JUDGE_MODEL=gemini-3.6-flash` in `.env` |
| Chunking | 1,200 characters, 200 overlap, page-bounded | Roughly one paragraph per chunk: specific enough to match a question sharply, small enough that 5 fit cheaply in a prompt |
| k | 5 passages per question | Best balance of answer faithfulness and cost in the evaluation below |
| Vector store | Chroma, cosine distance, collection `papers` | Simple local persistent store |

## Quickstart (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"            # needed in every new terminal
Copy-Item .env.example .env        # then paste your GEMINI_API_KEY and add JUDGE_MODEL=gemini-3.6-flash
python scripts\download_papers.py
python -m rag.ingest               # optional: writes data\index\chunks.jsonl so you can inspect the chunks
python -m rag.embed_store          # re-chunks, embeds (paid API calls) and fills Chroma + data\index
python -m rag.embed_store --from-index   # alternative: refill Chroma from data\index (no API calls)
python -m rag.query "How does vortex whistle frequency depend on flow velocity?"
python -m rag.query "..." --k 8 --hybrid # variants
```

A paid Gemini API key is recommended (see "Development log": the free tier allows only about 20 requests per
day per model, and one evaluation run needs about 80).

## Evaluation

### Question set

`eval/questions.jsonl` holds **32 questions**, each with a hand-written gold answer and the gold page(s)
where the answer is stated:

| Type | Count | What it tests |
|---|---|---|
| `single` | 20 | The answer is in one paper |
| `multi` | 6 | The answer needs facts from two different papers |
| `unanswerable` | 6 | The papers do not contain the answer; the correct response is a refusal |

The questions were drafted with anchors (an exact phrase from the paper), and `scripts/locate_pages.py` found
the page each anchor is on. The questions were then reworded by hand so they do not simply copy the paper's
wording. After the first run, alternate gold pages were added by hand for four questions (see "Development log").

### Metrics

Run with `python -m rag.evaluate --tag <name> [--k 8] [--hybrid] [--spotcheck 10] [--plot]`.
Each run writes `results/<tag>.json` (every question's details), `results/<tag>.md` (summary table) and
`results/<tag>_progress.jsonl` (checkpoint for resuming).

| Metric | Meaning |
|---|---|
| **Hit@k** (k = 1, 3, 5, 10) | Fraction of answerable questions where a gold page is among the top k retrieved chunks. Hit@5 matters most: it is exactly what the answer model sees. Always measured on a top-10 search, so it does not change with the `--k` setting. |
| **MRR** (mean reciprocal rank) | Average of 1 ÷ (rank of the first gold page), 0 if not in the top 10. 1.0 = always ranked first. |
| **Multi-doc source recall** | For multi questions: fraction of the needed papers present in the passages the model saw. |
| **Correctness** (0–2) | LLM judge compares the answer with the gold answer: 2 = all key facts, 1 = partly right or a small error, 0 = wrong or refused. |
| **Fully correct rate** | Fraction of answerable questions scored 2. |
| **Faithfulness** (0/1) | LLM judge checks that every claim is supported by the passages the answer cites, even if it is true in general. |
| **Correct refusal rate** | Fraction of unanswerable questions correctly refused. |
| **False refusal rate** | Fraction of answerable questions wrongly refused. |
| **Invalid-citation rate** | Citations to passages that do not exist (e.g. `[S7]` when only 5 were given). |
| Latency, tokens, cost | Per-request timing (p50/p95), prompt and answer size, and USD cost at paid rates. |

## Results

All runs: 32 questions, `gemini-3.1-flash-lite` answering, `gemini-3.6-flash` judging, 1,200/200 chunks.

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Multi recall | Correctness (0–2) | Fully correct | Faithfulness | Correct refusal | False refusal | p50 / p95 latency (s) | Tokens in / out | Cost / question (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (meaning only, k=5) | 0.577 | 0.769 | 0.885 | 0.962 | 0.693 | 0.917 | 1.54 | 0.73 | **0.95** | 1.00 | 0.15 | 1.27 / 1.59 | 1684 / 40 | 0.00048 |
| k8 (meaning only, k=8) | 0.577 | 0.769 | 0.885 | 0.962 | 0.693 | 1.000 | **1.62** | 0.69 | 0.80 | 1.00 | **0.04** | 1.31 / 1.59 | 2617 / 54 | 0.00074 |
| hybrid (BM25 + RRF, k=5) | 0.462 | **0.846** | **1.000** | **1.000** | 0.684 | 1.000 | 1.54 | 0.62 | 0.88 | 1.00 | 0.08 | 1.25 / 1.65 | 1662 / 50 | 0.00049 |
| hybrid_k8 (BM25 + RRF, k=8) | 0.462 | **0.846** | **1.000** | **1.000** | 0.684 | 1.000 | 1.58 | 0.65 | 0.80 | 1.00 | 0.04 | 1.33 / 1.64 | 2616 / 60 | 0.00074 |

Invalid-citation rate was 0.00 in every run. Total cost of each evaluation run (answers + judge):
baseline $0.082, k8 $0.115, hybrid $0.106, hybrid_k8 $0.125.

**First baseline, before the answer-key fix** (same answers, older gold pages): Hit@1/3/5/10 =
0.462 / 0.615 / 0.731 / 0.962, MRR 0.586. The jump to the numbers above came entirely from correcting the
answer key, not from changing the system (see "Development log").

### Per-question results

`h5` = Hit@5, `C` = correctness (0–2), `F` = faithfulness (– = refused, nothing to check), `R` = refused.
Unanswerable questions q27–q32 were correctly refused in every run and are omitted.

| id | type | baseline h5 / C / F | k8 | hybrid | hybrid_k8 |
|---|---|---|---|---|---|
| q01–q03, q11–q14, q18, q21 | mixed | all 1 / 2 / 1 | same | same | same |
| q04 | single | 0 / 0 / – R | 0 / 0 / 1 | **1 / 2 / 1** | 1 / 2 / 1 |
| q05 | single | 0 / 0 / – R | 0 / 1 / 1 | **1 / 2 / 1** | 1 / 1 / 1 |
| q06 | single | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| q07 | single | 0 / 0 / 1 | 0 / 1 / 0 | 1 / 1 / 1 | 1 / 1 / 1 |
| q08 | single | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 0 / – R | 1 / 2 / 1 |
| q09 | single | 1 / 2 / 1 | 1 / 1 / 1 | 1 / 1 / 0 | 1 / 1 / 0 |
| q10 | single | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 1 / 1 | 1 / 0 / 1 |
| q15 | single | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| q16 | single | 1 / 2 / 0 | 1 / 2 / 0 | 1 / 2 / 0 | 1 / 2 / 0 |
| q17 | single | 1 / 0 / – R | 1 / 0 / – R | 1 / 0 / – R | 1 / 0 / – R |
| q19 | single | 1 / 1 / 1 | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 2 / 1 |
| q20 | single | 1 / 2 / 1 | 1 / 2 / 0 | 1 / 2 / 1 | 1 / 2 / 0 |
| q22 | multi | 1 / 0 / – R | 1 / 1 / 0 | 1 / 1 / 1 | 1 / 1 / 1 |
| q23 | multi | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 2 / 0 |
| q24 | multi | 1 / 2 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| q25 | multi | 1 / 2 / 1 | 1 / 2 / 0 | 1 / 2 / 0 | 1 / 2 / 0 |
| q26 | multi | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 1 / 1 | 1 / 2 / 1 |

## Comparative analysis

**A note on size first.** There are 26 answerable questions, so one question moves a rate by about 0.04.
Differences of 0.04–0.08 (one or two questions) are within noise, and the judge itself varies a little
(q26 got near-identical answers scored 2 in one run and 1 in another). Only larger, consistent differences
are treated as real below.

**1. Hybrid search fixes retrieval.** Hit@5 rose from 0.885 to 1.000 and Hit@10 from 0.962 to 1.000.
The three questions meaning-only search missed (q04, q05, q07) all moved into the top 5, and q04 and q05
went from refusals to fully correct answers. Hybrid works here because these questions share distinctive
words with their gold passages ("2025 authors", "tube length", "exhaust nozzle", "variable-swirl").
Hit@1 fell (0.577 → 0.462): the merge occasionally pushes the best meaning match from 1st to 2nd, which does
not matter when 5 passages are shown. Cost and latency are unchanged.

**2. More passages (k = 8) trade faithfulness for fewer refusals.** With 8 passages the model refused
fewer answerable questions (false refusal 0.15 → 0.04) and correctness rose slightly (1.54 → 1.62), but
faithfulness dropped from 0.95 to 0.80: with more text in front of it, the model blends in claims its
citations do not support. It also costs about 50% more per question (2,617 vs. 1,684 input tokens).

**3. Better retrieval did not mean better answers.** Hybrid reached perfect Hit@5, yet correctness stayed at
1.54 and the fully correct rate fell from 0.73 to 0.62. Some questions got the right page but a different mix
of passages, and the model's answer got less complete (q06, q10) or it refused (q08). Once retrieval works,
**the answer step becomes the bottleneck**: the model often gives a correct but incomplete answer
(score 1), which is the most common non-perfect score in every run.

**4. Refusal behaviour is reliable.** Every run refused all 6 unanswerable questions and produced no invalid
citations. The cost of this caution is some false refusals on answerable questions, which more context
(k = 8) or better retrieval (hybrid) reduces.

**Chosen configuration: hybrid, k = 5.** It has perfect page-level retrieval at the same cost as the
baseline, and its remaining failures are answer-step problems that are clearly diagnosable. Baseline is the
better choice if faithfulness matters most (0.95 vs. 0.88); k = 8 is not worth the extra cost and faithfulness
loss.

## Judge reliability (manual spot-check)

The correctness scores come from an LLM judge, which is itself a model. To check it, 10 judged answers from the
baseline run were sampled (`--spotcheck 10` → `results/baseline_spotcheck.csv`) and graded by me against the
gold answers on the same 0–2 scale. Agreement was computed with
`python -m rag.evaluate --agreement results/baseline_spotcheck.csv`:

| Measure | Result |
|---|---|
| Exact agreement | **9 / 10 (90%)** |
| Agreement within one point | **10 / 10 (100%)** |

The one disagreement was q15: the judge gave 1 for leaving out the well depths, which are in the gold answer;
I gave 2 because the question asks how the cables were installed, not how deep. So the judge is slightly
stricter than me where the gold answer contains more detail than the question asks for.

The faithfulness judge is also strict in a specific way: it marks an answer unfaithful when the answer repeats
a name from the question ("Los Alamos", "LBL", "The Geysers") that does not appear in the cited chunk itself,
even though the paper as a whole supports it (q16, q20, q25). Some of the faithfulness drop in the k = 8 and
hybrid runs is this effect rather than real invented content.

## Failure analysis

| # | Question | Failure type | What happened | Fix tried | Result |
|---|---|---|---|---|---|
| 1 | q07 (swirl number where the PVC appears) | Retrieval: wrong paper | Two papers (Manoharan, Mason) describe swirling jets with different critical swirl numbers. The gold page (Manoharan p1, Sc = 0.61) ranked 6th, so the model answered 0.73 from Mason's experiment, a correct fact from the wrong paper. The question was ambiguous about which jet it meant. | Hybrid; k = 8 | Hybrid moved the gold page into the top 5 and the answer gave 0.61 (score 1: it missed the helical limit-cycle oscillation). k = 8 also reached it but mixed both papers' values. |
| 2 | q04, q05 (whistle dimensions, NASA tube and nozzle length) | Retrieval: near miss | Gold pages ranked outside the top 5, and the model correctly refused rather than guess. q05's source (Goglia 1984) is a scanned report with poor OCR text, which weakens meaning search. | Hybrid | Both reached the top 5 and became fully correct (score 2). The question words ("2025 authors", "exhaust nozzle length") matched the text exactly, which is what keyword search is good at. |
| 3 | q17 (ball-cavity sensor flow range) | Retrieval at chunk level, hidden by a page-level metric | Refused in all four runs despite Hit@5 = 1. The traces show the answer ("0.35 m/s to 6.5 m/s") is in chunk `P18 p1 c1`, which was never retrieved. Other chunks from the same page were, so the page-level Hit@5 counted it as a hit. The model correctly refused, since the passages it saw did not contain the number. | k = 8, hybrid | Not fixed. The lesson: Hit@k at page level overstates retrieval when a page has several chunks. A chunk-level gold label or a reranker would catch this. |
| 4 | q22 (PVC frequency vs. swirl in Manoharan and Mason) | Retrieval: one paper missing (multi) | In the baseline all 5 passages came from Mason (source recall 0.5), so the model refused. | k = 8, hybrid | Both brought in a Manoharan passage (recall 1.0), but not the one with the linear S − Sc relation, so the model said Manoharan had no information (score 1). |
| 5 | q15, q24, q06 (installation details, sensitivity location) | Generation: incomplete answer | Right passages, correct but partial answers: missing depths (q15), the looped cable configuration (q24), or the exact location (q06). | Hybrid; k = 8 | No consistent improvement. This "correct but incomplete" pattern is the most common non-perfect score. A prompt asking for all quantitative details may help. |
| 6 | q08 (exit constriction and PVC frequency) | Generation: false refusal | With hybrid, the passage stating "the PVC frequency is not altered" ranked 2nd, yet the model refused. In the baseline the same fact was answered correctly. | hybrid_k8 | Answered correctly with 8 passages. Shows the answer model's refusal decision is sensitive to passage order and mix. |
| 7 | q09, q10 (whistling frequency, suitable fluids) | Generation: regression with hybrid | Fully correct in the baseline, but with hybrid a different mix of passages led to vaguer answers that missed the aperture area (q09) or the constant-properties condition (q10). q09 was also judged unfaithful ("set by" overstated "in agreement with"). | – | Shows that better page-level retrieval can still change which details reach the model, for better or worse. |
| 8 | q16, q20, q25 (sensor and tool temperature limits) | Judge strictness (faithfulness) | Answers were correct (score 2) but marked unfaithful because they repeat a name from the question ("Los Alamos", "LBL", "The Geysers") that is not in the cited chunk itself. | – | Not a system error. Documented as a judge limitation. |

## Development log: changes made and why

| Problem | Cause | Change |
|---|---|---|
| `embed_store` crashed: Chroma expected dimension 256, got 768 | An old test collection with 256-d vectors was still in `./chroma_db` | A full build now deletes the existing collection before adding vectors. Added `--from-index` to refill Chroma from `data/index/` with no PDF parsing or API calls. |
| Evaluation crashed on `503 UNAVAILABLE` | The default judge model (`gemini-3.8-flash`) was overloaded | Answer and judge calls now retry up to 6 times with exponential backoff (2 s → 60 s), and the judge was switched to `gemini-3.6-flash` with `JUDGE_MODEL` in `.env`. |
| A crash lost a whole run's progress | Results were only written at the end | Each question's result is appended to `results/<tag>_progress.jsonl` as soon as it finishes, and a rerun with the same `--tag` resumes from there. `--fresh` deletes the checkpoint to start over (needed after changing the questions or settings). |
| `JSONDecodeError` from the judge | The judge is a "thinking" model, and thinking tokens count toward the output limit, so a 512-token cap cut its JSON off mid-string | Judge output limit raised to 2,048 tokens. An unreadable reply now records the score as missing (`None`, excluded from averages) instead of crashing the run. |
| A run hung and `Ctrl+C` did not stop it | An overloaded model kept the streaming connection open | Stopped the process manually. In VS Code, the terminal's bin icon or `Ctrl+Break` works when `Ctrl+C` does not. |
| `429 RESOURCE_EXHAUSTED` | The free API tier allows about 20 requests per day per model; one run makes about 80 (32 answers + up to 64 judge calls). A Google Cloud free-trial credit does not raise these limits. | Upgraded the API key to the paid tier ($8 prepaid). The five completed evaluation runs cost about $0.51 in total. |
| First baseline latency p95 of 30 s | Free-tier rate-limit retries during the first 8 questions | Not a system issue. The paid-tier reruns show p95 of about 1.6 s. |
| Four questions (q02, q06, q10, q18) were answered correctly but scored as retrieval misses | The answer key listed only one page per question, but these answers are also stated on other pages (e.g. q02's f⁶ law is in the abstract, p1, and worked out on p5) | Read the cited pages, confirmed they contain the answer, and added them to `pages` in `questions.jsonl` (the scorer counts a hit if **any** listed page is retrieved). Baseline Hit@5 rose from 0.731 to 0.885 with no change to the system. `scripts/locate_pages.py` is not rerun after this, because it would overwrite hand-added pages. |
| q07 was graded wrong, and the gold answer looked suspect | The value 0.61 is written as S_c = 0.61 in the abstract and is easy to miss; the system's 0.73 came from a different paper | Checked both papers: the gold answer is correct. The question was kept unchanged (changing a test after seeing results would be tuning to the test) and documented as failure #1. |
| A gold-page anchor for P06 was not found | P05, P06 and P08 store "fl"/"fi" as single ligature characters (ﬂ, ﬁ), so "flow" in the text is "ﬂow" | Fixed the anchor text. The ligatures remain in the index (see Limitations). |
| Printing chunk text failed with `UnicodeEncodeError` on Windows | The Windows console uses cp1252 | Write output to a UTF-8 file, or set `PYTHONIOENCODING=utf-8`. |

## Deployment

Not publicly deployed, to respect paper copyright and avoid uncontrolled API costs. The project runs locally.
A FastAPI service (`src/rag/app.py`) and a `Dockerfile` are included: the image contains the code and the
prebuilt index (no PDFs, no API key), rebuilds an in-memory Chroma at startup with no API calls, and has cost
guards (question length cap, a global rate limit of 20 requests per minute, an output token cap). It is ready
for Cloud Run with the key supplied as a secret.

## Limitations

- **Small evaluation set:** 32 questions written by one person. Differences of one or two questions
  (0.04–0.08) are within noise.
- **Page-level retrieval metrics:** Hit@k counts a page as found if any of its chunks is retrieved, so it can
  overstate retrieval (q17).
- **Text-only parsing:** tables and equations in PDFs are extracted as plain text and often garbled.
  Scanned reports (P03 Goglia 1984, P20 and P21 OSTI) have OCR text with stray spaces and characters.
- **Ligatures:** P05, P06 and P08 store "fl"/"fi" as single characters. Keyword search only recognises a–z
  and 0–9, so "ﬂow" is indexed as "ow" and these papers never match "flow", "fluid" etc. by keyword.
- **LLM judge:** the judge is a model too. It agreed with me on 9/10 sampled answers, is slightly strict about
  extra gold-answer detail, and its faithfulness check penalises names repeated from the question.
- **Refusal sensitivity:** the answer model sometimes refuses when the answer is present, depending on the
  order and mix of passages.
- Papers are not redistributed here; `scripts/download_papers.py` fetches them from their open-access hosts.

## Future work

- Fix ligatures during ingest (Unicode NFKC normalisation) and re-embed, to help keyword search on P05/P06/P08.
- Add a reranker, or chunk-level gold labels, to catch misses like q17.
- Adjust the prompt to ask for all quantitative details, to reduce "correct but incomplete" answers.
- Run a no-retrieval comparison (the same questions answered by the plain model) to measure what RAG adds.
- Grow the question set and spot-check more answers for stronger conclusions.

## Layout

```
src/rag/       config, ingest, embeddings, embed_store, retrieve, llm, query, judge, evaluate, app, trace, pricing
scripts/       download_papers.py, locate_pages.py
data/          papers.csv (committed), papers/ (ignored), index/ (built)
eval/          questions.jsonl
results/       <tag>.json, <tag>.md, <tag>_progress.jsonl, baseline_spotcheck.csv
traces/        requests.jsonl (every request: retrieved chunk ids, timings, tokens, cost)
tests/         offline tests (fake embedder + stub LLM)
vortex_agent/  optional Google ADK agent (separate venv)
```
