### Results: `hybrid`

| Config | gen=gemini-3.1-flash-lite | embed=gemini-embedding-2 | chunk=1200 | overlap=200 | k=5 | hybrid=True |
|---|---|

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | 26 / 6 |
| Hit@1 / Hit@3 / Hit@5 / Hit@10 | 0.462 / 0.846 / 1.000 / 1.000 |
| MRR | 0.684 |
| Multi-doc source recall@k | 1.000 |
| Answer correctness (0-2, LLM judge) | 1.54 |
| Fully correct rate | 0.62 |
| Citation faithfulness (LLM judge) | 0.88 |
| Correct refusal rate (unanswerable) | 1.00 |
| False refusal rate (answerable) | 0.08 |
| Invalid-citation rate | 0.00 |
| Latency p50 / p95 (s) | 1.25 / 1.65 |
| Time to first token p50 (s) | 0.85 |
| Decode tokens/s (median) | 390.4 |
| Tokens in / out (mean) | 1662 / 50 |
| Cost per request, paid rate (USD) | 0.000490 |
| Eval run total: answers + judge (USD) | 0.0157 + 0.0899 = 0.1056 |
