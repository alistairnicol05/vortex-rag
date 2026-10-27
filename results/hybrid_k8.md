### Results: `hybrid_k8`

| Config | gen=gemini-3.1-flash-lite | embed=gemini-embedding-2 | chunk=1200 | overlap=200 | k=8 | hybrid=True |
|---|---|

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | 26 / 6 |
| Hit@1 / Hit@3 / Hit@5 / Hit@10 | 0.462 / 0.846 / 1.000 / 1.000 |
| MRR | 0.684 |
| Multi-doc source recall@k | 1.000 |
| Answer correctness (0-2, LLM judge) | 1.58 |
| Fully correct rate | 0.65 |
| Citation faithfulness (LLM judge) | 0.80 |
| Correct refusal rate (unanswerable) | 1.00 |
| False refusal rate (answerable) | 0.04 |
| Invalid-citation rate | 0.00 |
| Latency p50 / p95 (s) | 1.33 / 1.64 |
| Time to first token p50 (s) | 0.83 |
| Decode tokens/s (median) | 356.2 |
| Tokens in / out (mean) | 2616 / 60 |
| Cost per request, paid rate (USD) | 0.000744 |
| Eval run total: answers + judge (USD) | 0.0238 + 0.1014 = 0.1252 |
