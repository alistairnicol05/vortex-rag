### Results: `baseline`

| Config | gen=gemini-3.1-flash-lite | embed=gemini-embedding-2 | chunk=1200 | overlap=200 | k=5 | hybrid=False |
|---|---|

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | 26 / 6 |
| Hit@1 / Hit@3 / Hit@5 / Hit@10 | 0.577 / 0.769 / 0.885 / 0.962 |
| MRR | 0.693 |
| Multi-doc source recall@k | 0.917 |
| Answer correctness (0-2, LLM judge) | 1.54 |
| Fully correct rate | 0.73 |
| Citation faithfulness (LLM judge) | 0.95 |
| Correct refusal rate (unanswerable) | 1.00 |
| False refusal rate (answerable) | 0.15 |
| Invalid-citation rate | 0.00 |
| Latency p50 / p95 (s) | 1.27 / 1.59 |
| Time to first token p50 (s) | 0.87 |
| Decode tokens/s (median) | 360.3 |
| Tokens in / out (mean) | 1684 / 40 |
| Cost per request, paid rate (USD) | 0.000481 |
| Eval run total: answers + judge (USD) | 0.0154 + 0.0665 = 0.0819 |
