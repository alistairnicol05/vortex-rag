"""Per-1M-token prices in USD (standard tier, paid).

Source: https://ai.google.dev/gemini-api/docs/pricing, read 2026-10-19.
Prices change. Re-check before you publish a cost number, and write the date next to it.
The free tier charges $0, so report cost at the PAID rate to get a meaningful number.
Thinking tokens are billed as output tokens.
"""
from __future__ import annotations

# (input_usd_per_1M, output_usd_per_1M)
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Introductory rate through 2026-12-31; doubles on 2027-01-01 to (1.50, 7.50).
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
}

# Embedding price per 1M input tokens: NOT verified from the official page. Left at None on purpose.
EMBED_PRICE_PER_M: dict[str, float | None] = {
    "gemini-embedding-001": None,
    "gemini-embedding-2": None,
}


def llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return cost in USD, or None if the model's price is not in the table."""
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
