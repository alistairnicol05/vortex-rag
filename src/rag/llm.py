"""LLM wrappers: GeminiLLM (real, streaming, token + latency accounting) and StubLLM (offline)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import REFUSAL_TEXT, Settings


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens, which are billed as output
    latency_s: float  # request start -> last chunk
    ttft_s: float  # request start -> first chunk

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Output tokens divided by time after the first chunk. None if it cannot be measured."""
        decode = self.latency_s - self.ttft_s
        return self.output_tokens / decode if decode > 1e-6 and self.output_tokens else None


class LLM(Protocol):
    model: str

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> LLMResult: ...


def token_counts(usage) -> tuple[int, int]:
    """(input, output) tokens from a google-genai UsageMetadata. Output includes thinking tokens.

    The visible-output field is named `response_token_count` in google-genai 2.x (older SDKs used
    `candidates_token_count`), so both are tried. Falls back to total - prompt if neither is set.
    """
    if usage is None:
        return 0, 0
    g = lambda name: int(getattr(usage, name, 0) or 0)  # noqa: E731
    in_tok = g("prompt_token_count")
    out_tok = (g("response_token_count") or g("candidates_token_count")) + g("thoughts_token_count")
    if out_tok == 0 and g("total_token_count"):
        out_tok = max(g("total_token_count") - in_tok, 0)
    return in_tok, out_tok


class GeminiLLM:
    def __init__(self, settings: Settings, model: str | None = None, max_output_tokens: int = 1024) -> None:
        from google import genai

        if not settings.api_key:
            raise RuntimeError("Set GEMINI_API_KEY (see .env.example).")
        self._client = genai.Client(api_key=settings.api_key)
        self.model = model or settings.gen_model
        self.max_output_tokens = max_output_tokens

    # On error (e.g. 429 rate limit, 503 model overloaded): try up to 6 times, waiting 2s, 4s, 8s, 16s, 32s,
    # ~1 min in total, long enough to outlast a per-minute rate-limit window or a short demand spike.
    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> LLMResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        usage = None
        for chunk in self._client.models.generate_content_stream(model=self.model, contents=prompt, config=cfg):
            if ttft is None:
                ttft = time.perf_counter() - t0
            if chunk.text:
                parts.append(chunk.text)
            if chunk.usage_metadata is not None:
                usage = chunk.usage_metadata  # the last chunk carries the final totals
        latency = time.perf_counter() - t0
        in_tok, out_tok = token_counts(usage)
        return LLMResult("".join(parts).strip(), self.model, in_tok, out_tok, latency, ttft or latency)


class StubLLM:
    """Offline stand-in. Answers by quoting the first context passage, or refuses if none match.
    Behaviour is deterministic so tests can assert on it."""

    model = "stub"

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        if json_mode:  # acting as a judge
            text = json.dumps({"correctness": 2, "faithful": 1, "reasoning": "stub judge"})
        elif "[S1]" in prompt and "REFUSE_ME" not in prompt:
            m = re.search(r"\[S1\][^\n]*\n(.+)", prompt)
            snippet = (m.group(1) if m else "")[:160]
            text = f"{snippet} [S1]"
        else:
            text = REFUSAL_TEXT
        n_in = len(prompt.split())
        n_out = len(text.split())
        dt = time.perf_counter() - t0
        return LLMResult(text, self.model, n_in, n_out, dt + 0.01, dt + 0.001)


def parse_json(text: str) -> dict:
    """Parse model JSON, tolerating code fences and surrounding text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise
