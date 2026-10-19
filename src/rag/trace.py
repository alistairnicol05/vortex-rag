"""Per-request JSONL trace logger. One line per request; stages timed with a context manager."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class Trace:
    """Collects timings and fields for a single request, then appends one JSON line."""

    def __init__(self, path: Path | None, **fields: Any) -> None:
        self.path = path
        self.record: dict[str, Any] = {
            "request_id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "stages_s": {},
            **fields,
        }
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.record["stages_s"][name] = round(time.perf_counter() - t, 4)

    def set(self, **fields: Any) -> None:
        self.record.update(fields)

    def write(self) -> dict[str, Any]:
        self.record["total_s"] = round(time.perf_counter() - self._t0, 4)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        return self.record


def read_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
