"""Fill gold page numbers by searching the downloaded PDFs for each question's anchor phrase.

Usage (from the project root, venv active):
    python scripts\\locate_pages.py                       # eval\\questions_draft.jsonl -> eval\\questions.jsonl
    python scripts\\locate_pages.py --draft eval\\questions_draft.jsonl --out eval\\questions.jsonl --papers data\\papers

Pages written are 1-based PDF page indices (the first page of the file is 1), which is what the pipeline cites.
This can differ from the number printed on the page. Always open each PDF at the reported page and read the passage.

Matching ignores case, whitespace, soft hyphens and all hyphens (so line-break hyphenation never blocks a match). If an anchor is not found, or found on
several pages, the script says so; fix those by editing the anchor or setting "pages" by hand.
Unanswerable questions may carry "absent_terms": the script lists where each term occurs so you can confirm that
no paper actually answers the question.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
_SPACE = re.compile(r"\s+")
_TRANS = str.maketrans({"−": "-", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'})


def norm(text: str) -> str:
    text = text.replace("­", "").translate(_TRANS)
    text = text.replace("-", "")  # drops hyphens everywhere, so "azi-\nmuthal" and "length-to-\ndiameter" both match
    return _SPACE.sub("", text).lower()


def load_pages(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [norm(page.get_text()) for page in doc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=ROOT / "eval" / "questions_draft.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "questions.jsonl")
    ap.add_argument("--papers", type=Path, default=ROOT / "data" / "papers")
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.draft.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache: dict[str, list[str] | None] = {}

    def pages_of(name: str) -> list[str] | None:
        if name not in cache:
            path = args.papers / name
            cache[name] = load_pages(path) if path.exists() else None
        return cache[name]

    problems: list[str] = []
    for q in questions:
        for item in q["gold"]:
            texts = pages_of(item["source"])
            if texts is None:
                problems.append(f"{q['id']}: missing PDF {item['source']}")
                continue
            needle = norm(item["anchor"])
            found = [i + 1 for i, t in enumerate(texts) if needle in t]
            if not found:
                problems.append(f"{q['id']}: anchor not found in {item['source']}: {item['anchor']!r}")
                item["pages"] = []
            else:
                item["pages"] = [found[0]]
                if len(found) > 1:
                    problems.append(f"{q['id']}: anchor on several pages {found} in {item['source']}; used {found[0]}. Check the right one.")
            print(f"{q['id']:>4}  {item['source']:<26} pages={item['pages']}  anchor={item['anchor']!r}")

        for term in q.get("absent_terms", []):
            needle = norm(term)
            hits = []
            for pdf in sorted(args.papers.glob("*.pdf")):
                texts = pages_of(pdf.name) or []
                pg = [i + 1 for i, t in enumerate(texts) if needle in t]
                if pg:
                    hits.append(f"{pdf.name} p{pg[:5]}")
            status = "NOT FOUND in any paper (good)" if not hits else "FOUND, read these before keeping the question: " + "; ".join(hits)
            print(f"{q['id']:>4}  unanswerable check {term!r}: {status}")
            if hits:
                problems.append(f"{q['id']}: term {term!r} appears in {len(hits)} paper(s); confirm none answers the question")

    args.out.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out} ({len(questions)} questions)")
    if problems:
        print(f"\n{len(problems)} item(s) need your attention:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
