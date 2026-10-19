"""PDF -> page-tagged chunks. Chunks never cross a page boundary, so (source, page) is exact.

Usage:
    python -m rag.ingest --papers data/papers --out data/index/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from .config import get_settings


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # PDF file name
    page: int  # 1-indexed
    chunk_index: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy character windows that prefer to end on a sentence or newline boundary."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.6:  # only snap if it does not shrink the chunk too much
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, clean_text(page.get_text("text"))))
    return pages


def chunk_pdf(pdf_path: Path, size: int, overlap: int, min_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in extract_pages(pdf_path):
        for j, piece in enumerate(split_text(text, size, overlap)):
            if len(piece) < min_chars:
                continue
            chunks.append(
                Chunk(
                    id=f"{pdf_path.name}:p{page_no}:c{j}",
                    source=pdf_path.name,
                    page=page_no,
                    chunk_index=j,
                    text=piece,
                )
            )
    return chunks


def ingest_dir(papers_dir: Path, size: int, overlap: int, min_chars: int) -> list[Chunk]:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {papers_dir}. Run scripts/download_papers.py first.")
    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        try:
            c = chunk_pdf(pdf, size, overlap, min_chars)
        except Exception as exc:  # a corrupt PDF should not stop the batch
            print(f"[skip] {pdf.name}: {exc}")
            continue
        if not c:
            print(f"[warn] {pdf.name}: no text extracted (scanned? needs OCR)")
        all_chunks.extend(c)
        print(f"[ok]   {pdf.name}: {len(c)} chunks")
    return all_chunks


def write_chunks(chunks: list[Chunk], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=Path, default=s.papers_dir)
    ap.add_argument("--out", type=Path, default=s.index_dir / "chunks.jsonl")
    ap.add_argument("--size", type=int, default=s.chunk_chars)
    ap.add_argument("--overlap", type=int, default=s.chunk_overlap)
    args = ap.parse_args()
    chunks = ingest_dir(args.papers, args.size, args.overlap, s.min_chunk_chars)
    write_chunks(chunks, args.out)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
