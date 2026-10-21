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


@dataclass(frozen=True)  # one piece of a paper, plus where it came from (needed for citations)
class Chunk:
    id: str  # unique name, e.g. "P01_Kristiansen2016.pdf:p3:c1"
    source: str  # PDF file name
    page: int  # 1-indexed
    chunk_index: int  # 0 = first chunk on this page, 1 = second, ...
    text: str


def clean_text(text: str) -> str: # prepare for embedding
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line breaks ("veloc-\nity" -> "velocity")
    text = re.sub(r"[ \t]+", " ", text) # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text) # collapse multiple blank lines
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy character windows that prefer to end on a sentence or newline boundary."""
    if overlap >= size:  # otherwise the window would never move forward
        raise ValueError("overlap must be smaller than chunk size")
    out: list[str] = []
    start = 0  # where the current chunk begins
    n = len(text)
    while start < n:
        end = min(start + size, n)  # tentative end: size characters later, or end of text
        if end < n:  # not the last chunk, so try to end on a clean boundary
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))  # last full stop or newline in the window
            if cut > size * 0.6:  # only snap if it does not shrink the chunk too much
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:  # reached the end of the page
            break
        start = max(end - overlap, start + 1)  # step back by overlap so neighbours share text; always move >= 1
    return out


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] for one PDF."""
    pages: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:  # "with" closes the file automatically
        for i, page in enumerate(doc.pages(), start=1):  # count pages from 1, like a PDF viewer
            pages.append((i, clean_text(page.get_text("text"))))  # plain text of this page only
    return pages


def chunk_pdf(pdf_path: Path, size: int, overlap: int, min_chars: int) -> list[Chunk]:
    """Split each page separately, so a chunk never spans two pages."""
    chunks: list[Chunk] = []
    for page_no, text in extract_pages(pdf_path):
        for j, piece in enumerate(split_text(text, size, overlap)):  # j = chunk number on this page
            if len(piece) < min_chars:  # skip tiny scraps
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
    """Chunk every PDF in the folder and print a one-line report per file."""
    pdfs = sorted(papers_dir.glob("*.pdf"))  # sorted = same order every run
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {papers_dir}. Run scripts/download_papers.py first.")
    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        try:
            c = chunk_pdf(pdf, size, overlap, min_chars)
        except Exception as exc:  # a corrupt PDF doesn't stop the batch
            print(f"[skip] {pdf.name}: {exc}")
            continue
        if not c:
            print(f"[warn] {pdf.name}: no text extracted (scanned? needs OCR)")
        all_chunks.extend(c)
        print(f"[ok]   {pdf.name}: {len(c)} chunks")
    return all_chunks


def write_chunks(chunks: list[Chunk], out: Path) -> None:
    """Save chunks as JSONL: one JSON object per line."""
    out.parent.mkdir(parents=True, exist_ok=True)  # create the folder if it doesn't exist
    with out.open("w", encoding="utf-8") as f: # write text in UTF-8 so symbols like "°" are preserved
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")  # ensure_ascii=False keeps symbols like "°" readable


def read_chunks(path: Path) -> list[Chunk]:
    """Load chunks back from a JSONL file (the reverse of write_chunks)."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]  # skip blank lines


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()  # command-line options; defaults come from config.py
    ap.add_argument("--papers", type=Path, default=s.papers_dir) # folder of PDFs to chunk
    ap.add_argument("--out", type=Path, default=s.index_dir / "chunks.jsonl") # where to save the chunks
    ap.add_argument("--size", type=int, default=s.chunk_chars) # max characters per chunk
    ap.add_argument("--overlap", type=int, default=s.chunk_overlap) # characters shared by neighbouring chunks
    args = ap.parse_args() # parse the command-line arguments
    chunks = ingest_dir(args.papers, args.size, args.overlap, s.min_chunk_chars) # chunk the PDFs
    write_chunks(chunks, args.out) # save the chunks to a JSONL file
    print(f"Wrote {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":  # only runs via "python -m rag.ingest", not when imported
    main()
