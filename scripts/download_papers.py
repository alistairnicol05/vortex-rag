"""Download the open-access PDFs listed in data/papers.csv into data/papers/.

Run LOCALLY (Windows PowerShell):
    $env:CONTACT_EMAIL = "you@example.com"      # goes into the User-Agent so site admins can reach you
    python scripts/download_papers.py

Polite by design: identifies itself, waits between requests, retries with backoff on 429/5xx,
never overwrites a good file, and verifies each download really is a PDF.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def make_session(contact: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = f"vortex-rag-student-project/0.1 (research use; contact: {contact})"
    s.headers["Accept"] = "application/pdf,*/*;q=0.8"
    retry = Retry(total=4, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download(session: requests.Session, url: str, dest: Path, timeout: int = 60) -> str:
    tmp = dest.with_suffix(".part")
    with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    if not is_pdf(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError("response is not a PDF (login/landing page or blocked?)")
    tmp.replace(dest)
    return f"{dest.stat().st_size / 1024:.0f} KB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "papers.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "papers")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = ap.parse_args()

    contact = os.getenv("CONTACT_EMAIL", "set-CONTACT_EMAIL-env-var")
    if contact.startswith("set-"):
        print("Tip: set CONTACT_EMAIL so the User-Agent includes a real contact address.\n")
    args.out.mkdir(parents=True, exist_ok=True)
    session = make_session(contact)

    ok, failed, manual = [], [], []
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        dest = args.out / row["file_name"]
        if dest.exists() and is_pdf(dest):
            print(f"[have] {row['file_name']}")
            ok.append(row["id"])
            continue
        if not row["pdf_url"]:
            manual.append(row)
            print(f"[manual] {row['id']}: no direct PDF URL; open {row['landing_url']}")
            continue
        try:
            size = download(session, row["pdf_url"], dest)
            print(f"[ok]   {row['file_name']} ({size})")
            ok.append(row["id"])
        except Exception as exc:
            print(f"[FAIL] {row['id']} {row['pdf_url']}\n       {exc}")
            failed.append(row)
        time.sleep(args.delay)

    print(f"\nDone: {len(ok)} ok, {len(failed)} failed, {len(manual)} need manual download.")
    for r in failed + manual:
        print(f"  - {r['id']}: save the PDF as {args.out / r['file_name']}  (landing page: {r['landing_url']})")
    print("Every PDF you cannot download is simply left out of the index; the pipeline still works with 15+ papers.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
