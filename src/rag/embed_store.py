"""Embed chunks and load them into Chroma. Also exports/loads a portable index (chunks + vectors)
so the Cloud Run container can rebuild Chroma at startup with NO API calls.

Usage:
    python -m rag.embed_store                 # embed with Gemini, write data/index/, fill ./chroma_db
    python -m rag.embed_store --fake          # offline smoke test, fake embedder
    python -m rag.embed_store --collection chunk800 --size 800 --overlap 100   # ablation
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from .config import Settings, get_settings
from .embeddings import Embedder, FakeEmbedder, GeminiEmbedder
from .ingest import Chunk, ingest_dir, read_chunks, write_chunks


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    page: int
    text: str
    score: float  # cosine similarity (1 - cosine distance)


def save_index(index_dir: Path, chunks: list[Chunk], vectors: np.ndarray, model: str) -> None:
    write_chunks(chunks, index_dir / "chunks.jsonl")
    np.save(index_dir / "embeddings.npy", vectors)
    (index_dir / "meta.json").write_text(
        json.dumps({"embed_model": model, "dim": int(vectors.shape[1]), "n": len(chunks)}), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple[list[Chunk], np.ndarray, dict]:
    chunks = read_chunks(index_dir / "chunks.jsonl")
    vectors = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    if len(chunks) != len(vectors):
        raise ValueError("chunks.jsonl and embeddings.npy are out of sync; re-run embed_store")
    return chunks, vectors, meta


class VectorStore:
    """Thin wrapper over one Chroma collection (cosine space)."""

    def __init__(self, client: chromadb.ClientAPI, name: str) -> None:
        self.collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 500) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.collection.upsert(
                ids=[c.id for c in part],
                embeddings=vectors[i : i + batch].tolist(),
                documents=[c.text for c in part],
                metadatas=[{"source": c.source, "page": c.page, "chunk_index": c.chunk_index} for c in part],
            )

    def query(self, vector: np.ndarray, k: int) -> list[Hit]:
        res = self.collection.query(query_embeddings=[vector.tolist()], n_results=k)
        hits: list[Hit] = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(id=id_, source=meta["source"], page=int(meta["page"]), text=doc, score=1.0 - float(dist)))
        return hits

    def count(self) -> int:
        return self.collection.count()


def open_persistent(settings: Settings, collection: str | None = None) -> VectorStore:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return VectorStore(client, collection or settings.collection)


def open_from_index(index_dir: Path, collection: str = "papers") -> tuple[VectorStore, list[Chunk], dict]:
    """In-memory Chroma rebuilt from the exported index. Used by the web app (no API calls)."""
    chunks, vectors, meta = load_index(index_dir)
    store = VectorStore(chromadb.EphemeralClient(), collection)
    store.add(chunks, vectors)
    return store, chunks, meta


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use the offline fake embedder")
    ap.add_argument("--collection", default=s.collection)
    ap.add_argument("--size", type=int, default=s.chunk_chars)
    ap.add_argument("--overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--papers", type=Path, default=s.papers_dir)
    ap.add_argument("--index-dir", type=Path, default=s.index_dir)
    args = ap.parse_args()

    chunks = ingest_dir(args.papers, args.size, args.overlap, s.min_chunk_chars)
    embedder: Embedder = FakeEmbedder() if args.fake else GeminiEmbedder(s)
    print(f"Embedding {len(chunks)} chunks ...")
    vectors = embedder.embed_documents([c.text for c in chunks])
    model = "fake" if args.fake else s.embed_model
    save_index(args.index_dir, chunks, vectors, model)

    store = open_persistent(s, args.collection)
    store.add(chunks, vectors)
    print(f"Chroma collection '{args.collection}' now holds {store.count()} chunks (model={model}).")


if __name__ == "__main__":
    main()
