"""Indexing > Step 4 — chunks + embeddings ko FAISS me store karna (disk cache ke saath)."""

import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from Indexing.chunks_to_vectors import get_embeddings

FAISS_STORE_DIR = Path(__file__).resolve().parent.parent / "faiss_store"


def _safe_key(key: str) -> str:
    """Make a filesystem-safe folder name from an input key."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-.")


def store_path(key: str) -> Path:
    """Folder where this source's FAISS index lives."""
    return FAISS_STORE_DIR / _safe_key(key)


def index_exists(key: str) -> bool:
    """True if a complete saved index already exists on disk."""
    folder = store_path(key)
    return (folder / "index.faiss").is_file() and (folder / "index.pkl").is_file()


def build_or_load(key: str, chunks: list[Document] | None = None) -> FAISS:
    """
    Load a source's saved index if it exists, else embed the chunks and save it.

    `key` is the deterministic identifier from load_input(), so re-uploading the
    same input never re-embeds it — the on-disk index is reused.
    """
    embeddings = get_embeddings()

    if index_exists(key):
        return FAISS.load_local(
            str(store_path(key)),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not chunks:
        raise ValueError("No chunks provided and no saved index found for this source.")

    FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_path(key)))
    return vectorstore


def merge_stores(stores: list[FAISS]) -> FAISS | None:
    """Union several per-source stores in memory (no re-embedding involved)."""
    if not stores:
        return None
    merged = stores[0]
    for extra in stores[1:]:
        merged.merge_from(extra)
    return merged
