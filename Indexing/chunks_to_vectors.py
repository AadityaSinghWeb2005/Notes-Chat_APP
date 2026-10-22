"""Indexing > Step 3 — chunks ko local embeddings me convert karna.

Model: embeddinggemma-2:270m (Ollama, fully local — no API key needed).
"""

from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "embeddinggemma-2:270m"


def get_embeddings() -> OllamaEmbeddings:
    """Return the local Ollama embedding model used for chunks and queries."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
