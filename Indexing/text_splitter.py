"""Indexing > Step 2 — loaded Documents ko overlapping chunks me todna."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


def split_into_chunks(documents: list[Document]) -> list[Document]:
    """Recursively split documents into chunks suitable for embedding."""
    return _splitter.split_documents(documents)
