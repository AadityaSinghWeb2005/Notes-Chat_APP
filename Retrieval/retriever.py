"""Retrieval > Step 2 — app ke liye single entry point: query -> top chunks."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from Retrieval.custom_retriever_function import retrieve_with_mmr


def get_relevant_documents(vectorstore: FAISS | None, question: str) -> list[Document]:
    """Fetch the most relevant chunks for a user question."""
    if vectorstore is None or not question.strip():
        return []
    return retrieve_with_mmr(vectorstore, question)
