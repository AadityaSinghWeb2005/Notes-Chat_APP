"""
Retrieval > Step 1 — custom retriever: Multi-Query + MMR.

- Multi-Query: gemma4:31b-cloud se user query ke alternatives banate hain
  (ambiguous queries ko better handle karne ke liye).
- MMR: FAISS se diverse top-k chunks nikalte hain taaki answer well-rounded ho.
"""

import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MULTI_QUERY_MODEL = "gemma4:31b-cloud"
TOP_K = 3          # final number of chunks handed to the LLM
MMR_FETCH_K = 20   # candidate pool MMR picks from
NUM_VARIANTS = 3   # extra query phrasings generated

_VARIANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search queries. Given a user question, produce {n} alternative "
            "phrasings that would retrieve the same information using different words "
            "or angles. Return only the queries, one per line, with no numbering and "
            "no extra text.",
        ),
        ("human", "{question}"),
    ]
)


def _query_variants(question: str) -> list[str]:
    """Original query + LLM-generated alternatives (falls back to the original only)."""
    try:
        raw = (
            _VARIANT_PROMPT
            | ChatOllama(model=MULTI_QUERY_MODEL, temperature=0.7)
            | StrOutputParser()
        ).invoke({"n": NUM_VARIANTS, "question": question})
    except Exception:
        # Cloud model unavailable / parsing issue — retrieval still works
        return [question]

    queries = [question]
    seen = {question.strip().lower()}

    for line in raw.splitlines():
        candidate = re.sub(r"^[\s\-*\d.)]+", "", line).strip().strip("\"'")
        if len(candidate) < 4 or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        queries.append(candidate)

    return queries[: NUM_VARIANTS + 1]


def retrieve_with_mmr(vectorstore, question: str) -> list[Document]:
    """
    Run Multi-Query expansion + MMR retrieval over the FAISS store.

    Every query variant is searched with MMR (diversity), results are
    de-duplicated, and the top 3 unique chunks are returned.
    """
    results: list[Document] = []
    seen_chunks: set[tuple[str, str]] = set()

    for query in _query_variants(question):
        for document in vectorstore.max_marginal_relevance_search(
            query, k=TOP_K, fetch_k=MMR_FETCH_K
        ):
            signature = (
                str(document.metadata.get("source", "")),
                document.page_content.strip()[:200],
            )
            if signature in seen_chunks:
                continue
            seen_chunks.add(signature)
            results.append(document)

    return results[:TOP_K]
