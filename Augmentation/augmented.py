"""
Augmentation — retrieved context + prompt template ko combine karna.

Yahan se jo prompt tayyar hota hai wahi Generation stage ko diya jata hai.
"""

from langchain_core.documents import Document

PROMPT_TEMPLATE = """You are a helpful note-making assistant.

Instructions:
- Answer the user's question using ONLY the context below (retrieved from the user's sources).
- If the question asks for detailed notes, a summary, or important points, structure the answer with clear Markdown headings and bullet points.
- If the answer is not in the context, say that it could not be found in the provided sources.
- Answer in the same language the question was asked in (English, Hindi, Hinglish, etc.).

Context:
--------
{context}
--------

Question: {question}

Answer:
"""


def build_augmented_prompt(question: str, documents: list[Document]) -> str:
    """Context (top retrieved chunks) + template -> final prompt for the LLM."""
    context = "\n\n---\n\n".join(document.page_content for document in documents)
    return PROMPT_TEMPLATE.format(context=context, question=question)
