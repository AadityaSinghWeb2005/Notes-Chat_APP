"""
Generation — final prompt ko LLM ko bhejkar answer generate karna.

Model : gpt-oss:120b-cloud (Ollama cloud)
Key   : OLLAMA_API_KEY from .env (local Ollama daemon auth is used as fallback)
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

GENERATION_MODEL = "gpt-oss:120b-cloud"


def _llm() -> ChatOllama:
    return ChatOllama(
        model=GENERATION_MODEL,
        temperature=0.3,
        num_ctx=8192,
    )


def stream_answer(prompt: str):
    """Yield answer chunks as they arrive (used with Streamlit's st.write_stream)."""
    for chunk in _llm().stream(prompt):
        if chunk.content:
            yield chunk.content
