"""
Notes-RAG — Streamlit entry point.

Sidebar se documents / YouTube link index karo, phir chat me
detailed notes, summary, important points ya koi bhi custom query puchho.

Pipeline: Indexing -> Retrieval (Multi-Query + MMR) -> Augmentation -> Generation
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from dotenv import load_dotenv

from Augmentation.augmented import build_augmented_prompt
from Generation.llm import stream_answer
from Indexing.FAISS_DB import build_or_load, merge_stores
from Indexing.input_loader import SUPPORTED_EXTENSIONS, load_input
from Indexing.text_splitter import split_into_chunks
from Retrieval.retriever import get_relevant_documents

load_dotenv()

st.set_page_config(page_title="Notes-RAG", page_icon="🗒️", layout="wide")

QUICK_ACTIONS = {
    "📝 Detailed Notes": "Make detailed, well-structured notes of the provided source.",
    "📌 Summary": "Summarize the provided source concisely, covering all key points.",
    "⭐ Important Points": "List the most important points from the provided source.",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed" not in st.session_state:
    st.session_state.indexed = {}
if "build_log" not in st.session_state:
    st.session_state.build_log = []


def build_index(uploaded_files, youtube_url) -> list[tuple[str, str]]:
    """Load sidebar inputs, then build / refresh the FAISS index for this session."""
    desired: dict[str, tuple[str, list]] = {}
    log: list[tuple[str, str]] = []

    for uploaded in uploaded_files or []:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            log.append(("warning", f"⚠️ {uploaded.name}: unsupported format."))
            continue
        try:
            with TemporaryDirectory() as tmpdir:
                file_path = Path(tmpdir) / uploaded.name
                file_path.write_bytes(bytes(uploaded.getbuffer()))
                loaded = load_input(str(file_path))
            desired[loaded["key"]] = (
                loaded["label"],
                split_into_chunks(loaded["docs"]),
            )
            log.append(("info", f"📄 {uploaded.name}: loaded."))
        except Exception as exc:  # surface any loader error to the user
            log.append(("error", f"❌ {uploaded.name}: {exc}"))

    if youtube_url and youtube_url.strip():
        try:
            loaded = load_input(youtube_url.strip())
            desired[loaded["key"]] = (
                loaded["label"],
                split_into_chunks(loaded["docs"]),
            )
            log.append(("info", f"▶️ {loaded['label']}: transcript loaded."))
        except Exception as exc:
            log.append(("error", f"❌ YouTube link: {exc}"))

    if not desired:
        if not log:
            log.append(("warning", "Add at least one document or a YouTube link first."))
        return log

    if (
        set(desired) == set(st.session_state.indexed)
        and st.session_state.vectorstore is not None
    ):
        log.append(("info", "Index already up to date — nothing to rebuild."))
        return log

    try:
        with st.spinner("Building FAISS index (first run embeds chunks locally)…"):
            stores = [build_or_load(key, desired[key][1]) for key in desired]
            st.session_state.vectorstore = merge_stores(stores)
        st.session_state.indexed = {key: desired[key][0] for key in desired}
        log.append(("success", f"✅ Index ready — {len(desired)} source(s)."))
    except Exception as exc:
        log.append(("error", f"❌ Indexing failed: {exc}"))

    return log


def render_context(documents) -> None:
    """Show the retrieved chunks behind an answer."""
    if not documents:
        return
    with st.expander("🔍 Retrieved context (top chunks)"):
        for index, document in enumerate(documents, 1):
            source = document.metadata.get("source", "unknown source")
            page = document.metadata.get("page")
            header = f"**Chunk {index}** — `{source}`"
            if isinstance(page, int):
                header += f" (page {page + 1})"
            st.markdown(header)
            content = document.page_content
            st.code(content[:800] + ("…" if len(content) > 800 else ""), language=None)


def answer_question(question: str) -> None:
    """Retrieve -> augment -> generate, rendering the answer as it streams."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    context = []
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            reply = (
                "**Pehle koi source add kijiye.** Sidebar se document upload karein "
                "ya YouTube link daalein, phir *Build / Refresh Index* dabaiye."
            )
            st.markdown(reply)
        else:
            try:
                with st.spinner("Retrieving relevant context…"):
                    context = get_relevant_documents(
                        st.session_state.vectorstore, question
                    )

                if not context:
                    reply = (
                        "Mujhe indexed sources me is question ka relevant context nahi "
                        "mila. Thoda rephrase karke puchhiye."
                    )
                    st.markdown(reply)
                else:
                    prompt = build_augmented_prompt(question, context)
                    reply = st.write_stream(stream_answer(prompt))
                    if not reply:
                        reply = "⚠️ The model returned an empty response. Please try again."
                        st.markdown(reply)
                    render_context(context)
            except Exception as exc:
                reply = f"❌ {exc}"
                st.error(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "context": context}
    )


# --------------------------------------------------------------------------- #
# Sidebar — sources & index
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📥 Sources")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, CSV, JSON, XML, XLSX, XLS",
    )
    youtube_url = st.text_input(
        "YouTube link",
        placeholder="https://www.youtube.com/watch?v=…",
        help="The video must have captions / a transcript available.",
    )

    if st.button("🔄 Build / Refresh Index", type="primary"):
        st.session_state.build_log = build_index(uploaded_files, youtube_url)

    for level, message in st.session_state.build_log[-8:]:
        {
            "success": st.success,
            "info": st.info,
            "warning": st.warning,
            "error": st.error,
        }[level](message)

    st.divider()

    if st.session_state.indexed:
        st.caption(f"**{len(st.session_state.indexed)} source(s) indexed**")
        for label in st.session_state.indexed.values():
            st.caption(f"• {label}")
    else:
        st.caption("No index yet — add a source and hit **Build / Refresh Index**.")

    st.divider()
    if os.getenv("OLLAMA_API_KEY"):
        st.caption("🔑 OLLAMA_API_KEY loaded from .env")
    else:
        st.caption("ℹ️ Using local Ollama daemon auth (no OLLAMA_API_KEY set)")

# --------------------------------------------------------------------------- #
# Main area — chat
# --------------------------------------------------------------------------- #
st.title("🗒️ Notes-RAG")
st.caption(
    "Documents ya YouTube links se detailed notes, summaries aur important "
    "points — sab kuch ek jagah."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("context"):
            render_context(message["context"])

chip_columns = st.columns(len(QUICK_ACTIONS))
clicked_question = None
for column, (label, preset) in zip(chip_columns, QUICK_ACTIONS.items()):
    if column.button(label, key=f"chip-{label}"):
        clicked_question = preset

typed_question = st.chat_input("Ask anything about your sources…")
question = typed_question or clicked_question
if question:
    answer_question(question)
