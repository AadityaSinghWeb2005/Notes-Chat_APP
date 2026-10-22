# 🗒️ Notes-RAG

Turn your documents and YouTube videos into **detailed notes, summaries, and important points** — then ask follow-up questions and get answers grounded in your own sources.

A Retrieval-Augmented Generation (RAG) app built with **Streamlit**, **LangChain**, **FAISS**, and **Ollama**.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, CSV, JSON, XML, XLSX/XLS
- ▶️ **YouTube transcripts** — paste a video link, it fetches the captions
- 🔍 **Smart retrieval** — Multi-Query expansion + MMR (Maximal Marginal Relevance) for diverse, relevant chunks
- 💬 **Grounded chat** — answers only from your indexed sources, with the retrieved context shown behind an expander
- 🧠 **Quick actions** — one-click "Detailed Notes", "Summary", and "Important Points"
- 💾 **On-disk FAISS cache** — re-uploading the same file reuses its existing index (no re-embedding)
- 🌐 **Multilingual answers** — replies in the language the question was asked in (English, Hindi, Hinglish, …)

---

## 🏗️ Architecture

```
                ┌─────────────────────────────────────────────┐
                │                 Streamlit UI                │
                └──────────────────────┬──────────────────────┘
                                       │
   ┌───────────────────────────────────┼───────────────────────────────────┐
   │                                   │                                   │
   ▼                                   ▼                                   ▼
┌──────────────┐              ┌──────────────────┐              ┌──────────────────┐
│  Indexing    │              │    Retrieval     │              │   Augmentation   │
│              │              │                  │              │                  │
│ load_input   │              │ Multi-Query      │              │ context + prompt │
│ split_chunks │─────────────▶│      +           │─────────────▶│    template      │
│ embed + FAISS│   FAISS DB   │ MMR search       │   top chunks │                  │
└──────────────┘              └──────────────────┘              └────────┬─────────┘
                                                                          │
                                                                          ▼
                                                                ┌──────────────────┐
                                                                │   Generation     │
                                                                │ Ollama chat model│
                                                                │  (streamed out)  │
                                                                └──────────────────┘
```

### Pipeline

1. **Indexing** (`Indexing/`)
   - `input_loader.py` — loads files / YouTube URLs into LangChain `Document`s and creates a deterministic cache key
   - `text_splitter.py` — splits documents into overlapping chunks
   - `chunks_to_vectors.py` — embeds chunks with a local Ollama embedding model
   - `FAISS_DB.py` — builds/loads per-source FAISS indexes from `faiss_store/` and merges them in memory
2. **Retrieval** (`Retrieval/`)
   - `custom_retriever_function.py` — generates query variants with an LLM, then runs MMR search for diverse top chunks
   - `retriever.py` — single entry point: question → relevant documents
3. **Augmentation** (`Augmentation/`)
   - `augmented.py` — combines retrieved context with the answer prompt
4. **Generation** (`Generation/`)
   - `llm.py` — streams the final answer from the Ollama chat model

---

## 🧰 Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running
- An **Ollama API key** for the cloud models (get one at <https://ollama.com/settings/keys>) — or sign in with `ollama signin`

---

## 🚀 Setup

```bash
# 1. Clone / enter the project
cd Notes-RAG

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local embedding model
ollama pull embeddinggemma-2:270m
```

### Configure `.env`

Create a `.env` file in the project root:

```env
OLLAMA_API_KEY=your_ollama_api_key_here
```

> The key is optional if your local Ollama daemon is already authenticated (`ollama signin`). It is required for the `-cloud` models when the daemon is not signed in.

### Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>).

---

## 🖥️ Usage

1. In the sidebar, upload one or more documents **or** paste a YouTube link.
2. Click **🔄 Build / Refresh Index** and wait for the success message.
3. Ask a question in the chat box, or use a quick action:
   - 📝 Detailed Notes
   - 📌 Summary
   - ⭐ Important Points
4. Expand **🔍 Retrieved context** under an answer to inspect the source chunks.

---

## 🧠 Models Used

| Stage | Model | Where it runs |
| --- | --- | --- |
| Embeddings | `embeddinggemma-2:270m` | Local Ollama daemon |
| Multi-Query expansion | `gemma4:31b-cloud` | Ollama cloud |
| Answer generation | `gpt-oss:120b-cloud` | Ollama cloud |

Change these in `Indexing/chunks_to_vectors.py`, `Retrieval/custom_retriever_function.py`, and `Generation/llm.py`.

---

## 📁 Project Structure

```
Notes-RAG/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env                            # OLLAMA_API_KEY (git-ignored)
├── faiss_store/                    # Cached per-source FAISS indexes (git-ignored)
├── Augmentation/
│   └── augmented.py                # Context + prompt template
├── Generation/
│   └── llm.py                      # Streamed answer generation
├── Indexing/
│   ├── input_loader.py             # File / YouTube loading
│   ├── text_splitter.py            # Chunking
│   ├── chunks_to_vectors.py        # Embeddings
│   └── FAISS_DB.py                 # FAISS build / load / merge
└── Retrieval/
    ├── custom_retriever_function.py  # Multi-Query + MMR
    └── retriever.py                  # Query -> documents
```

---

## ⚠️ Notes & Troubleshooting

- **"No readable text found"** — the file may be scanned/image-only. OCR is not included.
- **"This video has no transcript available"** — the YouTube video has no captions; try another video.
- **Cloud model errors** — make sure Ollama is running and `OLLAMA_API_KEY` is valid, or run `ollama signin`.
- **First indexing is slow** — the embedding model runs locally and embeds every chunk on the first run. Repeat runs reuse `faiss_store/` and are much faster.
- **Deploying to Streamlit Cloud** — chat models work with the API key, but **embeddings must reach an Ollama daemon**. Since Ollama Cloud does not serve embedding models, you need a self-hosted Ollama endpoint reachable from the app for indexing to work.

---

## 🗺️ Roadmap Ideas

- Re-ranking retrieved chunks
- Source citation links in answers
- Conversation memory / follow-up query rewriting
- Optional hosted embedding provider for serverless deployments
- Upload → auto-index without pressing a button

