"""
Indexing > Step 1 — user inputs ko LangChain Document objects me load karna.

Supported formats: PDF, DOCX, TXT, CSV, JSON, XML, XLSX/XLS, and YouTube links.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".xlsx",
    ".xls",
}

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def is_youtube_url(text: str) -> bool:
    """True if the string looks like a YouTube video URL."""
    return _YOUTUBE_RE.search(text) is not None


def _file_hash(path: Path) -> str:
    """Small content hash — same file + same content => same FAISS cache key."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


def _load_json(path: Path) -> list[Document]:
    """Flatten a JSON file into readable 'key: value' text (per top-level item)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    def flatten(node, prefix: str = "") -> list[str]:
        if isinstance(node, dict):
            lines: list[str] = []
            for key, value in node.items():
                lines.extend(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
            return lines
        if isinstance(node, list):
            lines = []
            for index, value in enumerate(node):
                lines.extend(flatten(value, f"{prefix}[{index}]"))
            return lines
        return [f"{prefix}: {node}" if prefix else str(node)]

    if isinstance(data, list):
        documents = []
        for index, item in enumerate(data):
            text = "\n".join(flatten(item))
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "item": index},
                    )
                )
        return documents

    text = "\n".join(flatten(data))
    return (
        [Document(page_content=text, metadata={"source": path.name})]
        if text.strip()
        else []
    )


def _load_xml(path: Path) -> list[Document]:
    """Read XML with the stdlib parser — one Document per top-level element."""
    tree = ET.parse(path)
    documents = []
    for index, child in enumerate(tree.getroot()):
        text = "\n".join(part.strip() for part in child.itertext() if part.strip())
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "element": index},
                )
            )
    if not documents:
        text = "\n".join(
            part.strip() for part in tree.getroot().itertext() if part.strip()
        )
        if text:
            documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def _load_excel(path: Path) -> list[Document]:
    """One Document per sheet, serialized as CSV text."""
    import pandas as pd

    documents = []
    for sheet_name, frame in pd.read_excel(path, sheet_name=None).items():
        text = f"Sheet: {sheet_name}\n{frame.to_csv(index=False)}"
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "sheet": str(sheet_name)},
            )
        )
    return documents


def _load_csv(path: Path) -> list[Document]:
    """LangChain CSVLoader with an encoding fallback, then pandas as last resort."""
    for encoding in ("utf-8", "latin-1"):
        try:
            documents = CSVLoader(file_path=str(path), encoding=encoding).load()
            if any(doc.page_content.strip() for doc in documents):
                return documents
        except Exception:
            continue

    import pandas as pd

    frame = pd.read_csv(path)
    return [
        Document(page_content=frame.to_csv(index=False), metadata={"source": path.name})
    ]


def _load_file(path: Path) -> list[Document]:
    """Dispatch a file to the right LangChain loader based on its extension."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        documents = Docx2txtLoader(str(path)).load()
    elif suffix == ".txt":
        documents = TextLoader(str(path), encoding="utf-8").load()
    elif suffix == ".csv":
        documents = _load_csv(path)
    elif suffix == ".json":
        documents = _load_json(path)
    elif suffix == ".xml":
        documents = _load_xml(path)
    elif suffix in (".xlsx", ".xls"):
        documents = _load_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")

    for document in documents:
        document.metadata["source"] = path.name

    return [doc for doc in documents if doc.page_content and doc.page_content.strip()]


def _load_youtube(url: str) -> list[Document]:
    """Load a YouTube video's transcript (English or Hindi captions)."""
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=["en", "hi"],
    )
    documents = loader.load()
    if not documents:
        raise ValueError(
            "This video has no transcript available. Try a video with captions."
        )
    return documents


def load_input(path_or_url: str) -> dict:
    """
    Load a single input (file path or YouTube URL).

    Returns a dict with:
        docs  — list[Document]
        key   — deterministic cache key (file name + content hash, or video id)
        label — human readable name shown in the UI
        type  — "file" or "youtube"
    """
    text = str(path_or_url).strip()

    if is_youtube_url(text):
        match = _YOUTUBE_RE.search(text)
        video_id = match.group(1) if match else text
        return {
            "docs": _load_youtube(text),
            "key": f"yt-{video_id}",
            "label": f"YouTube · {video_id}",
            "type": "youtube",
        }

    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")

    documents = _load_file(path)
    if not documents:
        raise ValueError("No readable text found in this file.")

    return {
        "docs": documents,
        "key": re.sub(r"[^A-Za-z0-9._-]+", "-", f"{suffix.lstrip('.')}-{path.name}-{_file_hash(path)}"),
        "label": path.name,
        "type": "file",
    }
