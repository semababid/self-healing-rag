"""Answers one question and returns the result as plain data.

The command line, the API and the web page all go through `answer`, so they
show the same thing.
"""

import time
from pathlib import Path

from src import config
from src.graph import answer_question
from src.llm import check_ollama
from src.logs import save_run
from src.nodes import cited_documents
from src.vectorstore import require_store_built


def problem() -> str | None:
    """What stops the system from answering right now, or None if it's ready."""
    try:
        check_ollama()
        require_store_built()
    except RuntimeError as err:
        return str(err)
    return None


def answer(question: str, source: str = "cli") -> dict:
    start = time.perf_counter()
    state = answer_question(question)
    seconds = time.perf_counter() - start
    save_run(question, state, seconds, source=source)

    documents = state.get("documents", [])
    cited = cited_documents(state.get("generation", ""), documents) if state.get("grounded") else []
    # A grounded answer without citations still shows what it was based on.
    shown = cited or (documents if state.get("grounded") else [])
    return {
        "question": question,
        "answer": state.get("generation", ""),
        "grounded": bool(state.get("grounded")),
        "retries": state.get("retries", 0),
        "seconds": round(seconds, 2),
        "sources_cited": bool(cited),
        "sources": [
            {
                "number": documents.index(doc) + 1,
                "file": Path(doc.metadata.get("source", "?")).name,
                "text": doc.page_content,
            }
            for doc in shown
        ],
        "attempts": state.get("attempts", []),
        "timings": state.get("timings", []),
    }


SUPPORTED = (".md", ".txt", ".pdf")


def add_document(name: str, data: bytes) -> Path:
    """Save an uploaded file to UPLOAD_DIR. Run ingest() afterwards to search it."""
    name = Path(name).name  # drop any folder parts, e.g. "../../x.md"
    if not name or name.startswith("."):
        raise ValueError(f"Can't add {name or 'a file without a name'}: hidden files aren't indexed")
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError(f"Can't add {name}: only .md, .txt and .pdf files are supported")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    return path


def remove_uploads() -> int:
    """Delete every uploaded file. Returns how many were removed."""
    if not config.UPLOAD_DIR.exists():
        return 0
    files = [p for p in config.UPLOAD_DIR.iterdir() if p.is_file()]
    for path in files:
        path.unlink()
    return len(files)
