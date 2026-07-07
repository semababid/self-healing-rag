"""Build the vector store from the documents in ``data/sample_docs``.

Run this once (and again whenever the documents change):

    python -m src.ingest

Pipeline: load files -> split into overlapping chunks -> embed -> persist.
The embeddings are computed locally, so this works fully offline.
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.vectorstore import get_vectorstore

# Which file extensions we know how to read.
_TEXT_SUFFIXES = {".md", ".txt"}
_PDF_SUFFIXES = {".pdf"}


def load_documents() -> list[Document]:
    """Load every supported file under ``DATA_DIR`` into LangChain Documents."""
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(f"No document directory at {config.DATA_DIR}")

    docs: list[Document] = []
    for path in sorted(config.DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif suffix in _PDF_SUFFIXES:
            docs.extend(PyPDFLoader(str(path)).load())
        else:
            print(f"  · skipping unsupported file: {path.name}")

    if not docs:
        raise RuntimeError(
            f"No supported documents (.md/.txt/.pdf) found in {config.DATA_DIR}"
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for finer-grained retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def ingest() -> None:
    """Load, chunk, embed, and persist all sample documents into Chroma."""
    print(f"Loading documents from {config.DATA_DIR} ...")
    documents = load_documents()
    print(f"  loaded {len(documents)} document(s)")

    chunks = split_documents(documents)
    print(f"  split into {len(chunks)} chunk(s)")

    # Rebuild from scratch so re-running ingest never duplicates chunks.
    store = get_vectorstore()
    existing_ids = store.get().get("ids", [])
    if existing_ids:
        print(f"  clearing {len(existing_ids)} existing chunk(s)")
        store.delete(ids=existing_ids)

    print(f"Embedding with {config.EMBEDDING_MODEL} (first run downloads the model) ...")
    store.add_documents(chunks)
    print(f"Done. Vector store persisted at {config.CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
