"""Builds the vector store from the files in data/sample_docs.

Run it once, and again whenever the documents change:

    python -m src.ingest

The first run downloads the embedding model; after that it works offline.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src import config
from src.vectorstore import get_vectorstore


def title_of(text: str, path: Path) -> str:
    """The first Markdown heading, or the file name if there is none."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text,
                         metadata={"source": str(path), "title": title_of(text, path)})]
    if suffix == ".pdf":
        pages = PdfReader(path).pages
        return [Document(page_content=text,
                         metadata={"source": str(path), "title": path.stem, "page": n})
                for n, page in enumerate(pages)
                if (text := page.extract_text() or "").strip()]
    print(f"  skipping {path.name} (unsupported file type)")
    return []


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(config.DATA_DIR.rglob("*")):
        if path.is_file() and not path.name.startswith("."):  # skips .DS_Store
            docs.extend(load_file(path))
    if not docs:
        raise RuntimeError(f"No .md, .txt or .pdf files found in {config.DATA_DIR}")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        # A chunk from the middle of a document doesn't say what the document is
        # about, e.g. which product a list of specs belongs to. Without that, the
        # model mixed up the N7 and N9. So every chunk starts with the title.
        title = chunk.metadata["title"]
        if title not in chunk.page_content.split("\n", 1)[0]:
            chunk.page_content = f"{title}\n\n{chunk.page_content}"
    return chunks


def ingest() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"{len(docs)} documents, {len(chunks)} chunks")
    print("Embedding them (the first run downloads the embedding model, about 90 MB) ...")

    # Start from an empty collection so running this twice doesn't duplicate chunks.
    store = get_vectorstore()
    old_ids = store.get(include=[])["ids"]
    if old_ids:
        store.delete(ids=old_ids)

    store.add_documents(chunks)
    print(f"Saved to {config.CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
