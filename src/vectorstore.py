"""Access to the Chroma vector store, shared by ingest.py and the retrieve step."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    # Loading the model takes a few seconds, so load it once.
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever(k: int | None = None):
    return get_vectorstore().as_retriever(search_kwargs={"k": k or config.RETRIEVER_K})


def require_store_built() -> None:
    """Raise RuntimeError if `python -m src.ingest` hasn't been run yet."""
    if not get_vectorstore().get(limit=1, include=[])["ids"]:
        raise RuntimeError("The vector store is empty. Build it first:\n    python -m src.ingest")
