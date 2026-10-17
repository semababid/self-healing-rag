"""Vector store + embeddings access.

Both the ingestion script and the retrieval node need to talk to the *same*
Chroma collection with the *same* embedding model, so that logic lives here in
one place.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

# Cache the embedding model at module level. Loading a sentence-transformers
# model takes a few seconds, and we don't want to pay that cost twice.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached local embedding model (no API key, no network cost)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection on disk."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def require_store_built() -> None:
    """Fail fast with a friendly message if ``python -m src.ingest`` hasn't run.

    Without this, an empty store silently retrieves nothing, the critic rejects
    every attempt, and the user gets "I don't know" after burning API calls.
    """
    if not get_vectorstore().get(limit=1, include=[])["ids"]:
        raise RuntimeError(
            "The vector store is empty. Build it first:\n"
            "    python -m src.ingest"
        )


def get_retriever(k: int | None = None):
    """Return a retriever that fetches the top-k most similar chunks."""
    k = k or config.RETRIEVER_K
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
