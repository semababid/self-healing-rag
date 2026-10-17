"""Access to the Chroma vector store, shared by ingest.py and the retrieve step."""

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

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


def require_store_built() -> None:
    """Raise RuntimeError if `python -m src.ingest` hasn't been run yet."""
    if not get_vectorstore().get(limit=1, include=[])["ids"]:
        raise RuntimeError("The vector store is empty. Build it first:\n    python -m src.ingest")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def search(query: str, k: int | None = None) -> list[Document]:
    """The k chunks that best match the query, best first.

    SEARCH=vector ranks by embedding similarity. SEARCH=hybrid also ranks by
    keyword overlap (BM25) and merges the two rankings with reciprocal rank
    fusion: each chunk scores 1 / (60 + rank) in each list, and the scores add up.
    """
    k = k or config.RETRIEVER_K
    store = get_vectorstore()
    if config.SEARCH == "vector":
        return store.similarity_search(query, k=k)

    # The BM25 index is rebuilt on every search. That's fine for a few dozen
    # chunks; a large collection would need a cached or dedicated index.
    stored = store.get(include=["documents", "metadatas"])
    chunks = [Document(page_content=text, metadata=meta)
              for text, meta in zip(stored["documents"], stored["metadatas"])]
    by_vector = store.similarity_search(query, k=len(chunks))
    bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])
    keyword_scores = bm25.get_scores(tokenize(query))
    by_keyword = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -keyword_scores[i])]
    return fuse([by_vector, by_keyword], k)


def fuse(rankings: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal rank fusion: a chunk that ranks well in several lists wins."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1 / (60 + rank)
            documents.setdefault(doc.page_content, doc)
    best = sorted(scores, key=lambda text: -scores[text])[:k]
    return [documents[text] for text in best]
