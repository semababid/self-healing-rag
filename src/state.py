"""The shared state object that flows through the LangGraph.

Every node receives the current state and returns a partial update, which
LangGraph merges back in. Using ``total=False`` means nodes only need to return
the keys they actually change.
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """State passed between nodes of the self-healing RAG graph."""

    question: str
    """The user's original question (never mutated)."""

    query: str
    """The current retrieval query — starts equal to ``question`` and gets
    rewritten by the ``transform_query`` node on each retry."""

    documents: List[Document]
    """Chunks retrieved for the current ``query``."""

    generation: str
    """The latest candidate answer produced by the ``generate`` node."""

    grounded: bool
    """The critic's verdict: is ``generation`` supported by ``documents``?"""

    critique: str
    """The critic's short justification (useful for logging / query rewrites)."""

    retries: int
    """How many times we've rewritten the query and retried so far."""
