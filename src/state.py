from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str  # the user's question, never changed
    query: str  # what we search for; starts as the question, rewritten on retries
    documents: list[Document]  # chunks retrieved for the current query
    generation: str  # the latest answer
    grounded: bool  # the critic's verdict on that answer
    critique: str  # the critic's reason, used when rewriting the query
    retries: int  # how many times the query has been rewritten
