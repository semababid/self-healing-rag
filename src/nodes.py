"""The graph nodes: retrieve, generate, critic, transform_query, give_up.

Each node is a plain function ``(state) -> partial state update``. LangGraph
calls them and merges the returned dict back into the shared ``GraphState``.
The self-healing behaviour emerges from the *edges* (wired in ``graph.py``);
here we only implement what each individual step does.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src import config
from src.llm import get_llm
from src.state import GraphState
from src.vectorstore import get_retriever

# Message shown when the pipeline honestly cannot answer from the documents.
FALLBACK_MESSAGE = (
    "I don't have enough information in the available documents to answer that "
    "reliably."
)


def _format_docs(documents: list[Document]) -> str:
    """Render retrieved chunks into a single numbered context block."""
    if not documents:
        return "(no documents retrieved)"
    return "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1)
    )


# --- Node 1: retrieve ------------------------------------------------------
def retrieve(state: GraphState) -> GraphState:
    """Fetch the top-k chunks for the current query from the vector store."""
    query = state.get("query") or state["question"]
    print(f"  [retrieve] query: {query!r}")
    documents = get_retriever().invoke(query)
    print(f"  [retrieve] got {len(documents)} chunk(s)")
    return {"documents": documents, "query": query}


# --- Node 2: generate ------------------------------------------------------
_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful question-answering assistant. Answer the user's "
            "question using ONLY the provided context. If the context does not "
            "contain enough information to answer, reply exactly: "
            f'"{FALLBACK_MESSAGE}" '
            "Do not use outside knowledge and do not guess.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)


def generate(state: GraphState) -> GraphState:
    """Produce a candidate answer grounded in the retrieved context."""
    print("  [generate] drafting answer ...")
    chain = _GENERATE_PROMPT | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "context": _format_docs(state.get("documents", [])),
            "question": state["question"],
        }
    )
    return {"generation": generation}


# --- Node 3: critic --------------------------------------------------------
class Grade(BaseModel):
    """The critic's structured verdict on a candidate answer."""

    grounded: bool = Field(
        description=(
            "True only if the answer is fully supported by the context AND the "
            "context actually contains the information needed to answer the "
            "question. False if the answer contains any unsupported claim, or if "
            "the context lacks the information (i.e. the answer should have been "
            "a refusal)."
        )
    )
    reasoning: str = Field(description="One or two sentences justifying the verdict.")


_CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checking critic for a RAG system. Decide "
            "whether the candidate answer is grounded in the provided context. "
            "An answer is grounded only if every factual claim it makes is "
            "directly supported by the context, and the context genuinely "
            "contains the information the question asks for. A refusal such as "
            "'I don't have enough information' means the context was "
            "insufficient, so it is NOT grounded.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\n"
            "Candidate answer: {generation}",
        ),
    ]
)


def critic(state: GraphState) -> GraphState:
    """Judge whether the candidate answer is grounded in the retrieved chunks."""
    print("  [critic] grading answer ...")
    grader = _CRITIC_PROMPT | get_llm().with_structured_output(Grade)
    result: Grade = grader.invoke(
        {
            "question": state["question"],
            "context": _format_docs(state.get("documents", [])),
            "generation": state.get("generation", ""),
        }
    )
    verdict = "GROUNDED" if result.grounded else "REJECTED"
    print(f"  [critic] {verdict} — {result.reasoning}")
    return {"grounded": result.grounded, "critique": result.reasoning}


# --- Node 4: transform_query (the "retry") --------------------------------
_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You reformulate search queries for a document retriever. The "
            "previous query did not surface enough information to answer the "
            "user's question. Produce ONE improved search query: use synonyms, "
            "key entities, and more specific terms. Return only the query text, "
            "with no quotes or explanation.",
        ),
        (
            "human",
            "User question: {question}\n"
            "Previous query: {query}\n"
            "Why the last attempt failed: {critique}\n\n"
            "Improved query:",
        ),
    ]
)


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the retrieval query and bump the retry counter."""
    retries = state.get("retries", 0) + 1
    print(f"  [transform_query] retry #{retries}: reformulating query ...")
    chain = _REWRITE_PROMPT | get_llm() | StrOutputParser()
    new_query = chain.invoke(
        {
            "question": state["question"],
            "query": state.get("query", state["question"]),
            "critique": state.get("critique", ""),
        }
    ).strip()
    print(f"  [transform_query] new query: {new_query!r}")
    return {"query": new_query, "retries": retries}


# --- Node 5: give_up -------------------------------------------------------
def give_up(state: GraphState) -> GraphState:
    """Emit the graceful fallback answer after exhausting retries."""
    print("  [give_up] out of retries — responding honestly")
    return {"generation": FALLBACK_MESSAGE, "grounded": False}


# --- Routing ---------------------------------------------------------------
def decide_after_critic(state: GraphState) -> str:
    """Conditional edge: where to go after the critic has graded the answer."""
    if state.get("grounded"):
        return "accept"
    if state.get("retries", 0) < config.MAX_RETRIES:
        return "retry"
    return "give_up"
