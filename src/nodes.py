"""The steps of the graph. Each one takes the state and returns the fields it changes."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src import config
from src.llm import get_llm
from src.state import GraphState
from src.vectorstore import get_retriever

log = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I don't have enough information in the available documents to answer that reliably."
)


def is_fallback(text: str) -> bool:
    # The model sometimes wraps the sentence in quotes.
    return text.strip().strip("\"'").strip() == FALLBACK_MESSAGE


def format_docs(documents: list[Document]) -> str:
    if not documents:
        return "(no documents retrieved)"
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, 1))


def retrieve(state: GraphState) -> GraphState:
    query = state.get("query") or state["question"]
    log.info("[retrieve] %r", query)
    return {"documents": get_retriever().invoke(query), "query": query}


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using only the context. Don't use outside knowledge. "
     "If the context doesn't contain the answer, reply with this sentence and "
     "nothing else:\n{fallback}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]).partial(fallback=FALLBACK_MESSAGE)


def generate(state: GraphState) -> GraphState:
    chain = GENERATE_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(state.get("documents", [])),
        "question": state["question"],
    }).strip()
    log.info("[generate] %r", answer)
    return {"generation": answer}


class Grade(BaseModel):
    grounded: bool = Field(
        description="true only if every claim in the answer is supported by the context"
    )
    reasoning: str = Field(description="one or two sentences explaining the verdict")


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check answers produced by a document search system. An answer is "
     "grounded if it answers the question and every fact it states appears in "
     "the context. If it adds, changes or guesses anything, it is not grounded. "
     "An answer saying the information isn't available is not grounded either."),
    ("human",
     "Question: {question}\n\nContext:\n{context}\n\nAnswer to check: {answer}"),
])


def critic(state: GraphState) -> GraphState:
    answer = state.get("generation", "")
    documents = state.get("documents", [])
    if is_fallback(answer):
        # The generator already found nothing, so there is nothing to check.
        grounded, critique = False, "The retrieved context did not contain the answer."
    else:
        grader = CRITIC_PROMPT | get_llm().with_structured_output(Grade, include_raw=True)
        grade = grader.invoke({
            "question": state["question"],
            "context": format_docs(documents),
            "answer": answer,
        })["parsed"]
        if grade is None:
            # Unreadable output counts as a rejection, never as a pass.
            grounded, critique = False, "The critic's verdict could not be read."
        else:
            grounded, critique = grade.grounded, grade.reasoning

    log.info("[critic] %s: %s", "grounded" if grounded else "rejected", critique)
    attempt = {
        "query": state.get("query", state["question"]),
        "sources": [Path(doc.metadata.get("source", "?")).name for doc in documents],
        "answer": answer,
        "grounded": grounded,
        "critique": critique,
    }
    return {"grounded": grounded, "critique": critique, "attempts": [attempt]}


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write search queries for a document search engine. The last query "
     "didn't find what was needed. Write one better query, for example with "
     "synonyms or more specific terms. Reply with the query only."),
    ("human",
     "Question: {question}\nLast query: {query}\nWhy it failed: {critique}"),
])


def transform_query(state: GraphState) -> GraphState:
    retries = state.get("retries", 0) + 1
    previous = state.get("query", state["question"])
    chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
    query = chain.invoke({
        "question": state["question"],
        "query": previous,
        "critique": state.get("critique", ""),
    }).strip().strip("\"'")
    log.info("[rewrite #%d] %r", retries, query)
    return {"query": query or previous, "retries": retries}


def give_up(state: GraphState) -> GraphState:
    log.info("[give up] out of retries")
    return {"generation": FALLBACK_MESSAGE, "grounded": False}


def decide_after_critic(state: GraphState) -> str:
    if state.get("grounded"):
        return "accept"
    if state.get("retries", 0) < config.MAX_RETRIES:
        return "retry"
    return "give_up"
