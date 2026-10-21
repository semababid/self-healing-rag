"""HTTP API.

    uvicorn src.api:app --reload

    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \\
         -d '{"question": "What is the maximum flight time of the Nimbus N7?"}'

Interactive docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm import LLM_ERRORS
from src.service import answer, problem

app = FastAPI(
    title="Self-healing RAG",
    description="Answers questions about the documents, checks each answer "
                "against them, and says so when it can't answer.",
)


class Question(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"\S",  # not just spaces
        examples=["What is the maximum flight time of the Nimbus N7?"],
    )


class Source(BaseModel):
    number: int = Field(description="the chunk's number in the context, as cited in the answer")
    file: str
    text: str


class Attempt(BaseModel):
    query: str
    sources: list[str] = Field(description="files the retrieved chunks came from")
    answer: str
    grounded: bool
    critique: str


class Timing(BaseModel):
    step: str
    seconds: float


class Answer(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(description="false means the system declined to answer")
    retries: int
    seconds: float
    sources_cited: bool = Field(description="false if the answer has no citations; "
                                            "sources then lists every chunk used")
    sources: list[Source]
    attempts: list[Attempt]
    timings: list[Timing]


@app.get("/health")
def health() -> dict:
    issue = problem()
    return {"ready": issue is None, "problem": issue}


@app.post("/ask", response_model=Answer, responses={503: {"description": "not ready"}})
def ask(body: Question) -> dict:
    issue = problem()
    if issue:
        raise HTTPException(status_code=503, detail=issue)
    try:
        return answer(body.question.strip(), source="api")
    except LLM_ERRORS as err:
        raise HTTPException(status_code=503, detail=f"The model call failed: {err}") from None
