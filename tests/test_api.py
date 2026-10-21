import httpx
import pytest
from fastapi.testclient import TestClient

from src import api, main
from src.nodes import FALLBACK_MESSAGE

QUESTION = "What is the maximum flight time of the Nimbus N7?"
ACCEPT = {"grounded": True, "reasoning": "The context says 38 minutes."}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "problem", lambda: None)
    return TestClient(api.app)


def test_ask_returns_answer_sources_attempts_and_timings(client, llm, retriever):
    llm.replies = ["38 minutes [1].", ACCEPT]

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "38 minutes [1]."
    assert body["grounded"] is True
    assert body["retries"] == 0
    assert body["sources_cited"] is True
    assert body["sources"] == [{
        "number": 1,
        "file": "nimbus_handbook.md",
        "text": retriever.documents[0].page_content,
    }]
    assert len(body["attempts"]) == 1
    assert [t["step"] for t in body["timings"]] == ["retrieve", "generate", "critic"]


def test_answer_without_citations_lists_every_chunk_used(client, llm, retriever):
    llm.replies = ["38 minutes.", ACCEPT]
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["sources_cited"] is False
    assert [s["file"] for s in body["sources"]] == ["nimbus_handbook.md"]


def test_refusal_has_no_sources(client, llm, retriever):
    llm.replies = [FALLBACK_MESSAGE, "query 2", FALLBACK_MESSAGE, "query 3", FALLBACK_MESSAGE]
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["grounded"] is False
    assert body["answer"] == FALLBACK_MESSAGE
    assert body["sources"] == []
    assert body["retries"] == 2


@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 501},
])
def test_bad_questions_are_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_not_ready_returns_503_with_the_reason(monkeypatch):
    monkeypatch.setattr(api, "problem", lambda: "Can't reach Ollama at http://localhost:11434.")
    client = TestClient(api.app)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert "Can't reach Ollama" in response.json()["detail"]
    assert client.get("/health").json() == {
        "ready": False,
        "problem": "Can't reach Ollama at http://localhost:11434.",
    }


def test_model_failure_returns_503(client, monkeypatch):
    def fail(question, source):
        raise httpx.ConnectError("Connection refused")
    monkeypatch.setattr(api, "answer", fail)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert "Connection refused" in response.json()["detail"]


def test_health_when_ready(client):
    assert client.get("/health").json() == {"ready": True, "problem": None}


def test_cli_prints_the_answer_and_cited_files(llm, retriever, monkeypatch, capsys):
    monkeypatch.setattr(main, "problem", lambda: None)
    llm.replies = ["38 minutes [1].", ACCEPT]

    assert main.main(["main", QUESTION]) == 0

    out = capsys.readouterr().out
    assert "38 minutes [1]." in out
    assert "[1] nimbus_handbook.md" in out


def test_cli_stops_when_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(main, "problem", lambda: "The vector store is empty.")
    assert main.main(["main", QUESTION]) == 1
    assert "The vector store is empty." in capsys.readouterr().err
