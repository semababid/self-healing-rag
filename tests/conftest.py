import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field, ValidationError

from src import config, nodes, vectorstore


class FakeLLM(BaseChatModel):
    """Stands in for Ollama. Returns the scripted replies in order.

    Use strings for generate and rewrite calls, and dicts like
    {"grounded": True, "reasoning": "..."} for critic calls.
    """

    replies: list = Field(default_factory=list)
    prompts: list = Field(default_factory=list)  # messages of every call, for checks

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _next_reply(self, messages):
        self.prompts.append(messages)
        if not self.replies:
            raise AssertionError("the model was called more often than the test expected")
        return self.replies.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next_reply(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def grade(prompt):
            reply = self._next_reply(prompt.to_messages())
            raw = AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))
            try:
                parsed = schema.model_validate_json(raw.content)
            except ValidationError:
                parsed = None
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(grade)


class FakeSearch:
    """Stands in for vectorstore.search. Always returns the same chunks."""

    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def __call__(self, query, k=None):
        self.queries.append(query)
        return self.documents


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    return fake


@pytest.fixture
def retriever(monkeypatch):
    fake = FakeSearch([
        Document(
            page_content="The N7 has a maximum flight time of 38 minutes on a single charge.",
            metadata={"source": "data/sample_docs/nimbus_handbook.md"},
        )
    ])
    monkeypatch.setattr(nodes, "search", fake)
    return fake


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text("The N7 has a maximum flight time of 38 minutes.\n" * 40)
    (folder / "parts.txt").write_text("Replacement parts are shipped from Tallinn.")
    (folder / ".DS_Store").write_bytes(b"\x00\x01")
    monkeypatch.setattr(config, "DATA_DIR", folder)
    return folder


@pytest.fixture
def temp_store(tmp_path, monkeypatch, docs_dir):
    # A store in a temp folder with fake embeddings: no model download, real chroma_db untouched.
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))


@pytest.fixture(autouse=True)
def default_retries(monkeypatch):
    # Don't let a MAX_RETRIES set in .env change what the tests expect.
    monkeypatch.setattr(config, "MAX_RETRIES", 2)


@pytest.fixture(autouse=True)
def runs_file(tmp_path, monkeypatch):
    # Tests never write to the real runs/ folder.
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(config, "RUNS_FILE", path)
    return path
