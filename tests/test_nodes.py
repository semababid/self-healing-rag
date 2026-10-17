import pytest
from langchain_core.documents import Document

from src import config, nodes
from src.nodes import FALLBACK_MESSAGE


@pytest.mark.parametrize("state, expected", [
    ({"grounded": True, "retries": 0}, "accept"),
    ({"grounded": True, "retries": 2}, "accept"),
    ({"grounded": False}, "retry"),
    ({"grounded": False, "retries": 1}, "retry"),
    ({"grounded": False, "retries": 2}, "give_up"),  # MAX_RETRIES is 2 in tests
])
def test_decide_after_critic(state, expected):
    assert nodes.decide_after_critic(state) == expected


def test_give_up_replaces_the_answer_with_the_fallback():
    assert nodes.give_up({"generation": "45 minutes."}) == {
        "generation": FALLBACK_MESSAGE,
        "grounded": False,
    }


@pytest.mark.parametrize("text, expected", [
    (FALLBACK_MESSAGE, True),
    (f'"{FALLBACK_MESSAGE}"', True),
    (f"  {FALLBACK_MESSAGE}\n", True),
    ("38 minutes.", False),
    ("", False),
])
def test_is_fallback(text, expected):
    assert nodes.is_fallback(text) == expected


@pytest.mark.parametrize("text", [
    f"{FALLBACK_MESSAGE} [1]",
    f'"{FALLBACK_MESSAGE}" [1, 2]',
])
def test_is_fallback_ignores_citations(text):
    assert nodes.is_fallback(text)


def test_cited_documents():
    docs = [Document(page_content=f"chunk {n}") for n in (1, 2, 3)]
    answer = "38 minutes [3]. It carries 2.1 kg [1, 3][9]."
    assert nodes.cited_documents(answer, docs) == [docs[2], docs[0]]
    assert nodes.cited_documents("no citations", docs) == []


def test_format_docs_numbers_the_chunks():
    docs = [Document(page_content="first"), Document(page_content="second")]
    assert nodes.format_docs(docs) == "[1] first\n\n[2] second"


def test_format_docs_without_documents():
    assert nodes.format_docs([]) == "(no documents retrieved)"


def test_retrieve_searches_for_the_current_query(retriever):
    out = nodes.retrieve({"question": "original question", "query": "rewritten query"})
    assert retriever.queries == ["rewritten query"]
    assert out["query"] == "rewritten query"
    assert out["documents"] == retriever.documents


def test_retrieve_uses_the_question_when_there_is_no_query(retriever):
    nodes.retrieve({"question": "original question"})
    assert retriever.queries == ["original question"]


def test_retrieve_replaces_chunks_on_a_retry_by_default(retriever):
    old = Document(page_content="old chunk")
    out = nodes.retrieve({"question": "q", "query": "q2", "retries": 1, "documents": [old]})
    assert out["documents"] == retriever.documents


def test_retrieve_can_keep_earlier_chunks(retriever, monkeypatch):
    monkeypatch.setattr(config, "KEEP_CHUNKS", True)
    old = Document(page_content="old chunk")
    again = Document(page_content=retriever.documents[0].page_content)  # found twice
    out = nodes.retrieve({"question": "q", "query": "q2", "retries": 1, "documents": [again, old]})
    assert out["documents"] == retriever.documents + [old]


def test_filter_keeps_only_the_chosen_chunks(llm):
    docs = [Document(page_content="about the N5"), Document(page_content="about the N7")]
    llm.replies = [{"chunks": [2]}]
    assert nodes.filter_chunks({"question": "q", "documents": docs}) == {"documents": [docs[1]]}


def test_filter_keeps_everything_if_its_reply_cant_be_read(llm, retriever):
    llm.replies = ["not json"]
    assert nodes.filter_chunks({"question": "q", "documents": retriever.documents}) == {}


def test_generate_without_chunks_returns_the_fallback_without_the_model(llm):
    assert nodes.generate({"question": "q", "documents": []}) == {"generation": FALLBACK_MESSAGE}
    assert llm.prompts == []


def test_generate_sends_question_and_context(llm, retriever):
    llm.replies = ["  38 minutes.\n"]
    out = nodes.generate({"question": "How long can the N7 fly?", "documents": retriever.documents})
    assert out == {"generation": "38 minutes."}
    prompt = llm.prompts[0][-1].content
    assert "How long can the N7 fly?" in prompt
    assert "38 minutes on a single charge" in prompt


def test_critic_accepts_a_grounded_answer(llm, retriever):
    llm.replies = [{"grounded": True, "reasoning": "The context says 38 minutes."}]
    out = nodes.critic({"question": "q", "documents": retriever.documents, "generation": "38 minutes."})
    assert out["grounded"] is True
    assert out["critique"] == "The context says 38 minutes."


def test_critic_rejects_an_unsupported_answer(llm, retriever):
    llm.replies = [{"grounded": False, "reasoning": "45 minutes isn't in the context."}]
    out = nodes.critic({"question": "q", "documents": retriever.documents, "generation": "45 minutes."})
    assert out["grounded"] is False
    assert out["critique"] == "45 minutes isn't in the context."


def test_critic_records_the_attempt(llm, retriever):
    llm.replies = [{"grounded": False, "reasoning": "45 minutes isn't in the context."}]
    out = nodes.critic({
        "question": "How long can the N7 fly?",
        "query": "N7 flight time",
        "documents": retriever.documents,
        "generation": "45 minutes.",
    })
    assert out["attempts"] == [{
        "query": "N7 flight time",
        "sources": ["nimbus_handbook.md"],
        "answer": "45 minutes.",
        "grounded": False,
        "critique": "45 minutes isn't in the context.",
    }]


def test_critic_rejects_when_its_output_cant_be_read(llm, retriever):
    llm.replies = ["this is not json"]
    out = nodes.critic({"question": "q", "documents": retriever.documents, "generation": "38 minutes."})
    assert out["grounded"] is False


def test_critic_doesnt_call_the_model_for_the_fallback(llm):
    out = nodes.critic({"question": "q", "documents": [], "generation": FALLBACK_MESSAGE})
    assert out["grounded"] is False
    assert out["attempts"][0]["answer"] == FALLBACK_MESSAGE
    assert llm.prompts == []


def test_transform_query_counts_the_retry_and_passes_on_the_critique(llm):
    llm.replies = ['"Nimbus N7 flight time"']
    out = nodes.transform_query({
        "question": "How long can the N7 fly?",
        "query": "How long can the N7 fly?",
        "critique": "45 minutes isn't in the context.",
        "retries": 1,
    })
    assert out == {"query": "Nimbus N7 flight time", "retries": 2}
    assert "45 minutes isn't in the context." in llm.prompts[0][-1].content


def test_transform_query_keeps_the_old_query_if_the_reply_is_empty(llm):
    llm.replies = ["   "]
    out = nodes.transform_query({"question": "q", "query": "old query"})
    assert out == {"query": "old query", "retries": 1}
