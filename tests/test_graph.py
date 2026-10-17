import pytest

from src import config, graph
from src.graph import answer_question
from src.nodes import FALLBACK_MESSAGE

QUESTION = "What is the maximum flight time of the Nimbus N7?"
ACCEPT = {"grounded": True, "reasoning": "The context says 38 minutes."}
REJECT = {"grounded": False, "reasoning": "That number isn't in the context."}


def test_answer_accepted_on_the_first_try(llm, retriever):
    llm.replies = ["38 minutes.", ACCEPT]

    state = answer_question(QUESTION)

    assert state["generation"] == "38 minutes."
    assert state["grounded"] is True
    assert state["retries"] == 0
    assert retriever.queries == [QUESTION]
    assert llm.replies == []


def test_rejected_answer_is_fixed_by_a_retry(llm, retriever):
    llm.replies = [
        "45 minutes.", REJECT,
        "Nimbus N7 battery flight time",
        "38 minutes.", ACCEPT,
    ]

    state = answer_question(QUESTION)

    assert state["generation"] == "38 minutes."
    assert state["grounded"] is True
    assert state["retries"] == 1
    assert retriever.queries == [QUESTION, "Nimbus N7 battery flight time"]
    assert llm.replies == []


def test_state_keeps_every_attempt_and_step_time(llm, retriever):
    llm.replies = [
        "45 minutes.", REJECT,
        "Nimbus N7 battery flight time",
        "38 minutes.", ACCEPT,
    ]

    state = answer_question(QUESTION)

    assert [(a["query"], a["answer"], a["grounded"]) for a in state["attempts"]] == [
        (QUESTION, "45 minutes.", False),
        ("Nimbus N7 battery flight time", "38 minutes.", True),
    ]
    assert state["attempts"][0]["critique"] == REJECT["reasoning"]
    assert [t["step"] for t in state["timings"]] == [
        "retrieve", "generate", "critic",
        "transform_query", "retrieve", "generate", "critic",
    ]
    assert all(t["seconds"] >= 0 for t in state["timings"])


def test_gives_up_when_nothing_is_found(llm, retriever):
    # The critic is skipped for fallback answers, so only generate and rewrite call the model.
    llm.replies = [FALLBACK_MESSAGE, "query 2", FALLBACK_MESSAGE, "query 3", FALLBACK_MESSAGE]

    state = answer_question(QUESTION)

    assert state["generation"] == FALLBACK_MESSAGE
    assert state["grounded"] is False
    assert state["retries"] == 2
    assert retriever.queries == [QUESTION, "query 2", "query 3"]
    assert llm.replies == []


def test_answers_the_critic_keeps_rejecting_are_never_returned(llm, retriever):
    llm.replies = [
        "45 minutes.", REJECT, "query 2",
        "50 minutes.", REJECT, "query 3",
        "55 minutes.", REJECT,
    ]

    state = answer_question(QUESTION)

    assert state["generation"] == FALLBACK_MESSAGE
    assert state["grounded"] is False
    assert llm.replies == []


@pytest.fixture
def with_filter(monkeypatch):
    monkeypatch.setattr(config, "FILTER_CHUNKS", True)
    monkeypatch.setattr(graph, "app", graph.build_graph())


def test_filter_step_runs_between_retrieve_and_generate(llm, retriever, with_filter):
    llm.replies = [{"chunks": [1]}, "38 minutes.", ACCEPT]

    state = answer_question(QUESTION)

    assert state["generation"] == "38 minutes."
    assert [t["step"] for t in state["timings"]] == ["retrieve", "filter_chunks", "generate", "critic"]
    assert llm.replies == []


def test_filter_dropping_every_chunk_leads_to_a_retry(llm, retriever, with_filter):
    # No chunks left: generate and critic skip the model, and the query is rewritten.
    llm.replies = [{"chunks": []}, "query 2", {"chunks": [1]}, "38 minutes.", ACCEPT]

    state = answer_question(QUESTION)

    assert state["generation"] == "38 minutes."
    assert state["retries"] == 1
    assert llm.replies == []


def test_many_retries_with_the_filter_fit_in_the_step_limit(llm, retriever, with_filter, monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 8)
    llm.replies = [{"chunks": []}] + ["another query", {"chunks": []}] * 8

    state = answer_question(QUESTION)

    assert state["retries"] == 8


def test_many_retries_fit_in_the_step_limit(llm, retriever, monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 8)
    llm.replies = [FALLBACK_MESSAGE] + ["another query", FALLBACK_MESSAGE] * 8

    state = answer_question(QUESTION)

    assert state["retries"] == 8
    assert state["generation"] == FALLBACK_MESSAGE
