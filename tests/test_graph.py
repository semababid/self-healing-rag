from src import config
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


def test_many_retries_fit_in_the_step_limit(llm, retriever, monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 8)
    llm.replies = [FALLBACK_MESSAGE] + ["another query", FALLBACK_MESSAGE] * 8

    state = answer_question(QUESTION)

    assert state["retries"] == 8
    assert state["generation"] == FALLBACK_MESSAGE
