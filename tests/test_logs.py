import json
import logging

import pytest

from src import evaluate
from src.graph import answer_question
from src.logs import save_run, setup_logging, step_totals

QUESTION = "What is the maximum flight time of the Nimbus N7?"
ACCEPT = {"grounded": True, "reasoning": "The context says 38 minutes."}


@pytest.fixture
def src_logger():
    logger = logging.getLogger("src")
    level = logger.level
    yield logger
    logger.setLevel(level)


def test_setup_logging_is_quiet_unless_verbose(src_logger):
    setup_logging(verbose=False)
    assert src_logger.level == logging.WARNING
    setup_logging(verbose=True)
    assert src_logger.level == logging.INFO


def test_steps_are_logged(llm, retriever, caplog):
    llm.replies = ["38 minutes.", ACCEPT]
    with caplog.at_level(logging.INFO, logger="src"):
        answer_question(QUESTION)
    assert f"[retrieve] {QUESTION!r}" in caplog.text
    assert "[generate] '38 minutes.'" in caplog.text
    assert "[critic] grounded: The context says 38 minutes." in caplog.text


def test_step_totals_adds_up_repeated_steps():
    timings = [
        {"step": "retrieve", "seconds": 0.5},
        {"step": "generate", "seconds": 2.0},
        {"step": "retrieve", "seconds": 0.25},
    ]
    assert step_totals(timings) == {"retrieve": 0.75, "generate": 2.0}


def test_save_run_appends_one_json_line_per_question(llm, retriever, runs_file):
    llm.replies = ["38 minutes.", ACCEPT, "38 minutes.", ACCEPT]
    save_run(QUESTION, answer_question(QUESTION), 1.234)
    save_run(QUESTION, answer_question(QUESTION), 2.0, source="test")

    records = [json.loads(line) for line in runs_file.read_text().splitlines()]
    assert len(records) == 2
    first = records[0]
    assert first["question"] == QUESTION
    assert first["answer"] == "38 minutes."
    assert first["grounded"] is True
    assert first["seconds"] == 1.23
    assert len(first["attempts"]) == 1
    assert [t["step"] for t in first["timings"]] == ["retrieve", "generate", "critic"]
    assert records[1]["source"] == "test"


def test_evaluate_saves_each_case_with_its_result(llm, retriever, runs_file):
    llm.replies = ["38 minutes.", ACCEPT]
    result = evaluate.run(evaluate.Case(QUESTION, ("38",)))

    assert result.failure is None
    record = json.loads(runs_file.read_text())
    assert record["source"] == "evaluate"
    assert record["passed"] is True
    assert record["failure"] is None
