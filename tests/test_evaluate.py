import pytest

from src import config, evaluate, ingest
from src.evaluate import CASES, Case, check, evidence_rank
from src.nodes import FALLBACK_MESSAGE

FLIGHT_TIME = Case("What is the maximum flight time of the Nimbus N7?", ("38",))
CEO = Case("Who is the CEO of Nimbus Robotics?", ("Farkas",))
COLOURS = Case("What colours is the Nimbus N7 available in?", None)


@pytest.mark.parametrize("case, state, expected", [
    (FLIGHT_TIME, {"generation": "38 minutes.", "grounded": True}, None),
    (CEO, {"generation": "dr. lena farkas", "grounded": True}, None),
    (FLIGHT_TIME, {"generation": FALLBACK_MESSAGE, "grounded": False},
     "refused an answerable question"),
    (FLIGHT_TIME, {"generation": "38 minutes.", "grounded": False},
     "the critic rejected the answer"),
    (FLIGHT_TIME, {"generation": "45 minutes.", "grounded": True},
     "answer doesn't contain ['38']"),
    (COLOURS, {"generation": FALLBACK_MESSAGE, "grounded": False}, None),
    (COLOURS, {"generation": "It comes in grey and white.", "grounded": True},
     "answered instead of refusing"),
])
def test_check(case, state, expected):
    assert check(case, state) == expected


def test_cases_are_well_formed():
    questions = [case.question for case in CASES]
    assert len(questions) == len(set(questions))
    for case in CASES:
        assert case.expect is None or len(case.expect) > 0
        # Answerable cases need evidence for --retrieval; unanswerable ones have none.
        assert (case.evidence is None) == (case.expect is None)


def test_every_evidence_text_is_in_the_documents():
    # Catches typos: evidence that appears nowhere can never be "found".
    from src import config
    text = "\n".join(path.read_text() for path in config.DATA_DIR.glob("*.md"))
    missing = [case.evidence for case in CASES if case.evidence and case.evidence not in text]
    assert missing == []


def test_evidence_rank_finds_the_answer_chunk(temp_store):
    ingest.ingest()
    assert evidence_rank(Case("Where do parts ship from?", ("Tallinn",), "shipped from Tallinn")) in (1, 2)
    assert evidence_rank(Case("?", ("x",), "text that is in no chunk")) is None


def test_check_retrieval_reports_hits(temp_store, monkeypatch, capsys):
    ingest.ingest()
    monkeypatch.setattr(evaluate, "CASES", [
        Case("Where do parts ship from?", ("Tallinn",), "shipped from Tallinn"),
    ])
    monkeypatch.setattr(config, "RETRIEVER_K", 2)  # the temp store has only two chunks
    assert evaluate.check_retrieval() is True
    assert "1/1 answer chunks in the top 2" in capsys.readouterr().out
