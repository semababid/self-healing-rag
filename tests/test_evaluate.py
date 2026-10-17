import pytest

from src.evaluate import CASES, Case, check
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
