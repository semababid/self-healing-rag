"""Scores the pipeline on a fixed set of questions.

Answerable questions must get an answer the critic accepts and that contains
the expected fact. Unanswerable questions sound plausible but aren't covered by
the documents; they must end in the fallback reply instead of a made-up answer.
The documents describe a fictional company, so the model can't know these
answers from training.

    python -m src.evaluate        # results only
    python -m src.evaluate -v     # also show each step

Each case is saved to runs/runs.jsonl. Exits with status 1 if any case fails.
"""

import argparse
import sys
import time
from dataclasses import dataclass

from src import config
from src.graph import answer_question
from src.llm import LLM_ERRORS, check_ollama
from src.logs import save_run, setup_logging, step_totals
from src.nodes import is_fallback
from src.vectorstore import require_store_built


@dataclass
class Case:
    question: str
    expect: tuple[str, ...] | None  # text the answer must contain; None if unanswerable


CASES = [
    Case("What is the maximum flight time of the Nimbus N7?", ("38",)),
    Case("Who is the CEO of Nimbus Robotics?", ("Farkas",)),
    Case("How long is the warranty on the Nimbus N7?", ("24",)),
    Case("How much does Nimbus Care+ cost?", ("349",)),
    Case("How many paid vacation days do full-time employees get?", ("30",)),
    Case("Where are replacement parts shipped from?", ("Tallinn",)),
    # Not in the documents. Avoid questions they answer by omission: "Is there
    # a Berlin office?" has a grounded answer (no), since every office is listed.
    Case("What colours is the Nimbus N7 available in?", None),
    Case("How much does the Nimbus N7 drone itself cost to buy?", None),  # only Care+ has a price
    Case("Who is the CFO of Nimbus Robotics?", None),  # only the CEO and CTO are named
    Case("What is the maximum altitude the Nimbus N7 can reach?", None),
    Case("What battery chemistry does the Nimbus N7 use?", None),
]


@dataclass
class Result:
    case: Case
    state: dict
    seconds: float
    failure: str | None  # None means the case passed


def check(case: Case, state: dict) -> str | None:
    answer = state.get("generation", "")
    if case.expect is None:
        return None if is_fallback(answer) else "answered instead of refusing"
    if is_fallback(answer):
        return "refused an answerable question"
    if not state.get("grounded"):
        return "the critic rejected the answer"
    missing = [fact for fact in case.expect if fact.lower() not in answer.lower()]
    return f"answer doesn't contain {missing}" if missing else None


def run(case: Case) -> Result:
    start = time.perf_counter()
    state = answer_question(case.question)
    result = Result(case, state, time.perf_counter() - start, check(case, state))
    save_run(case.question, state, result.seconds, source="evaluate",
             passed=result.failure is None, failure=result.failure)
    return result


def summarize(results: list[Result]) -> None:
    answerable = [r for r in results if r.case.expect is not None]
    unanswerable = [r for r in results if r.case.expect is None]
    # Unanswerable cases always use up their retries before refusing, so only
    # answerable ones count as fixed by a retry.
    healed = [r for r in answerable if r.failure is None and r.state.get("retries", 0) > 0]

    print(f"\n{sum(r.failure is None for r in results)}/{len(results)} passed")
    print(f"  answerable, answered correctly:  "
          f"{sum(r.failure is None for r in answerable)}/{len(answerable)}")
    print(f"  unanswerable, refused:           "
          f"{sum(r.failure is None for r in unanswerable)}/{len(unanswerable)}")
    print(f"  answered only after a retry:     {len(healed)}")
    total = sum(r.seconds for r in results)
    print(f"  total time:                      {total:.0f}s")

    totals = step_totals([t for r in results for t in r.state.get("timings", [])])
    print("\ntime per step, all questions together:")
    for step, seconds in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {step:16} {seconds:6.1f}s  {100 * seconds / total:3.0f}%")

    for r in results:
        if r.failure:
            print(f"\nFAIL: {r.case.question}\n  why:    {r.failure}"
                  f"\n  answer: {r.state.get('generation', '')[:200]}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline on a fixed question set.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each step")
    args = parser.parse_args(argv[1:])
    setup_logging(args.verbose)

    try:
        check_ollama()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"model={config.LLM_MODEL} thinking={config.LLM_THINKING} "
          f"max_retries={config.MAX_RETRIES} k={config.RETRIEVER_K}\n")
    results = []
    for case in CASES:
        try:
            result = run(case)
        except LLM_ERRORS as err:
            print(f"Error: the model call failed ({err})", file=sys.stderr)
            return 1
        results.append(result)
        status = "PASS" if result.failure is None else "FAIL"
        print(f"{status}  retries={result.state.get('retries', 0)}  "
              f"{result.seconds:5.1f}s  {case.question}")

    summarize(results)
    return 0 if all(r.failure is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
