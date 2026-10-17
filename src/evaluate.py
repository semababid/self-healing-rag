"""Scores the pipeline on a fixed set of questions.

Answerable questions must get an answer the critic accepts and that contains
the expected fact. Unanswerable questions sound plausible but aren't covered by
the documents; they must end in the fallback reply instead of a made-up answer.
The documents describe a fictional company, so the model can't know these
answers from training.

    python -m src.evaluate               # results only
    python -m src.evaluate -v            # also show each step
    python -m src.evaluate --retrieval   # only check the search, no model needed

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
from src.vectorstore import get_vectorstore, require_store_built, search


@dataclass
class Case:
    question: str
    expect: tuple[str, ...] | None  # text the answer must contain; None if unanswerable
    evidence: str | None = None  # text in the chunk that holds the answer, for --retrieval


CASES = [
    Case("What is the maximum flight time of the Nimbus N7?", ("38",),
         "maximum flight time of 38 minutes"),
    Case("Who is the CEO of Nimbus Robotics?", ("Farkas",), "The CEO is Dr. Lena Farkas"),
    Case("How long is the warranty on the Nimbus N7?", ("24",), "24-month limited warranty"),
    Case("How much does Nimbus Care+ cost?", ("349",), "costs 349 EUR"),
    Case("How many paid vacation days do full-time employees get?", ("30",),
         "30 paid vacation days"),
    Case("Where are replacement parts shipped from?", ("Tallinn",),
         "shipped from the Tallinn office"),

    # Worded differently from the documents, so the search has less to go on.
    Case("How long until a battery pack wears out?", ("500",), "rated for 500 charge cycles"),
    Case("What has to be on the memory card before the drone will take off?", ("zone.cfg",),
         "SD card"),
    Case("Who is in charge of technology at Nimbus?", ("Reyes",), "The CTO is Marco Reyes"),
    Case("How much notice do I need to give before taking time off?", ("14",),
         "at least 14 days in advance"),
    Case("What happens to holiday days I don't use?", ("March 31",), "expire on March 31"),
    Case("How heavy a load can the N7 carry?", ("2.1",), "2.1 kg payload"),
    Case("How fast does the N7 fly inside a building?", ("6.5",), "cruises at 6.5 m/s indoors"),
    Case("How quickly do I get a reply if I write to support without Care+?", ("48",),
         "48-hour response target"),
    Case("How long is scan data kept?", ("18 months",), "retained for 18 months"),
    Case("In which city is customer data stored?", ("Frankfurt",), "Frankfurt"),
    Case("How wide must the corridors between shelves be?", ("2.4",), "at least 2.4 m wide"),
    Case("How close will the drone get to a person?", ("0.8",), "clearance of 0.8 m"),
    Case("When does the drone fly home by itself?", ("20%",), "falls to 20%"),
    Case("Can I undo a firmware update?", ("14 days",), "roll back to the previous Cirrus"),
    Case("How much money do I get each year for courses?", ("1,500",),
         "learning budget of 1,500 EUR"),
    Case("How many days a week can I work from home?", ("3 days",), "up to 3 days per week"),
    Case("What do I do if the drone can't read tags properly?", ("ecalibrat",),
         "Recalibrate via the Halo portal"),
    # Correct answers phrase this as "not covered" or "covers ... but not crashes
    # caused by manual override", so check for both parts separately.
    Case("Who pays for repairs if a pilot crashes the drone by hand?", ("not", "manual override"),
         "manual override"),

    # Not in the documents. Avoid questions they answer by omission: "Is there
    # a Berlin office?" has a grounded answer (no), since every office is listed.
    # The N5 and N9 documents give colours, prices, heights and batteries for
    # other models, so these also test that the N7 isn't mixed up with them.
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


def evidence_rank(case: Case) -> int | None:
    """Position (1 = best) of the first chunk containing the case's evidence."""
    total = len(get_vectorstore().get(include=[])["ids"])
    for rank, doc in enumerate(search(case.question, k=total), 1):
        if case.evidence in doc.page_content:
            return rank
    return None


def check_retrieval() -> bool:
    """Print where the answer chunk ranks for each question. True if all are in the top k."""
    cases = [case for case in CASES if case.evidence]
    ranks = [evidence_rank(case) for case in cases]
    for case, rank in zip(cases, ranks):
        hit = rank is not None and rank <= config.RETRIEVER_K
        print(f"{'hit ' if hit else 'MISS'}  rank={rank or '-':>3}  {case.question}")
    hits = sum(rank is not None and rank <= config.RETRIEVER_K for rank in ranks)
    # Mean reciprocal rank: 1.0 if the answer chunk always comes first, 0.5 if always second.
    mrr = sum(1 / rank for rank in ranks if rank) / len(ranks)
    print(f"\n{hits}/{len(cases)} answer chunks in the top {config.RETRIEVER_K}, "
          f"mean reciprocal rank {mrr:.2f}")
    return hits == len(cases)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline on a fixed question set.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each step")
    parser.add_argument("--retrieval", action="store_true",
                        help="only check where the search ranks the answer; no model needed")
    args = parser.parse_args(argv[1:])
    setup_logging(args.verbose)

    try:
        if not args.retrieval:
            check_ollama()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"search={config.SEARCH} chunk_size={config.CHUNK_SIZE} "
          f"overlap={config.CHUNK_OVERLAP} k={config.RETRIEVER_K}")
    if args.retrieval:
        print()
        return 0 if check_retrieval() else 1

    print(f"model={config.LLM_MODEL} thinking={config.LLM_THINKING} "
          f"max_retries={config.MAX_RETRIES}\n")
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
