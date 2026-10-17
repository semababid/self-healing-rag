"""Phase 6: evaluate the self-healing behaviour on a curated question set.

The whole point of this pipeline is a claim: it answers what the documents
support and *refuses to hallucinate* on what they don't. This script turns that
claim into something measurable.

We run a fixed set of questions through the real graph and score each one
against its expected outcome:

* **answerable**   — the docs contain the answer → the pipeline should produce a
  grounded answer (critic accepts, no fallback).
* **unanswerable** — the docs do NOT contain the answer, though the question
  sounds plausible → the pipeline should refuse honestly rather than invent one.

Run it after building the store (``python -m src.ingest``):

    python -m src.evaluate            # full report
    python -m src.evaluate --quiet    # hide per-node trace, show only the table

Exit code is non-zero if any case fails, so this also works as a smoke test.

Note: this calls the Anthropic API once per question (plus extra calls on any
retries), so it needs ``ANTHROPIC_API_KEY`` set and will incur a little cost.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass

from src import config
from src.graph import answer_question
from src.nodes import FALLBACK_MESSAGE
from src.vectorstore import get_vectorstore


@dataclass(frozen=True)
class Case:
    """One evaluation question and the behaviour we expect from the pipeline."""

    question: str
    answerable: bool
    note: str  # what a correct response hinges on — for the human reading output


# The fictional Nimbus docs make this safe: an "unanswerable" question has no
# real answer anywhere, so a confident response can only be a hallucination.
CASES: list[Case] = [
    # --- Answerable: the facts are in the documents ------------------------
    Case("What is the maximum flight time of the Nimbus N7?", True, "38 minutes"),
    Case("Who is the CEO of Nimbus Robotics?", True, "Dr. Lena Farkas"),
    Case("How long is the warranty on the Nimbus N7?", True, "24 months"),
    Case("How much does Nimbus Care+ cost?", True, "349 EUR"),
    Case("How many paid vacation days do full-time employees get?", True, "30 days"),
    Case("Where are replacement parts shipped from?", True, "Tallinn office"),
    # --- Unanswerable: plausible, but absent from the documents ------------
    Case("What colours is the Nimbus N7 available in?", False, "never stated"),
    Case("How much does the Nimbus N7 drone itself cost to buy?", False, "no price given"),
    Case("Who is the CFO of Nimbus Robotics?", False, "only CEO/CTO are named"),
    Case("What is the maximum altitude the Nimbus N7 can reach?", False, "not in docs"),
    Case("Does Nimbus Robotics have an office in Berlin?", False, "offices are Cottbus/Porto/Tallinn"),
]


@dataclass
class Result:
    """The outcome of running one case through the pipeline."""

    case: Case
    refused: bool
    grounded: bool
    retries: int
    answer: str

    @property
    def passed(self) -> bool:
        """A case passes when the pipeline's behaviour matches expectation.

        Answerable → must produce a grounded, non-refusal answer.
        Unanswerable → must refuse (the honest fallback), not fabricate.
        """
        if self.case.answerable:
            return self.grounded and not self.refused
        return self.refused


def _check_store_ready() -> None:
    """Fail early with a clear hint if the vector store hasn't been built."""
    if not get_vectorstore().get().get("ids"):
        raise SystemExit(
            "The vector store is empty. Build it first:\n"
            "    python -m src.ingest"
        )


def run_case(case: Case, *, quiet: bool) -> Result:
    """Run one question through the full graph and capture the outcome."""
    # The nodes print a live trace; in --quiet mode we swallow it so only the
    # summary table remains.
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        state = answer_question(case.question)

    answer = (state.get("generation") or "").strip()
    return Result(
        case=case,
        refused=answer == FALLBACK_MESSAGE,
        grounded=bool(state.get("grounded")),
        retries=int(state.get("retries", 0)),
        answer=answer,
    )


def _print_report(results: list[Result]) -> None:
    """Print a per-case table followed by aggregate accuracy numbers."""
    print("\n" + "=" * 78)
    print("SELF-HEALING RAG — EVALUATION REPORT")
    print("=" * 78)
    header = f"{'result':8} {'expected':13} {'retries':8} question"
    print(header)
    print("-" * 78)
    for r in results:
        mark = "PASS ✓" if r.passed else "FAIL ✗"
        expected = "answerable" if r.case.answerable else "unanswerable"
        print(f"{mark:8} {expected:13} {r.retries:^8} {r.case.question}")

    # Aggregate: overall, plus a breakdown by the two behaviours we care about.
    total = len(results)
    passed = sum(r.passed for r in results)
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    healed = [r for r in results if r.passed and r.retries > 0]

    print("-" * 78)
    print(f"Overall:       {passed}/{total} passed")
    print(
        f"  grounded answers:  "
        f"{sum(r.passed for r in answerable)}/{len(answerable)} answerable "
        f"questions answered correctly"
    )
    print(
        f"  honest refusals:   "
        f"{sum(r.passed for r in unanswerable)}/{len(unanswerable)} unanswerable "
        f"questions refused (no hallucination)"
    )
    if healed:
        print(
            f"  self-healed:       {len(healed)} case(s) only passed after a "
            f"query rewrite + retry"
        )
    print("=" * 78 + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide the per-node trace; print only the summary table",
    )
    args = parser.parse_args(argv[1:])

    config.require_anthropic_key()
    _check_store_ready()

    print(f"Evaluating {len(CASES)} question(s) — this calls the API per case ...")
    results = [run_case(case, quiet=args.quiet) for case in CASES]
    _print_report(results)

    # Non-zero exit on any failure makes this usable as a CI / smoke test.
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
