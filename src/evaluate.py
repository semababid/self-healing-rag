"""Phase 6: evaluate the self-healing behaviour on a curated question set.

The whole point of this pipeline is a claim: it answers what the documents
support and *refuses to hallucinate* on what they don't. This script turns that
claim into something measurable.

We run a fixed set of questions through the real graph and score each one
against its expected outcome:

* **answerable**: the docs contain the answer, so the pipeline should produce a
  grounded answer (critic accepts, no fallback) that states the expected fact.
* **unanswerable**: the question sounds plausible but the docs don't answer it,
  so the pipeline should refuse honestly rather than invent an answer.

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
from src.vectorstore import require_store_built


@dataclass(frozen=True)
class Case:
    """One evaluation question and the behaviour we expect from the pipeline."""

    question: str
    answerable: bool
    # For answerable cases: substrings the answer must contain (case-insensitive),
    # so a grounded-but-wrong answer still fails. Keep them short and unambiguous.
    expect: tuple[str, ...] = ()


# The fictional Nimbus docs make this safe: an "unanswerable" question has no
# real answer anywhere, so a confident response can only be a hallucination.
# Avoid questions the docs answer *by omission*. For "is there a Berlin
# office?", the docs list every office, so "No" is a grounded answer.
CASES: list[Case] = [
    # --- Answerable: the facts are in the documents ------------------------
    Case("What is the maximum flight time of the Nimbus N7?", True, ("38",)),
    Case("Who is the CEO of Nimbus Robotics?", True, ("Farkas",)),
    Case("How long is the warranty on the Nimbus N7?", True, ("24",)),
    Case("How much does Nimbus Care+ cost?", True, ("349",)),
    Case("How many paid vacation days do full-time employees get?", True, ("30",)),
    Case("Where are replacement parts shipped from?", True, ("Tallinn",)),
    # --- Unanswerable: plausible, but absent from the documents ------------
    Case("What colours is the Nimbus N7 available in?", False),
    Case("How much does the Nimbus N7 drone itself cost to buy?", False),  # only Care+ is priced
    Case("Who is the CFO of Nimbus Robotics?", False),  # only CEO/CTO are named
    Case("What is the maximum altitude the Nimbus N7 can reach?", False),
    Case("What battery chemistry does the Nimbus N7 use?", False),
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

        Answerable → must produce a grounded, non-refusal answer containing
        every expected fact.
        Unanswerable → must refuse (the honest fallback), not fabricate.
        """
        if self.case.answerable:
            return self.grounded and not self.refused and not self.missing
        return self.refused

    @property
    def missing(self) -> list[str]:
        """Expected facts that don't appear in the answer."""
        answer = self.answer.lower()
        return [fact for fact in self.case.expect if fact.lower() not in answer]


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
    print("SELF-HEALING RAG: EVALUATION REPORT")
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

    # For each failure, show what came back so it can be diagnosed.
    failures = [r for r in results if not r.passed]
    if failures:
        print("-" * 78)
        print("Failures:")
        for r in failures:
            if r.case.answerable and r.refused:
                why = "refused an answerable question"
            elif r.case.answerable and not r.grounded:
                why = "critic did not accept the answer"
            elif r.case.answerable:
                why = f"answer is missing expected fact(s): {r.missing}"
            else:
                why = "answered instead of refusing (possible hallucination)"
            print(f"  ✗ {r.case.question}\n    why:    {why}")
            print(f"    answer: {r.answer[:200]}")
    print("=" * 78 + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide the per-node trace; print only the summary table",
    )
    args = parser.parse_args(argv[1:])

    try:
        config.require_anthropic_key()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(CASES)} question(s). Each one calls the API ...")
    results = [run_case(case, quiet=args.quiet) for case in CASES]
    _print_report(results)

    # Non-zero exit on any failure makes this usable as a CI / smoke test.
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
