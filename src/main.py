"""Ask questions from the command line.

    python -m src.main "your question"       # one question
    python -m src.main                       # keep asking until you type quit
    python -m src.main -v "your question"    # also show each step

Every question is saved to runs/runs.jsonl.
"""

import argparse
import sys

from src.llm import LLM_ERRORS
from src.logs import setup_logging, step_totals
from src.service import answer, problem


def ask(question: str, verbose: bool) -> bool:
    try:
        result = answer(question, source="cli")
    except LLM_ERRORS as err:
        print(f"Error: the model call failed ({err})", file=sys.stderr)
        return False

    if verbose:
        print()
    print(result["answer"])
    print(f"\n(grounded={result['grounded']}, retries={result['retries']}, "
          f"{result['seconds']:.1f}s)")
    if result["sources_cited"]:
        for source in result["sources"]:
            print(f"[{source['number']}] {source['file']}")
    elif result["sources"]:
        print("retrieved from:", ", ".join(sorted({s["file"] for s in result["sources"]})))
    if verbose:
        totals = step_totals(result["timings"])
        print("time per step:", ", ".join(f"{step} {secs:.1f}s" for step, secs in totals.items()))
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ask the documents a question.")
    parser.add_argument("question", nargs="*", help="leave out to ask several questions")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each step")
    args = parser.parse_args(argv[1:])
    setup_logging(args.verbose)

    issue = problem()
    if issue:
        print(f"Error: {issue}", file=sys.stderr)
        return 1

    if args.question:
        return 0 if ask(" ".join(args.question), args.verbose) else 1

    print("Ask a question, or type quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            ask(question, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
