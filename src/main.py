"""Ask questions from the command line.

    python -m src.main "your question"       # one question
    python -m src.main                       # keep asking until you type quit
    python -m src.main -v "your question"    # also show each step

Every question is saved to runs/runs.jsonl.
"""

import argparse
import sys
import time
from pathlib import Path

from src.graph import answer_question
from src.llm import LLM_ERRORS, check_ollama
from src.logs import save_run, setup_logging, step_totals
from src.nodes import cited_documents
from src.vectorstore import require_store_built


def ask(question: str, verbose: bool) -> bool:
    start = time.perf_counter()
    try:
        state = answer_question(question)
    except LLM_ERRORS as err:
        print(f"Error: the model call failed ({err})", file=sys.stderr)
        return False
    seconds = time.perf_counter() - start
    save_run(question, state, seconds)

    if verbose:
        print()
    print(state["generation"])
    print(f"\n(grounded={state.get('grounded')}, retries={state.get('retries', 0)}, "
          f"{seconds:.1f}s)")
    if state.get("grounded"):
        documents = state["documents"]
        cited = cited_documents(state["generation"], documents)
        if cited:
            for doc in cited:
                print(f"[{documents.index(doc) + 1}] {Path(doc.metadata['source']).name}")
        else:
            sources = sorted({Path(doc.metadata["source"]).name for doc in documents})
            print("retrieved from:", ", ".join(sources))
    if verbose:
        totals = step_totals(state.get("timings", []))
        print("time per step:", ", ".join(f"{step} {secs:.1f}s" for step, secs in totals.items()))
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ask the documents a question.")
    parser.add_argument("question", nargs="*", help="leave out to ask several questions")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each step")
    args = parser.parse_args(argv[1:])
    setup_logging(args.verbose)

    try:
        check_ollama()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
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
