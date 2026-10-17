"""Command-line entrypoint for the Self-Healing RAG pipeline.

Usage:
    python -m src.main "your question here"     # one-shot
    python -m src.main                          # interactive prompt

Make sure you've built the vector store first:
    python -m src.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

from src import config
from src.graph import answer_question
from src.vectorstore import require_store_built


def _sources(documents: list[Document]) -> str:
    """List the distinct source files the retrieved chunks came from."""
    names = sorted({Path(d.metadata.get("source", "?")).name for d in documents})
    return ", ".join(names) or "none"


def _print_result(question: str, state: dict) -> None:
    """Pretty-print the final answer plus a little run metadata."""
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(state.get("generation", "(no answer produced)"))
    print("-" * 70)
    print(
        f"grounded={state.get('grounded')}  "
        f"retries={state.get('retries', 0)}  "
        f"chunks_used={len(state.get('documents', []))}"
    )
    # Only cite sources for an accepted answer. A refusal didn't use them.
    if state.get("grounded"):
        print(f"sources: {_sources(state.get('documents', []))}")
    print("=" * 70 + "\n")


def _run_once(question: str) -> None:
    print(f"\nAsking: {question}\n")
    state = answer_question(question)
    _print_result(question, state)


def main(argv: list[str]) -> int:
    # Fail fast on setup problems, before loading models or calling the API.
    try:
        config.require_anthropic_key()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if len(argv) > 1:
        _run_once(" ".join(argv[1:]))
        return 0

    # Interactive mode.
    print("Self-Healing RAG — type a question (or 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            _run_once(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
