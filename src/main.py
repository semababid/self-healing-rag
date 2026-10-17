"""Ask questions from the command line.

    python -m src.main "your question"    # one question
    python -m src.main                    # keep asking until you type quit
"""

import sys
from pathlib import Path

from src.graph import answer_question
from src.llm import LLM_ERRORS, check_ollama
from src.vectorstore import require_store_built


def ask(question: str) -> bool:
    try:
        state = answer_question(question)
    except LLM_ERRORS as err:
        print(f"Error: the model call failed ({err})", file=sys.stderr)
        return False

    print("\n" + state["generation"])
    print(f"\n(grounded={state.get('grounded')}, retries={state.get('retries', 0)})")
    if state.get("grounded"):
        sources = sorted({Path(doc.metadata["source"]).name for doc in state["documents"]})
        print("retrieved from:", ", ".join(sources))
    return True


def main(argv: list[str]) -> int:
    try:
        check_ollama()
        require_store_built()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if len(argv) > 1:
        return 0 if ask(" ".join(argv[1:])) else 1

    print("Ask a question, or type quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            ask(question)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
