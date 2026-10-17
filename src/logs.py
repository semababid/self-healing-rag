"""Logging setup, and the run log: one JSON line per question in runs/runs.jsonl.

To look at the last run:

    tail -n 1 runs/runs.jsonl | python -m json.tool
"""

import json
import logging
from datetime import datetime

from src import config


def setup_logging(verbose: bool) -> None:
    # Only this project's messages get through at INFO; libraries such as
    # httpx (which logs every request) stay at WARNING.
    logging.basicConfig(level=logging.WARNING, format="  %(message)s")
    logging.getLogger("src").setLevel(logging.INFO if verbose else logging.WARNING)


def step_totals(timings: list[dict]) -> dict[str, float]:
    """Total seconds per step, e.g. {"retrieve": 0.2, "generate": 9.1, ...}."""
    totals: dict[str, float] = {}
    for timing in timings:
        totals[timing["step"]] = totals.get(timing["step"], 0.0) + timing["seconds"]
    return totals


def save_run(question: str, state: dict, seconds: float, **extra) -> None:
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.LLM_MODEL,
        "thinking": config.LLM_THINKING,
        "max_retries": config.MAX_RETRIES,
        "k": config.RETRIEVER_K,
        "question": question,
        "answer": state.get("generation"),
        "grounded": state.get("grounded"),
        "retries": state.get("retries", 0),
        "seconds": round(seconds, 2),
        **extra,
        "attempts": state.get("attempts", []),
        "timings": state.get("timings", []),
    }
    config.RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
