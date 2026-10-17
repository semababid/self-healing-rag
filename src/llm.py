"""Claude chat-model factory.

A single place that constructs the LLM so every node uses identical settings.

Note on ``temperature``: current Claude models (Opus 5.5 / Opus 4.8 / Sonnet 5.5
/ etc.) reject an explicit ``temperature`` and would return HTTP 400, so we
simply don't set one. ``langchain-anthropic`` omits the field when it's ``None``.

Note on ``max_tokens``: on newer models (Opus 5.5, Sonnet 5.5, Haiku 5.5)
thinking is on by default and its tokens count toward ``max_tokens``. A tight
cap can cut off an answer, or the critic's JSON verdict, partway through. So we
use a generous ceiling. It is a limit, not a cost: short answers stay short.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src import config

_llm: ChatAnthropic | None = None


def get_llm() -> ChatAnthropic:
    """Return a cached ChatAnthropic instance configured from ``config``."""
    global _llm
    if _llm is None:
        config.require_anthropic_key()
        _llm = ChatAnthropic(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            timeout=60,
            # temperature intentionally omitted — see module docstring.
        )
    return _llm
