"""Claude chat-model factory.

A single place that constructs the LLM so every node uses identical settings.

Note on ``temperature``: current Claude models (Opus 4.8 / Sonnet 5 / etc.)
reject an explicit ``temperature`` and would return HTTP 400, so we simply
don't set one. ``langchain-anthropic`` omits the field when it's ``None``.
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
            max_tokens=1024,
            timeout=60,
            # temperature intentionally omitted — see module docstring.
        )
    return _llm
