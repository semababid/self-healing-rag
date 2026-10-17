"""The chat model: a local model served by Ollama."""

import httpx
import ollama
from langchain_ollama import ChatOllama

from src import config

# What a failed model call can raise: server down, or model missing.
LLM_ERRORS = (ConnectionError, httpx.HTTPError, ollama.ResponseError)

_llm = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,  # repeatable answers, so eval runs can be compared
            reasoning=config.LLM_THINKING,
        )
    return _llm


def check_ollama() -> None:
    """Raise RuntimeError if Ollama isn't running or the model isn't downloaded."""
    try:
        models = ollama.Client(host=config.OLLAMA_BASE_URL).list().models
    except ConnectionError:
        raise RuntimeError(
            f"Can't reach Ollama at {config.OLLAMA_BASE_URL}. Start it with:\n"
            "    ollama serve"
        ) from None

    name = config.LLM_MODEL
    if ":" not in name:
        name += ":latest"
    if name not in {m.model for m in models}:
        raise RuntimeError(
            f"The model {config.LLM_MODEL} isn't downloaded. Get it with:\n"
            f"    ollama pull {config.LLM_MODEL}"
        )
