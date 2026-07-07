"""Central configuration for the Self-Healing RAG pipeline.

All tunable knobs live here so the rest of the code never hard-codes a path,
model name, or magic number. Values can be overridden with environment
variables (loaded from a local `.env` file if present).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file into os.environ (no-op if absent).
load_dotenv()

# --- Paths -----------------------------------------------------------------
# Project root = the directory that contains the `src/` package.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_COLLECTION = "self_healing_rag"

# --- Models ----------------------------------------------------------------
# Claude model used for BOTH answer generation and the critic/grader.
# Default to the most capable Opus tier; override via CLAUDE_MODEL to save cost
# (e.g. "claude-haiku-4-5" or "claude-sonnet-5").
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# Local embedding model — runs on your machine via sentence-transformers,
# so it costs nothing and needs no API key.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Retrieval / self-healing behaviour ------------------------------------
# How many chunks to pull from the vector store per query.
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# How many times the critic may reject an answer and trigger a re-try
# (query rewrite + re-retrieve) before we give up and say "I don't know".
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def require_anthropic_key() -> None:
    """Fail fast with a friendly message if the API key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Create a .env file (copy .env.example) and add your key, or run:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
