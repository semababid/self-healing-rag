"""Settings. Each one can be overridden with an environment variable or .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"  # documents added through the web page
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_COLLECTION = "self_healing_rag"

# Local model served by Ollama. It answers, grades and rewrites queries.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Thinking mode makes each call several times slower, so it's off by default.
LLM_THINKING = os.getenv("LLM_THINKING", "0") == "1"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _int(name: str, default: int) -> int:
    # An empty value (e.g. "CHUNK_SIZE=" in .env) means "use the default".
    return int(os.getenv(name) or default)


CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)
RETRIEVER_K = _int("RETRIEVER_K", 4)
# "vector" (embedding similarity) or "hybrid" (embeddings plus keyword search)
SEARCH = os.getenv("SEARCH") or "vector"
if SEARCH not in ("vector", "hybrid"):
    raise ValueError(f"SEARCH must be 'vector' or 'hybrid', not {SEARCH!r}")
# Ask the model to drop irrelevant chunks before answering (one extra call per attempt).
FILTER_CHUNKS = os.getenv("FILTER_CHUNKS", "0") == "1"
# On a retry, keep the chunks found so far as well as the new ones.
KEEP_CHUNKS = os.getenv("KEEP_CHUNKS", "0") == "1"

# How often a rejected answer leads to a new query before we give up.
MAX_RETRIES = _int("MAX_RETRIES", 2)

# Every question asked through main or evaluate is appended here as one JSON line.
RUNS_FILE = Path(os.getenv("RUNS_FILE", ROOT_DIR / "runs" / "runs.jsonl"))
