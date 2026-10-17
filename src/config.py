"""Settings. Each one can be overridden with an environment variable or .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_COLLECTION = "self_healing_rag"

# Local model served by Ollama. It answers, grades and rewrites queries.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Thinking mode makes each call several times slower, so it's off by default.
LLM_THINKING = os.getenv("LLM_THINKING", "0") == "1"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# How often a rejected answer leads to a new query before we give up.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
