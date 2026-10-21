# The app: web page (default), API or command line. The language model runs in
# Ollama, either in its own container or on the host (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# The CPU build of PyTorch. The default one includes CUDA and is several GB larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model during the build, so the container starts
# quickly and doesn't need to reach Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src
COPY data/sample_docs ./data/sample_docs
COPY .streamlit ./.streamlit

# Build the search index from the sample documents, then run as a normal user.
RUN python -m src.ingest \
    && useradd --create-home app \
    && chown -R app /app
USER app

EXPOSE 8501 8000
CMD ["streamlit", "run", "src/web.py", "--server.address=0.0.0.0", "--server.headless=true"]
