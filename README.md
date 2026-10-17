# Self-healing RAG

A question-answering system over a small set of documents that checks its own
answers before returning them. After drafting an answer, a second model call
(the critic) checks whether every fact in it comes from the retrieved text. If
it doesn't, the system rewrites the search query and tries again. After two
failed retries it says it doesn't know instead of guessing.

The workflow is a LangGraph graph with a loop:

```
question -> retrieve -> generate -> critic --grounded--> answer
               ^                      |
               |                  rejected
               |                      v
               +-- rewrite query <-- retries left? --no--> "I don't know"
```

Everything runs on your own machine: the language model through Ollama, the
embeddings with sentence-transformers, and the vector store with Chroma. No API
key or account is needed.

## Setup

You need Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3.5:9b   # about 6.6 GB
ollama serve             # skip if Ollama is already running

python -m src.ingest     # builds the vector store
```

`ingest` downloads the embedding model (about 90 MB) the first time.

## Usage

```bash
python -m src.main "What is the maximum flight time of the Nimbus N7?"
python -m src.main               # ask questions until you type quit
python -m src.evaluate --quiet   # score it on the test questions
python -m src.graph              # print the graph
```

Each run prints the steps it takes (`[retrieve]`, `[generate]`, `[critic]`,
`[rewrite #1]`, ...), so you can watch the retry loop work.

Settings are read from environment variables or a `.env` file; see
`.env.example`. Any Ollama model can be used by setting `LLM_MODEL`. Smaller
models are faster but make a weaker critic.

## Documents

`data/sample_docs` holds six short documents about Nimbus Robotics, a company
that doesn't exist: a handbook, a support guide, release notes, safety rules,
benefits and an IT policy. Because the company is made up, the model can't know
the answers from its training data, so any answer that isn't in the documents
is a hallucination. Put your own `.md`, `.txt` or `.pdf` files in the folder and
run `python -m src.ingest` again to use them instead.

## Evaluation

`src/evaluate.py` runs 11 fixed questions. Six are answered in the documents,
and the answer must contain the expected fact (for example "38" for the
N7's flight time). Five sound plausible but aren't covered (the N7's colours,
its maximum altitude, the CFO's name, ...), and those must end in the "I don't
have enough information" reply.

With `qwen3.5:9b` (thinking off) on an Apple M5 with 16 GB, all 11 pass, in
about 80 seconds in total:

```
11/11 passed
  answerable, answered correctly:  6/6
  unanswerable, refused:           5/5
  answered only after a retry:     0
```

So far no answerable question has needed a retry, because retrieval finds the
right chunk on the first try. The retry loop has only been exercised on the
unanswerable questions, where it tries two rewritten queries before giving up.
Harder questions are planned (see the roadmap, 9.1).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace the model with a fake one that returns scripted replies, so
they don't need Ollama and finish in under a second. They cover each step of
the graph, full runs through the retry loop (answered first time, fixed by a
retry, gave up), loading and storing documents in a temporary folder, and the
evaluation's scoring. GitHub Actions runs them on every push, on Python 3.10
and 3.13.

The evaluation above is different: it uses the real model, so run it with
Ollama when you change prompts or models.

## Files

```
src/config.py       settings
src/ingest.py       load documents, split into chunks, store in Chroma
src/vectorstore.py  Chroma and embedding model access
src/llm.py          the Ollama chat model
src/state.py        the data passed between steps
src/nodes.py        the steps: retrieve, generate, critic, rewrite, give up
src/graph.py        connects the steps into the graph
src/main.py         command-line interface
src/evaluate.py     the test questions and scoring
tests/              unit and graph tests (pytest, no model needed)
```

[ROADMAP.md](ROADMAP.md) lists what's done and what's planned.
