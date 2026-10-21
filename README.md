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

### Web page

```bash
streamlit run src/web.py
```

Opens at http://localhost:8501. Ask a question or click one of the three
examples (answered directly, needs a retry, not in the documents). The page
shows the answer with its sources and every attempt of the retry loop: what it
searched for, which files the chunks came from, what it answered and what the
critic said. Documents can be added from the sidebar; they go to
`data/uploads/` and can be removed again with one click.

### API

```bash
uvicorn src.api:app
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "How heavy a load can the N7 carry?"}'
```

`POST /ask` returns the answer, whether it was grounded, the number of retries,
the cited chunks with their text, every attempt and the time of each step.
`GET /health` says whether Ollama and the vector store are ready. The
interactive docs are at http://localhost:8000/docs.

### Command line

```bash
python -m src.main "What is the maximum flight time of the Nimbus N7?"
python -m src.main                       # ask questions until you type quit
python -m src.evaluate                   # score it on the test questions
python -m src.evaluate --retrieval       # only check the search (no model, a few seconds)
python -m src.graph                      # print the graph
```

Answers cite the chunks they used, like "The N7 carries a 2.1 kg payload [4]",
and `main` lists the file each cited chunk came from.

Add `-v` to `main` or `evaluate` to see each step as it happens (`[retrieve]`,
`[generate]`, `[critic]`, `[rewrite #1]`, ...) and how long each one took.
Without it you only get the answer.

Every question is also saved to `runs/runs.jsonl`, one JSON object per line:
the answer, the settings used, every attempt (query, files the chunks came
from, answer, critic verdict and reason) and the time of each step. To look at
the last one:

```bash
tail -n 1 runs/runs.jsonl | python -m json.tool
```

Settings are read from environment variables or a `.env` file; see
`.env.example`. Any Ollama model can be used by setting `LLM_MODEL`. Smaller
models are faster but make a weaker critic.

## Documents

`data/sample_docs` holds nine short documents about Nimbus Robotics, a company
that doesn't exist: a handbook, a support guide, release notes, safety rules,
benefits, an IT policy, and documents about two other drone models (the older
N5 and the upcoming N9) and the charging docks. Because the company is made up,
the model can't know the answers from its training data, so any answer that
isn't in the documents is a hallucination. The N5 and N9 documents are there
on purpose: they give prices, colours and battery types for products other
than the N7, which tests whether the system mixes products up.

Put your own `.md`, `.txt` or `.pdf` files in the folder and run
`python -m src.ingest` again to use them instead. Each chunk is stored with
its document's title in front (the first `#` heading, or the file name), so a
chunk from the middle of a document still says what it's about.

## Evaluation

`src/evaluate.py` runs 29 fixed questions:

- 6 worded close to the documents, like "What is the maximum flight time of
  the Nimbus N7?". The answer must contain the expected fact ("38").
- 18 worded differently from the documents, like "How long until a battery
  pack wears out?" for "rated for 500 charge cycles".
- 5 that sound plausible but aren't answered, like the N7's colours or its
  battery type. These must end in the "I don't have enough information"
  reply. For four of them the documents do give the answer for the N5 or N9,
  which makes it easy to answer with the wrong product's facts.

With `qwen3.5:9b` (thinking off) on an Apple M5 with 16 GB, all 29 pass, in
about four minutes in total:

```
29/29 passed
  answerable, answered correctly:  24/24
  unanswerable, refused:           5/5
  answered only after a retry:     2
```

The last line is the retry loop at work. For "How much does Nimbus Care+
cost?" and "When does the drone fly home by itself?", the first search didn't
bring up the right chunk, so the first reply was "I don't have enough
information". The rewritten queries ("Nimbus Care+ pricing plans and
subscription costs") found it, and the second answers were correct. Most of the time goes to the model: about half to the critic, 40%
to writing answers, 5% to rewriting queries and 2% to searching.

[EXPERIMENTS.md](EXPERIMENTS.md) has the measurements behind the default
settings, including a case where better search results gave worse answers.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace the model with a fake one that returns scripted replies, so
they don't need Ollama and finish in about a second. They cover each step of
the graph, full runs through the retry loop (answered first time, fixed by a
retry, gave up), loading and storing documents in a temporary folder, the
evaluation's scoring, the API, and the web page (run headless with Streamlit's
test runner). GitHub Actions runs them on every push, on Python 3.10 and 3.13.

The evaluation above is different: it uses the real model, so run it with
Ollama when you change prompts or models.

## Files

```
src/config.py       settings
src/ingest.py       load documents, split into chunks, store in Chroma
src/vectorstore.py  Chroma, embeddings, vector and hybrid search
src/llm.py          the Ollama chat model
src/logs.py         logging setup and the runs/runs.jsonl log
src/state.py        the data passed between steps
src/nodes.py        the steps: retrieve, (filter), generate, critic, rewrite, give up
src/graph.py        connects the steps into the graph
src/service.py      answers a question as plain data; used by the CLI, API and page
src/main.py         command-line interface
src/api.py          HTTP API (FastAPI)
src/web.py          web page (Streamlit)
src/evaluate.py     the test questions and scoring
tests/              tests for all of the above (pytest, no model needed)
```

[ROADMAP.md](ROADMAP.md) lists what's done and what's planned, and
[EXPERIMENTS.md](EXPERIMENTS.md) has the results of tuning the search.
