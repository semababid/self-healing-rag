# Roadmap

Phases 1 to 6 are done. Phases 7 to 12 are planned, and each one is split into
sub-phases small enough to finish and commit separately. A sub-phase is
finished when its "done when" condition holds.

## Done

| Phase | What it added |
| ----- | ------------- |
| 1 | Project scaffold, config, sample documents |
| 2 | Ingestion: load, chunk, embed, persist to Chroma |
| 3 | LLM factory, graph state, the retrieve/generate/critic/rewrite nodes |
| 4 | The cyclical LangGraph with the retry loop |
| 5 | CLI entrypoint |
| 6 | Evaluation harness (`src/evaluate.py`) |
| 6.1 | Fixes for gaps in phases 1 to 6 (listed below) |

### Phase 6.1: gaps fixed in earlier phases

- The corpus was 4 chunks and the retriever fetched 4, so every query returned
  everything and a rewritten query could never change the result. Four more
  documents bring it to 16 chunks.
- The critic used forced tool calling for structured output. Claude Opus 5.5,
  Sonnet 5.5 and Fable 5.1 reject that, so it now uses `method="json_schema"`.
- `max_tokens=1024` could cut off answers on models that think by default. It
  is now 16000.
- `main.py` ran even with no vector store, spending API calls before ending in
  "I don't know". It now checks the key and the store first.
- Answers now list the source files they came from.
- The graph sets an explicit recursion limit based on `MAX_RETRIES`. LangGraph
  0.x stopped at 25 steps, which crashed runs with `MAX_RETRIES >= 6`.
- The evaluation now checks that answers contain the expected fact. The Berlin
  office case was replaced, because the docs do answer it ("No").
- `requirements.txt` now matches the imports. It drops the unused `langchain`
  and the deprecated `langchain-community`, and adds `langchain-text-splitters`
  and `grandalf` (needed by `python -m src.graph`).

## Phase 7: automated tests and CI

Right now the only check is `evaluate.py`, which needs an API key and costs
money. This phase adds tests that run offline in seconds.

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 7.1 | Add `requirements-dev.txt` (pytest) and a `tests/` folder with shared fixtures, including a stub LLM | `pytest` runs and collects zero tests without errors |
| 7.2 | Unit tests for routing: `decide_after_critic` (accept, retry, give up at the limit), `transform_query` raising the retry count, `give_up` returning the fallback | All three outcomes are covered at the retry boundary |
| 7.3 | Graph tests with stubbed nodes: accept first try, heal on a later try, give up, and a high `MAX_RETRIES` | Full runs pass with no API key |
| 7.4 | Ingestion tests on a temporary folder: Markdown and PDF loading, chunking, re-ingest does not duplicate chunks | Tests pass without touching `chroma_db/` |
| 7.5 | Tests for the evaluation scoring (`Result.passed`, `Result.missing`) | Each failure reason in the report has a test |
| 7.6 | GitHub Actions workflow that runs the tests on every push | A green check on the GitHub repo |
| 7.7 | Optional: `ruff` for linting and formatting, run in CI | CI fails on lint errors |

## Phase 8: observability

The nodes print a trace to the terminal, and nothing is kept after a run.

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 8.1 | Replace `print` with the `logging` module and add a `--verbose` flag | Quiet by default, full trace with `--verbose` |
| 8.2 | Add an `attempts` list to the graph state: query, retrieved sources, critic verdict and reason for each try | The final state shows every attempt |
| 8.3 | Save each run as one line of JSON in a `runs/` folder | A run can be inspected after the fact |
| 8.4 | Record token usage per call and estimate cost per question | The eval report shows total tokens and cost |
| 8.5 | Optional: LangSmith tracing, switched on by environment variables | Runs show up in LangSmith when enabled |

## Phase 9: retrieval quality

The critic can only accept what retrieval finds. This phase measures how often
the rewrite loop helps and then improves retrieval.

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 9.1 | Add "hard" eval questions worded differently from the docs (for example "how long until a battery pack wears out?" for the 500-cycle rating) | Some cases fail on the first query and need a rewrite |
| 9.2 | Add a grade-documents node that drops irrelevant chunks before generation | Fewer off-topic chunks reach the generator, measured on the eval |
| 9.3 | Hybrid search: BM25 keyword search combined with vector search | Eval score with hybrid search is reported next to vector-only |
| 9.4 | Optional: a cross-encoder reranker | Same comparison as 9.3 |
| 9.5 | Decide whether chunks from earlier attempts are kept or replaced on a retry, and test both | The choice is backed by eval numbers |
| 9.6 | Tune `CHUNK_SIZE`, `CHUNK_OVERLAP` and `RETRIEVER_K` with the eval | Chosen values are written down with their scores |
| 9.7 | Inline citations: the answer marks claims with `[1]`, `[2]` that map to sources | The CLI prints the answer with numbered sources |

## Phase 10: interface

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 10.1 | A FastAPI `/ask` endpoint returning the answer, grounded flag, retries, sources and attempts | `curl` against it returns JSON |
| 10.2 | A Streamlit or Gradio page that shows the answer and each attempt, with a grounded/refused label | The self-healing loop is visible in a browser |
| 10.3 | Upload a document from the UI and ingest it | A new file can be asked about without the command line |

## Phase 11: experiments

These produce the numbers for a report or presentation.

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 11.1 | Ablation on `MAX_RETRIES` (0, 1, 2, 3): how much does self-healing improve the score, and what does it cost? | A table of score, retries used and cost per setting |
| 11.2 | Critic on vs. off: how many hallucinations does the critic catch? | Hallucination rate with and without the critic |
| 11.3 | Model comparison (Opus 4.8, Opus 5.5, Sonnet 5.5, Haiku 5.5) on the eval | Score, cost and latency per model, and a pick for the default |
| 11.4 | Run each setting several times, since LLM output varies between runs | Results report a mean and spread, not a single run |

## Phase 12: packaging and write-up

| Sub-phase | What | Done when |
| --------- | ---- | --------- |
| 12.1 | `pyproject.toml` and a locked dependency file (pip-tools or uv) | A fresh clone installs the same versions every time |
| 12.2 | Optional: a Dockerfile | `docker run` answers a question |
| 12.3 | Architecture write-up: the graph diagram, one full self-healing run traced step by step, and the Phase 11 results | A document ready to hand in |
| 12.4 | Add a licence, if you want the repo to be reusable | `LICENSE` file present |

## Open decisions

- Default model. The default is still `claude-opus-4-8`. The code now works with
  the 5.5 models, but switching should wait for the Phase 11.3 comparison.
