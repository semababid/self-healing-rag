# Roadmap

## Done

1. Project setup, config, sample documents
2. Ingestion: load, chunk, embed, store in Chroma
3. Model setup, graph state, and the retrieve / generate / critic / rewrite steps
4. The graph with the retry loop
5. Command-line interface
6. Evaluation script with 11 test questions; switched to a local Ollama model
7. Tests (using a fake model and fake embeddings) and GitHub Actions
8. Logging with `-v`, every attempt kept in the state, a JSON lines log of all
   runs, and timing for each step

## 9. Better retrieval

So far every answerable test question is found on the first try, so the retry
loop has only been tested on questions it can't answer.

| | Task | Done when |
|---|---|---|
| 9.1 | Harder questions worded differently from the documents ("how long until a battery pack wears out?") | some questions need a rewrite to be answered |
| 9.2 | A step that drops irrelevant chunks before generating | eval score compared with and without it |
| 9.3 | Hybrid search (keyword + vector) | eval score compared with vector only |
| 9.4 | On a retry, keep the earlier chunks or replace them? | decided by eval results |
| 9.5 | Tune chunk size, overlap and k | chosen values written down with their scores |
| 9.6 | Numbered citations in answers | the answer points to the chunks it used |

## 10. Interface

| | Task | Done when |
|---|---|---|
| 10.1 | FastAPI endpoint `/ask` | returns answer, verdict, retries and sources as JSON |
| 10.2 | Streamlit page showing each attempt | the retry loop is visible in the browser |
| 10.3 | Upload a document from the page | no command line needed to add documents |

## 11. Experiments

| | Task | Done when |
|---|---|---|
| 11.1 | `MAX_RETRIES` = 0, 1, 2, 3 | table of score and time per setting |
| 11.2 | Critic on vs. off | how many hallucinations the critic catches |
| 11.3 | Compare models: a small one (around 4B), `qwen3.5:9b`, and a larger one if it fits in memory | score and time per model |
| 11.4 | Thinking mode on vs. off | does the slower mode change the score? |
| 11.5 | Repeat each setting a few times | results show the spread, not one run |

## 12. Packaging and write-up

| | Task | Done when |
|---|---|---|
| 12.1 | `pyproject.toml` and a lock file | a fresh install gets the same versions |
| 12.2 | Report: how it works, one run traced step by step, results from phase 11 | ready to hand in |
| 12.3 | Licence, if the repo should be reusable | `LICENSE` file |
