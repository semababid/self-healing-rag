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
9. Retrieval: 29 test questions, three documents about other products, a
   search-only benchmark, chunk titles, hybrid search, a chunk filter and
   keeping chunks across retries (both off by default), and citations. The
   measurements are in [EXPERIMENTS.md](EXPERIMENTS.md).

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
| 11.6 | Shorter critic replies: the critic takes over half the run time, mostly writing its explanation | critic time drops without losing score |
| 11.7 | Rewritten queries sometimes use search engine syntax (quotes, `OR`) that means nothing to vector search | rewrite prompt asks for plain words; retries compared |
| 11.8 | Re-test hybrid search and the chunk filter with a smaller model | they may help a weaker model even though they didn't help `qwen3.5:9b` |

## 12. Packaging and write-up

| | Task | Done when |
|---|---|---|
| 12.1 | `pyproject.toml` and a lock file | a fresh install gets the same versions |
| 12.2 | Report: how it works, one run traced step by step, results from phase 11 | ready to hand in |
| 12.3 | Licence, if the repo should be reusable | `LICENSE` file |
