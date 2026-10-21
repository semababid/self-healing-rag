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
10. Interface: a FastAPI endpoint (`POST /ask`), a Streamlit page that shows
    every attempt of the retry loop, and document upload from the page
11. Packaging: Docker (`docker compose up` runs Ollama, the web page and the API) and a
    README that opens with a demo GIF and the results; MIT licence, public repo;
    lock files with the exact version of every package

## 12. Experiments

The first three answer the questions most likely to come up about the project.

| | Task | Done when |
|---|---|---|
| 12.1 | Critic on vs. off | how many hallucinations the critic catches |
| 12.2 | `MAX_RETRIES` = 0, 1, 2, 3 | how many answers the retry loop rescues, and what it costs in time |
| 12.3 | Compare models: a small one (around 4B) and `qwen3.5:9b` | score and time per model |

Later:

| | Task | Done when |
|---|---|---|
| 12.4 | Thinking mode on vs. off | does the slower mode change the score? |
| 12.5 | Repeat each setting a few times | results show the spread, not one run |
| 12.6 | Shorter critic replies: the critic takes over half the run time, mostly writing its explanation | critic time drops without losing score |
| 12.7 | Rewritten queries sometimes use search engine syntax (quotes, `OR`) that means nothing to vector search | rewrite prompt asks for plain words; retries compared |
| 12.8 | Re-test hybrid search and the chunk filter with a smaller model | they may help a weaker model even though they didn't help `qwen3.5:9b` |
