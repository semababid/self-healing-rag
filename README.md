# 🔁 Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that **critiques its own answers and retries**
instead of blindly retrieving-and-generating. Built as a **stateful, cyclical workflow** with
[LangGraph](https://langchain-ai.github.io/langgraph/) — not a simple linear chain.

## The idea

When a user asks a question, the pipeline:

1. **Retrieves** relevant document chunks from a vector store.
2. **Generates** an answer with an LLM, grounded only in those chunks.
3. **Critiques** the answer: *is it actually supported by the retrieved chunks, or did the
   model hallucinate?*
4. If the critic **rejects** the answer → it **reformulates the query and re-retrieves**, up to
   a retry limit. If it still can't find support, it **gracefully says "I don't have enough
   information"** instead of making things up.

```
          ┌─────────────┐
  query → │  retrieve   │ ◄────────────────────┐
          └──────┬──────┘                      │
                 ▼                              │ (rewritten query)
          ┌─────────────┐                      │
          │  generate   │                      │
          └──────┬──────┘              ┌────────────────┐
                 ▼                     │ transform_query│
          ┌─────────────┐   reject    └────────────────┘
          │   critic    │──── & retries left ───┘
          └──────┬──────┘
        accept   │   reject & out of retries
                 ▼          ▼
              ┌───────────────┐
              │      END      │  (answer, or "I don't know")
              └───────────────┘
```

## Tech stack

| Concern            | Choice                                                        |
| ------------------ | ------------------------------------------------------------- |
| Orchestration      | LangGraph (`StateGraph` with a conditional loop)              |
| LLM (gen + critic) | Anthropic Claude via `langchain-anthropic`                    |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` (local, free)        |
| Vector store       | Chroma (persistent, on disk)                                  |

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env        # then edit .env and paste your key

# 4. Build the vector store from the sample docs
python -m src.ingest

# 5. Ask a question
python -m src.main "What is the maximum flight time of the Nimbus N7?"

# 6. (Optional) Score the self-healing behaviour on a curated question set
python -m src.evaluate            # full trace + report
python -m src.evaluate --quiet    # just the summary table
```

## Does the self-healing actually work?

`src/evaluate.py` turns the project's central claim — *answers what the docs
support, refuses to hallucinate on what they don't* — into a measurable check.
The sample docs are fictional on purpose, so any confident answer to a question
they don't cover can only be a hallucination.

It runs a fixed set of questions through the real graph and scores each one:

- **answerable** questions (e.g. *"How long is the N7 warranty?"*) must produce a
  grounded answer the critic accepts;
- **unanswerable** questions (e.g. *"What is the N7's maximum altitude?"*) must
  trigger the honest fallback instead of a made-up answer.

The report breaks results into grounded-answers vs. honest-refusals and flags
any case that only passed *after* a query rewrite + retry (the self-healing path
in action). The script exits non-zero on any failure, so it also works as a
smoke test.

## Project layout

```
self-healing-rag/
├── data/sample_docs/     # source documents (fictional, so hallucination is testable)
├── src/
│   ├── config.py         # all settings & env overrides
│   ├── ingest.py         # load → chunk → embed → persist to Chroma   (Phase 2)
│   ├── vectorstore.py    # open the store / build a retriever          (Phase 2)
│   ├── llm.py            # Claude chat-model factory                   (Phase 3)
│   ├── state.py          # the graph's shared state                    (Phase 3)
│   ├── nodes.py          # retrieve / generate / critic / rewrite      (Phase 3)
│   ├── graph.py          # wire the cyclical LangGraph                 (Phase 4)
│   ├── main.py           # CLI entrypoint                              (Phase 5)
│   └── evaluate.py       # scored eval of the self-healing behaviour   (Phase 6)
└── requirements.txt
```

Built step by step — see the git history for each phase.
