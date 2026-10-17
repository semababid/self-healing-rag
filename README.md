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

# 5. Ask a question (the answer lists the source files it came from)
python -m src.main "What is the maximum flight time of the Nimbus N7?"

# 6. (Optional) Score the self-healing behaviour on a curated question set
python -m src.evaluate            # full trace + report
python -m src.evaluate --quiet    # just the summary table

# 7. (Optional) Print the workflow diagram. No API key needed.
python -m src.graph
```

`main` and `evaluate` check for the API key and the vector store before doing
anything else. If either is missing, they stop with a one-line explanation
instead of spending API calls on a run that can only end in "I don't know".

## Choosing a model

The generator and the critic share one model, set by `CLAUDE_MODEL` in `.env`.
The default is `claude-opus-4-8`. Current options are `claude-opus-5-5`, or
`claude-sonnet-5-5` and `claude-haiku-5-5` to save cost. The critic uses native
structured output (`method="json_schema"`), which every current model accepts.
The older tool-forcing approach fails on the 5.5 models.

## Does the self-healing actually work?

`src/evaluate.py` tests the project's main claim: the pipeline answers what the
documents support and refuses when they don't. The sample documents describe a
made-up company, so a confident answer to a question they don't cover can only
be a hallucination.

The script runs a fixed set of questions through the real graph and scores each
one. Answerable questions, such as *"How long is the N7 warranty?"*, must get a
grounded answer that the critic accepts and that contains the expected fact
("24"). Unanswerable questions, such as *"What is the N7's maximum altitude?"*,
must end in the honest fallback.

The report counts grounded answers and honest refusals separately. It also
flags any case that passed only after a query rewrite and retry, which is the
self-healing path at work. For each failure it prints the answer and the reason
it failed. The script exits non-zero if anything fails, so it doubles as a
smoke test.

The corpus is six documents split into 16 chunks, and the retriever fetches 4
per query. Retrieval therefore has to choose, which is what gives a rewritten
query something to improve.

## Project layout

```
self-healing-rag/
├── data/sample_docs/     # 6 fictional company docs (so hallucination is testable)
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
├── ROADMAP.md            # planned phases 7 to 11, split into sub-phases
└── requirements.txt
```

The project was built one phase at a time, and the git history has a commit for
each phase. [ROADMAP.md](ROADMAP.md) lists what comes next.
