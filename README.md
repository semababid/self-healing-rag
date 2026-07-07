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
```

## Project layout

```
sa/
├── data/sample_docs/     # source documents (fictional, so hallucination is testable)
├── src/
│   ├── config.py         # all settings & env overrides
│   ├── ingest.py         # load → chunk → embed → persist to Chroma   (Phase 2)
│   ├── vectorstore.py    # open the store / build a retriever          (Phase 2)
│   ├── llm.py            # Claude chat-model factory                   (Phase 3)
│   ├── state.py          # the graph's shared state                    (Phase 3)
│   ├── nodes.py          # retrieve / generate / critic / rewrite      (Phase 3)
│   ├── graph.py          # wire the cyclical LangGraph                 (Phase 4)
│   └── main.py           # CLI entrypoint                              (Phase 5)
└── requirements.txt
```

Built step by step — see the git history for each phase.
