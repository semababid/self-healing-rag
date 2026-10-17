"""Wires the nodes into a graph with a retry loop.

retrieve -> generate -> critic, then the critic's verdict decides: accept the
answer, rewrite the query and go back to retrieve, or give up. With
FILTER_CHUNKS=1 a filter step sits between retrieve and generate.
"""

import time

from langgraph.graph import END, START, StateGraph

from src import config, nodes
from src.state import GraphState

STEPS = {
    "retrieve": nodes.retrieve,
    "filter_chunks": nodes.filter_chunks,
    "generate": nodes.generate,
    "critic": nodes.critic,
    "transform_query": nodes.transform_query,
    "give_up": nodes.give_up,
}


def timed(name, step):
    """Wrap a step so it also reports how long it took."""
    def run(state):
        start = time.perf_counter()
        update = step(state)
        seconds = round(time.perf_counter() - start, 3)
        return {**update, "timings": [{"step": name, "seconds": seconds}]}
    return run


def build_graph():
    graph = StateGraph(GraphState)
    for name, step in STEPS.items():
        if name != "filter_chunks" or config.FILTER_CHUNKS:
            graph.add_node(name, timed(name, step))

    graph.add_edge(START, "retrieve")
    if config.FILTER_CHUNKS:
        graph.add_edge("retrieve", "filter_chunks")
        graph.add_edge("filter_chunks", "generate")
    else:
        graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")
    graph.add_conditional_edges(
        "critic",
        nodes.decide_after_critic,
        {"accept": END, "retry": "transform_query", "give_up": "give_up"},
    )
    graph.add_edge("transform_query", "retrieve")  # the loop
    graph.add_edge("give_up", END)
    return graph.compile()


app = build_graph()


def answer_question(question: str) -> GraphState:
    # A first attempt is 3 or 4 steps, each retry adds a rewrite on top, and
    # give_up is one more. A tight step limit stops a routing bug from looping
    # (LangGraph's own limit is about 10,000).
    attempt = 4 if config.FILTER_CHUNKS else 3
    limit = attempt + (attempt + 1) * config.MAX_RETRIES + 1 + 5
    return app.invoke(
        {"question": question, "query": question, "retries": 0},
        {"recursion_limit": limit},
    )


if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
