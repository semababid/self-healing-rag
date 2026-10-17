"""Wires the nodes into a graph with a retry loop.

retrieve -> generate -> critic, then the critic's verdict decides: accept the
answer, rewrite the query and go back to retrieve, or give up.
"""

import time

from langgraph.graph import END, START, StateGraph

from src import config, nodes
from src.state import GraphState

STEPS = {
    "retrieve": nodes.retrieve,
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
        graph.add_node(name, timed(name, step))

    graph.add_edge(START, "retrieve")
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
    # One run is 3 steps, plus 4 per retry, plus give_up. A tight step limit
    # stops a routing bug from looping (LangGraph's own limit is ~10,000).
    limit = 3 + 4 * config.MAX_RETRIES + 1 + 5
    return app.invoke(
        {"question": question, "query": question, "retries": 0},
        {"recursion_limit": limit},
    )


if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
