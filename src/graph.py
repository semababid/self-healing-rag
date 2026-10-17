"""Wires the nodes into a graph with a retry loop.

retrieve -> generate -> critic, then the critic's verdict decides: accept the
answer, rewrite the query and go back to retrieve, or give up.
"""

from langgraph.graph import END, START, StateGraph

from src import config, nodes
from src.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("critic", nodes.critic)
    graph.add_node("transform_query", nodes.transform_query)
    graph.add_node("give_up", nodes.give_up)

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
