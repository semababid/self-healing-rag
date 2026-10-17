"""Assemble the self-healing RAG workflow as a cyclical LangGraph.

The graph is what makes this "self-healing": after the critic grades an answer,
a conditional edge either accepts it, loops back to retrieve with a rewritten
query, or gives up gracefully. The ``transform_query -> retrieve`` edge is the
cycle that a plain linear chain cannot express.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src import config, nodes
from src.state import GraphState


def build_graph():
    """Build and compile the self-healing RAG state graph."""
    graph = StateGraph(GraphState)

    # Register the nodes.
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("critic", nodes.critic)
    graph.add_node("transform_query", nodes.transform_query)
    graph.add_node("give_up", nodes.give_up)

    # Linear spine: retrieve -> generate -> critic.
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    # The decision point. decide_after_critic returns one of these keys.
    graph.add_conditional_edges(
        "critic",
        nodes.decide_after_critic,
        {
            "accept": END,              # grounded → we're done
            "retry": "transform_query",  # rejected, retries left → self-heal
            "give_up": "give_up",        # rejected, out of retries → honest refusal
        },
    )

    # The cycle: a rewritten query goes back through retrieval.
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("give_up", END)

    return graph.compile()


# Compile once at import; reuse across questions.
app = build_graph()


def _recursion_limit() -> int:
    """Upper bound on node executions for one run, derived from MAX_RETRIES.

    A full run is the first pass (retrieve, generate, critic), 4 steps per
    retry (transform_query, retrieve, generate, critic), and a final give_up.
    Setting the limit explicitly does two jobs:

    * LangGraph 0.x (still allowed by requirements.txt) aborts after 25 steps,
      so MAX_RETRIES >= 6 would crash instead of giving up gracefully.
    * LangGraph 1.x defaults to ~10,000 steps, so a routing bug could loop and
      burn API calls; a tight bound acts as a circuit breaker.
    """
    return 3 + 4 * config.MAX_RETRIES + 1 + 5  # +5 safety margin


def answer_question(question: str) -> GraphState:
    """Run the full pipeline for one question and return the final state."""
    initial: GraphState = {"question": question, "query": question, "retries": 0}
    return app.invoke(initial, {"recursion_limit": _recursion_limit()})


def print_graph() -> None:
    """Print an ASCII diagram of the compiled graph (handy for the report)."""
    print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    # `python -m src.graph` shows the workflow structure without calling the API.
    print_graph()
