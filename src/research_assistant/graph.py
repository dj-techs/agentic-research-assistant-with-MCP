"""LangGraph wiring for the research assistant.

Topology::

    START -> planner -> router ─┬─> search    ─┐
                                ├─> code       ├─> router (loop) -> synthesizer -> END
                                └─> summarize  ┘

The router examines the next pending subtask and dispatches to the matching
specialist. When all subtasks are done, it routes to the synthesizer.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.planner import plan_node
from .agents.specialist import make_specialist_node
from .agents.synthesizer import synthesizer_node
from .state import GraphState


def _route_next(state: GraphState) -> str:
    cursor = state.get("cursor", 0)
    subtasks = state.get("subtasks", [])
    if cursor >= len(subtasks):
        return "synthesizer"
    return subtasks[cursor].specialist


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("planner", plan_node)
    g.add_node("search", make_specialist_node("search"))
    g.add_node("code", make_specialist_node("code"))
    g.add_node("summarize", make_specialist_node("summarize"))
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_next,
        {
            "search": "search",
            "code": "code",
            "summarize": "summarize",
            "synthesizer": "synthesizer",
        },
    )
    for s in ("search", "code", "summarize"):
        g.add_conditional_edges(
            s,
            _route_next,
            {
                "search": "search",
                "code": "code",
                "summarize": "summarize",
                "synthesizer": "synthesizer",
            },
        )
    g.add_edge("synthesizer", END)
    return g.compile()
