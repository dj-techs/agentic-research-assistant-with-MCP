"""Test the graph router in isolation (no Claude calls)."""

from __future__ import annotations

from research_assistant.graph import _route_next
from research_assistant.state import Subtask


def test_router_dispatches_to_first_pending_specialist():
    state = {
        "subtasks": [
            Subtask(id=1, specialist="search", instruction="x"),
            Subtask(id=2, specialist="code", instruction="y"),
        ],
        "cursor": 0,
    }
    assert _route_next(state) == "search"


def test_router_advances_after_completion():
    state = {
        "subtasks": [
            Subtask(id=1, specialist="search", instruction="x", status="done", result="ok"),
            Subtask(id=2, specialist="summarize", instruction="y"),
        ],
        "cursor": 1,
    }
    assert _route_next(state) == "summarize"


def test_router_routes_to_synthesizer_when_done():
    state = {
        "subtasks": [
            Subtask(id=1, specialist="code", instruction="x", status="done", result="ok"),
        ],
        "cursor": 1,
    }
    assert _route_next(state) == "synthesizer"


def test_router_handles_empty_plan():
    assert _route_next({"subtasks": [], "cursor": 0}) == "synthesizer"
