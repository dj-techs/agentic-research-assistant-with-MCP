"""Typed state passed through the LangGraph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Specialist = Literal["search", "code", "summarize"]


class Subtask(BaseModel):
    """A single unit of work the planner has decided to delegate."""

    id: int
    specialist: Specialist
    instruction: str = Field(..., description="What the specialist should do, in plain English.")
    status: Literal["pending", "done", "error"] = "pending"
    result: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)


class Plan(BaseModel):
    """Output of the planner. A list of subtasks with rationale."""

    rationale: str
    subtasks: list[Subtask]


def _append(left: list, right: list) -> list:
    return [*left, *right]


class GraphState(TypedDict, total=False):
    """LangGraph state. `total=False` so partial updates from nodes are allowed."""

    query: str
    plan: Plan | None
    subtasks: list[Subtask]
    cursor: int  # index of the next pending subtask
    answer: str
    citations: list[str]
    trace: Annotated[list[dict], _append]
    usage: dict  # cumulative token counts
