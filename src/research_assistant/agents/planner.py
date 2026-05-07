"""Planner agent.

Takes the user query and produces a ``Plan``: a sequenced list of
specialist subtasks. We use Claude with structured output (Pydantic schema)
so the plan is guaranteed-parseable.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from ..config import SETTINGS
from ..state import GraphState, Plan, Subtask

PLANNER_SYSTEM = """\
You are the planner of a multi-agent research assistant. You decompose a user's
research query into a sequence of subtasks, each delegated to ONE specialist:

- "search":    web/arxiv search, fetch URLs. Use when the query needs external facts.
- "code":      execute Python or arithmetic. Use for calculations, data crunching, conversions.
- "summarize": condense long text into key points. Use AFTER search to digest sources.

Rules:
- Be minimal. Most queries need 1-3 subtasks. Do not invent work.
- Order matters: search before summarize, code wherever it helps.
- Each instruction must be self-contained: the specialist will not see the original query.
- If the user asks a pure-math question, just emit one "code" subtask.
- If the user asks for a definition only, one "search" subtask is usually enough.
"""


def _planner_model() -> ChatAnthropic:
    return ChatAnthropic(
        model=SETTINGS.planner_model,
        api_key=SETTINGS.anthropic_api_key,
        max_tokens=1024,
        temperature=0,
    )


async def plan_node(state: GraphState) -> dict:
    """LangGraph node: produce a Plan and seed the subtasks list."""
    query = state["query"]
    model = _planner_model().with_structured_output(Plan)
    plan: Plan = await model.ainvoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Query: {query}\n\nProduce a plan."},
        ]
    )

    # Re-number ids so they're contiguous and 1-indexed.
    subtasks = [
        Subtask(
            id=i + 1,
            specialist=s.specialist,
            instruction=s.instruction,
            status="pending",
        )
        for i, s in enumerate(plan.subtasks)
    ]
    return {
        "plan": Plan(rationale=plan.rationale, subtasks=subtasks),
        "subtasks": subtasks,
        "cursor": 0,
        "trace": [{"node": "planner", "rationale": plan.rationale,
                   "subtasks": [s.model_dump() for s in subtasks]}],
    }
