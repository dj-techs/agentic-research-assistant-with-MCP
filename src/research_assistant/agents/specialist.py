"""Specialist agent factory.

Every specialist follows the same pattern:

1. Receive a ``Subtask`` instruction.
2. Load tools from its dedicated MCP server.
3. Run a tool-using ReAct loop (Claude decides which tools to call).
4. Return the final natural-language result + the tool-call trace.

Differences between specialists are captured purely in their system prompts
and which MCP server they bind to. This keeps the graph simple.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..config import SETTINGS
from ..mcp_client import get_tools
from ..state import GraphState, Specialist, Subtask

SYSTEM_PROMPTS: dict[Specialist, str] = {
    "search": (
        "You are the SEARCH specialist. Use web_search, arxiv_search, and fetch_url "
        "to gather evidence. Prefer arxiv for academic queries. Return a concise "
        "answer that cites the URLs you used."
    ),
    "code": (
        "You are the CODE specialist. Use python_exec for general code and calculate "
        "for pure arithmetic. Show your work briefly and report the final result."
    ),
    "summarize": (
        "You are the SUMMARIZE specialist. Use the summarize and extract_key_points "
        "tools. Choose the style that best fits the instruction."
    ),
}

MAX_STEPS = 6


async def _next_step(
    model: ChatAnthropic, messages: list, tools: list[BaseTool]
) -> AIMessage:
    bound = model.bind_tools(tools)
    return await bound.ainvoke(messages)


async def _run_specialist(specialist: Specialist, instruction: str) -> dict[str, Any]:
    """Execute a single subtask. Returns {result, tool_calls}."""
    tools = await get_tools(specialist)
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    model = ChatAnthropic(
        model=SETTINGS.planner_model,
        api_key=SETTINGS.anthropic_api_key,
        max_tokens=1024,
        temperature=0,
    )

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPTS[specialist]),
        HumanMessage(content=instruction),
    ]
    tool_call_log: list[dict] = []

    for _ in range(MAX_STEPS):
        ai = await _next_step(model, messages, tools)
        messages.append(ai)

        if not ai.tool_calls:
            return {"result": ai.content if isinstance(ai.content, str)
                    else _join_text(ai.content), "tool_calls": tool_call_log}

        for call in ai.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            tool = tools_by_name.get(name)
            if tool is None:
                output = f"Unknown tool: {name}"
            else:
                try:
                    output = await tool.ainvoke(args)
                except Exception as e:
                    output = f"Tool error: {e}"
            tool_call_log.append({"tool": name, "args": args, "output_preview": str(output)[:300]})
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    # Hit step cap: ask the model for a final answer with no tools.
    final = await model.ainvoke(messages + [HumanMessage(
        content="Step limit reached. Give your best final answer now, no more tools."
    )])
    return {
        "result": final.content if isinstance(final.content, str) else _join_text(final.content),
        "tool_calls": tool_call_log,
    }


def _join_text(content: list) -> str:
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def make_specialist_node(specialist: Specialist):
    """Create a LangGraph node bound to a specialist."""

    async def node(state: GraphState) -> dict:
        cursor = state.get("cursor", 0)
        subtasks: list[Subtask] = list(state["subtasks"])
        if cursor >= len(subtasks):
            return {}

        task = subtasks[cursor]
        if task.specialist != specialist:
            # Defensive: router should have prevented this.
            return {}

        out = await _run_specialist(specialist, task.instruction)
        updated = task.model_copy(update={
            "status": "done",
            "result": out["result"],
            "tool_calls": out["tool_calls"],
        })
        subtasks[cursor] = updated

        return {
            "subtasks": subtasks,
            "cursor": cursor + 1,
            "trace": [{
                "node": f"specialist:{specialist}",
                "subtask_id": task.id,
                "instruction": task.instruction,
                "result_preview": out["result"][:300],
                "tool_calls": out["tool_calls"],
            }],
        }

    node.__name__ = f"{specialist}_node"
    return node
