"""CLI entrypoint.

Usage::

    python -m research_assistant "What is retrieval-augmented generation?"
    python -m research_assistant "What is 1234 * 4567?"
    python -m research_assistant --json "Summarize the abstract of the original BERT paper"
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .graph import build_graph

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


async def _run(query: str) -> dict:
    graph = build_graph()
    final = await graph.ainvoke({"query": query})
    return final


@app.command()
def ask(
    query: str = typer.Argument(..., help="Research query"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of pretty output"),
    show_trace: bool = typer.Option(False, "--trace", help="Print the agent trace"),
) -> None:
    """Ask the research assistant a question."""
    final = asyncio.run(_run(query))

    if as_json:
        # Pydantic objects -> dicts
        out = {
            "query": final.get("query"),
            "answer": final.get("answer"),
            "citations": final.get("citations", []),
            "subtasks": [s.model_dump() for s in final.get("subtasks", [])],
            "trace": final.get("trace", []),
        }
        console.print_json(json.dumps(out))
        return

    console.print(Panel.fit(query, title="Query", border_style="blue"))

    plan = final.get("plan")
    if plan:
        console.print(Panel.fit(plan.rationale, title="Plan rationale", border_style="cyan"))
        for st in final.get("subtasks", []):
            console.print(f"  [{st.specialist}] {st.instruction}")

    if show_trace:
        for entry in final.get("trace", []):
            console.print(Panel.fit(json.dumps(entry, indent=2)[:1500],
                                    title=entry.get("node", "trace"),
                                    border_style="yellow"))

    console.print(Panel(Markdown(final.get("answer", "(no answer)")),
                        title="Answer", border_style="green"))
    cites = final.get("citations") or []
    if cites:
        console.print("\n[bold]Citations[/bold]")
        for i, u in enumerate(cites, 1):
            console.print(f"  [{i}] {u}")


def main() -> None:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Allow bare `python -m research_assistant "query"`.
        sys.argv.insert(1, "ask")
    app()


if __name__ == "__main__":
    main()
