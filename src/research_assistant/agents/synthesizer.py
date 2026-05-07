"""Synthesizer agent.

Reads all completed subtask results plus the original query, and writes the
final answer with inline citations to the URLs surfaced by the search
specialist.
"""

from __future__ import annotations

import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import SETTINGS
from ..state import GraphState

SYNTH_SYSTEM = """\
You are the synthesizer. Combine the specialist results into ONE direct answer
to the user's query. Be specific, calibrated, and concise. If the evidence is
inconclusive say so. Cite URLs inline as [1], [2], etc., matching the order in
which they first appear. Do not invent sources.
"""

URL_RE = re.compile(r"https?://[^\s)\]]+")


def _extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for m in URL_RE.findall(text):
        if m not in seen:
            seen.append(m)
    return seen


async def synthesizer_node(state: GraphState) -> dict:
    query = state["query"]
    subtasks = state.get("subtasks", [])

    evidence_blocks = []
    all_urls: list[str] = []
    for st in subtasks:
        if st.result:
            evidence_blocks.append(
                f"## Subtask {st.id} ({st.specialist})\n"
                f"Instruction: {st.instruction}\n"
                f"Result: {st.result}"
            )
            for u in _extract_urls(st.result):
                if u not in all_urls:
                    all_urls.append(u)

    user_msg = (
        f"User query: {query}\n\n"
        f"Specialist evidence:\n\n" + "\n\n".join(evidence_blocks)
    )

    model = ChatAnthropic(
        model=SETTINGS.synthesizer_model,
        api_key=SETTINGS.anthropic_api_key,
        max_tokens=1024,
        temperature=0,
    )
    msg = await model.ainvoke([
        SystemMessage(content=SYNTH_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    answer = msg.content if isinstance(msg.content, str) else "".join(
        b.get("text", "") for b in msg.content if isinstance(b, dict)
    )

    return {
        "answer": answer,
        "citations": all_urls,
        "trace": [{"node": "synthesizer", "answer_preview": answer[:300],
                   "n_citations": len(all_urls)}],
    }
