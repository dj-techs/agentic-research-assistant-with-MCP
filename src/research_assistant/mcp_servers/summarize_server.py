"""Summarize MCP server.

Two tools:

* ``summarize`` - condense a long text using Claude Haiku.
* ``extract_key_points`` - return N bullet-point key claims (extractive style).

Calling Claude from inside an MCP server is unusual but legitimate: the
specialist's "skill" is *writing summaries*, and the LLM is its tool. The
agent that calls this server doesn't need to know how the summary is made.
"""

from __future__ import annotations

import json
import os
import sys

from anthropic import Anthropic
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("summarize")

_MODEL = os.getenv("SUMMARIZER_MODEL", "claude-haiku-4-5-20251001")
_client: Anthropic | None = None


def _client_get() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _claude(prompt: str, system: str, max_tokens: int = 600) -> str:
    msg = _client_get().messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
    return "".join(parts).strip()


@mcp.tool()
def summarize(text: str, style: str = "brief") -> str:
    """Summarize ``text``. ``style`` is one of: brief, detailed, bullet, executive.

    Returns JSON {summary, style, input_chars}.
    """
    style = (style or "brief").lower()
    if style not in {"brief", "detailed", "bullet", "executive"}:
        style = "brief"

    instructions = {
        "brief": "Write a 2-3 sentence summary capturing the core claim.",
        "detailed": "Write a 1-2 paragraph summary preserving important nuance.",
        "bullet": "Return 4-6 markdown bullets covering the key points.",
        "executive": "Write an executive summary: 1 sentence headline, then 3 bullets.",
    }[style]

    system = "You are a careful summarizer. Be faithful. Do not invent facts."
    prompt = f"{instructions}\n\nTEXT:\n{text[:12000]}"
    try:
        out = _claude(prompt, system)
    except Exception as e:  # surface a useful error to the caller
        return json.dumps({"error": f"summarize failed: {e}"})

    return json.dumps({"summary": out, "style": style, "input_chars": len(text)})


@mcp.tool()
def extract_key_points(text: str, n: int = 5) -> str:
    """Extract ``n`` key claims from text. Returns JSON {points: [..]}."""
    n = max(1, min(int(n), 12))
    system = (
        "You extract key claims from text. Return only the claims, one per line, "
        "no numbering, no preamble."
    )
    prompt = f"Extract exactly {n} key claims.\n\nTEXT:\n{text[:12000]}"
    try:
        out = _claude(prompt, system, max_tokens=400)
    except Exception as e:
        return json.dumps({"error": f"extract_key_points failed: {e}"})

    points = [line.strip("-•* \t") for line in out.splitlines() if line.strip()]
    return json.dumps({"points": points[:n]})


if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)
