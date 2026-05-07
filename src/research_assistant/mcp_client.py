"""Wraps ``langchain_mcp_adapters.MultiServerMCPClient`` so the rest of the
codebase doesn't have to think about session management.

We launch each MCP server as a child stdio process and load its tools
lazily on first use. Tools are returned as LangChain ``BaseTool`` objects
which Claude (via ``langchain-anthropic``) can call directly through the
model's tool-use API.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import server_command

# Env vars worth propagating to child MCP servers. Anything sensitive (API
# keys) is passed only when actually needed by that server.
_SHARED_ENV = ("PATH", "HOME", "PYTHONPATH", "PYTHONUNBUFFERED", "SEARCH_MODE",
               "CODE_TIMEOUT_SECONDS", "CODE_MEMORY_MB")


def _child_env(extra: tuple[str, ...] = ()) -> dict[str, str]:
    keys = set(_SHARED_ENV) | set(extra)
    return {k: os.environ[k] for k in keys if k in os.environ}


def _server_specs() -> dict[str, dict]:
    """Build the connection spec MultiServerMCPClient expects."""
    specs: dict[str, dict] = {}
    env_for = {
        "search": _child_env(),
        "code": _child_env(),
        "summarize": _child_env(("ANTHROPIC_API_KEY", "SUMMARIZER_MODEL")),
    }
    for name in ("search", "code", "summarize"):
        cmd, args = server_command(name)
        specs[name] = {
            "command": cmd,
            "args": args,
            "transport": "stdio",
            "env": env_for[name],
        }
    return specs


@lru_cache(maxsize=1)
def get_client() -> MultiServerMCPClient:
    """Return a process-wide MCP client. The client lazily spawns servers."""
    return MultiServerMCPClient(_server_specs())


async def get_tools(server: str | None = None) -> list[BaseTool]:
    """Get tools from one server, or all servers if ``server`` is None."""
    client = get_client()
    if server is None:
        return await client.get_tools()
    return await client.get_tools(server_name=server)
