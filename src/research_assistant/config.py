"""Centralised configuration sourced from environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
SERVERS_DIR = Path(__file__).resolve().parent / "mcp_servers"
FIXTURES_DIR = ROOT / "evals" / "fixtures"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    planner_model: str
    synthesizer_model: str
    summarizer_model: str
    judge_model: str
    search_mode: str
    code_timeout: int
    code_memory_mb: int

    @classmethod
    def load(cls) -> "Settings":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        return cls(
            anthropic_api_key=key,
            planner_model=os.getenv("PLANNER_MODEL", "claude-sonnet-4-6"),
            synthesizer_model=os.getenv("SYNTHESIZER_MODEL", "claude-sonnet-4-6"),
            summarizer_model=os.getenv("SUMMARIZER_MODEL", "claude-haiku-4-5-20251001"),
            judge_model=os.getenv("JUDGE_MODEL", "claude-sonnet-4-6"),
            search_mode=os.getenv("SEARCH_MODE", "real"),
            code_timeout=int(os.getenv("CODE_TIMEOUT_SECONDS", "10")),
            code_memory_mb=int(os.getenv("CODE_MEMORY_MB", "256")),
        )


SETTINGS = Settings.load()


def server_command(name: str) -> tuple[str, list[str]]:
    """Return (command, args) for launching one of our stdio MCP servers.

    Uses ``sys.executable`` so the child runs in the same interpreter / venv
    as the parent — no PATH surprises.
    """
    script = SERVERS_DIR / f"{name}_server.py"
    return (sys.executable, [str(script)])
