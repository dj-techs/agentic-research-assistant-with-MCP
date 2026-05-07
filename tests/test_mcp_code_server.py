"""Tests for the code MCP server's tool functions.

We import the underlying functions directly rather than going through MCP
stdio so the tests run fast and don't need a process.
"""

from __future__ import annotations

import json

from research_assistant.mcp_servers.code_server import calculate, python_exec


def test_calculate_basic():
    out = json.loads(calculate("2 + 3 * 4"))
    assert out["value"] == 14


def test_calculate_rejects_calls():
    out = json.loads(calculate("__import__('os').system('echo hi')"))
    assert "error" in out


def test_python_exec_runs_print():
    out = json.loads(python_exec("print('hello'); print(2+2)"))
    assert out["exit"] == 0
    assert "hello" in out["stdout"]
    assert "4" in out["stdout"]


def test_python_exec_captures_exit_code():
    out = json.loads(python_exec("import sys; sys.exit(3)"))
    assert out["exit"] == 3
