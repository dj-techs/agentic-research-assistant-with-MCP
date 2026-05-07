"""Code MCP server.

Provides a sandboxed Python execution tool and a safe arithmetic tool.

Sandboxing strategy:

* Run user code in a fresh ``python`` subprocess (no shared state).
* Strip the environment of secrets before exec.
* Apply a wall-clock timeout via ``subprocess.run(timeout=...)``.
* Apply a memory limit on POSIX via ``resource.setrlimit`` in a preexec_fn.
* Capture stdout/stderr; return both, plus exit status.

This is good enough for evaluation and demonstration; treat it as untrusted
and never point it at a production network/filesystem.
"""

from __future__ import annotations

import ast
import json
import operator as op
import os
import resource
import subprocess
import sys
import tempfile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code")

CODE_TIMEOUT = int(os.getenv("CODE_TIMEOUT_SECONDS", "10"))
CODE_MEMORY_MB = int(os.getenv("CODE_MEMORY_MB", "256"))


def _limit_resources() -> None:  # pragma: no cover - runs in child process
    bytes_limit = CODE_MEMORY_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CODE_TIMEOUT, CODE_TIMEOUT))
    except (ValueError, OSError):
        pass


@mcp.tool()
def python_exec(code: str) -> str:
    """Execute Python in a sandboxed subprocess. Returns JSON {stdout, stderr, exit}."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            env=env,
            preexec_fn=_limit_resources if sys.platform != "win32" else None,
        )
        return json.dumps(
            {
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-2000:],
                "exit": proc.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"stdout": "", "stderr": f"timeout after {CODE_TIMEOUT}s", "exit": 124}
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a pure arithmetic expression safely. Returns JSON {value} or {error}."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_ast(tree)
        return json.dumps({"value": value})
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)
