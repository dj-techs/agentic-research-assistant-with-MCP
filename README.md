# Agentic Research Assistant with MCP

A multi-agent research assistant where a **planner** decomposes a query and delegates
to specialist sub-agents (**search**, **code**, **summarize**), each implemented as a
real **MCP server** spawned over stdio. Orchestration is **LangGraph**, the reasoning
model is **Claude Sonnet 4.6**, and there is a real **eval harness** that measures
task-completion accuracy, specialist routing, keyword recall, latency, and tool-call
efficiency.

## Architecture

```
                     ┌────────────────────────┐
       query ──────▶ │  Planner (Claude)      │ structured output → Plan
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  Router (state-based)  │
                     └────┬─────────┬────────┬┘
            search │     │ code    │  summarize │
                   ▼     ▼         ▼            ▼
   ┌──────────────────────────────────────────────────┐
   │   Specialist sub-agents (Claude tool-use)        │
   │   each connects via stdio MCP to its server      │
   ├──────────────────────────────────────────────────┤
   │  search MCP    code MCP        summarize MCP     │
   │  • web_search  • python_exec   • summarize       │
   │  • arxiv_search• calculate     • extract_key_pts │
   │  • fetch_url                                     │
   └──────────────────────────────────────────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Synthesizer (Claude)   │ → final answer + citations
                     └────────────────────────┘
```

* **Planner** — `agents/planner.py`. Uses Claude with Pydantic structured output
  (`Plan`) so the plan is always valid.
* **Specialists** — `agents/specialist.py`. A single factory builds a tool-using
  ReAct loop bound to one MCP server per specialist. The system prompt is the only
  thing that differs.
* **Synthesizer** — `agents/synthesizer.py`. Reads all subtask results, writes the
  final answer, extracts citation URLs.
* **Graph** — `graph.py`. A `StateGraph` over `GraphState` with conditional edges
  driven by a cursor over the subtask list.

## Quick start

```bash
make dev                                  # install editable + dev deps
cp .env.example .env && $EDITOR .env      # set ANTHROPIC_API_KEY
make run Q="What is the Model Context Protocol?"
```

Useful flags:

```bash
python -m research_assistant "..." --trace      # show full agent trace
python -m research_assistant "..." --json       # machine-readable output
```

## Running the eval harness

The harness runs every case in `evals/dataset.json` through the graph and writes
both a JSON and a Markdown report into `evals/results/`.

```bash
make eval              # real DuckDuckGo + arXiv
make eval-fixture      # deterministic, no network
```

Metrics reported per-case:

| Metric                     | What it measures                                     |
| -------------------------- | ---------------------------------------------------- |
| `judge_score` (1-5)        | LLM-as-judge against the per-case rubric             |
| `judge_pass` (bool)        | Hard pass/fail decision from the judge               |
| `keyword_recall`           | Fraction of expected keywords present in the answer  |
| `specialist_jaccard`       | Set overlap between specialists used vs expected     |
| `first_specialist_correct` | Did the planner route the *first* subtask correctly? |
| `tool_call_count`          | How many MCP tool calls the run cost                 |
| `elapsed_s`                | Wall-clock latency                                   |

Aggregates: pass rate, mean score, mean recall, mean routing accuracy, p50/p95
latency, error count.

## MCP servers

Each lives under `src/research_assistant/mcp_servers/` and is a self-contained
`FastMCP` script. They are launched as child processes by
`langchain-mcp-adapters` over stdio.

You can also run any of them standalone for inspection:

```bash
python -m research_assistant.mcp_servers.search_server      # speaks MCP on stdio
python -m research_assistant.mcp_servers.code_server
python -m research_assistant.mcp_servers.summarize_server
```

### Sandbox notes

`code_server.py` runs user code in an isolated `python -I` subprocess with a
wall-clock timeout (`CODE_TIMEOUT_SECONDS`) and POSIX `RLIMIT_AS` /
`RLIMIT_CPU` caps. This is good enough for evals and demos; do not point it at
production data. The server scrubs the environment of all variables except
`PATH` before exec.

## Tests

```bash
pytest -v
```

Covers MCP server functions (search & code), graph routing logic, and metrics.
The graph router and specialist factory are testable without making any Claude
calls. End-to-end tests run through the eval harness.

## Project layout

```
src/research_assistant/
├── __main__.py            # python -m research_assistant
├── main.py                # Typer CLI
├── config.py              # env-driven settings
├── state.py               # Pydantic state + GraphState TypedDict
├── mcp_client.py          # MultiServerMCPClient wrapper
├── graph.py               # LangGraph StateGraph
├── agents/
│   ├── planner.py
│   ├── specialist.py      # factory: search/code/summarize nodes
│   └── synthesizer.py
└── mcp_servers/
    ├── search_server.py   # web_search, arxiv_search, fetch_url
    ├── code_server.py     # python_exec, calculate
    └── summarize_server.py# summarize, extract_key_points

evals/
├── dataset.json           # eval cases with rubrics
├── fixtures/search.json   # deterministic search responses
├── metrics.py             # keyword_recall, specialist_accuracy, etc.
├── judge.py               # LLM-as-judge with Pydantic verdict
└── harness.py             # runs dataset, writes JSON + Markdown reports
```

## Why this design

* **Specialists as MCP servers, not just functions.** Each specialist's tools
  live behind a real protocol boundary, the same way an enterprise system would
  expose them. Swapping `search_server.py` for a Brave/Tavily-backed server, or
  adding a new specialist, requires no graph changes.
* **Planner with structured output.** No prompt-parsing fragility — the plan is
  validated by Pydantic before the router ever sees it.
* **Cursor-based router** instead of a meta-agent loop. Simple, debuggable, and
  the whole control flow is visible in `graph.py`.
* **Two judges.** Programmatic metrics (recall, routing) catch regressions
  cheaply; LLM-as-judge handles open-ended correctness.
* **Fixture mode for evals.** Real search is great for demos; deterministic
  fixtures are what you actually want in CI.
