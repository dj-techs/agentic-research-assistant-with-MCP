"""Eval harness.

Runs every case in ``dataset.json`` through the graph and reports:

* Per-case: judge score, pass/fail, keyword recall, specialist routing accuracy,
  latency, tool-call count, agent trace.
* Aggregate: pass rate, mean score, mean recall, mean routing accuracy,
  total wall-clock time.

Outputs both a machine-readable JSON file and a human-readable Markdown
report into ``evals/results/``.

Run::

    python -m evals.harness                  # uses real search
    SEARCH_MODE=fixture python -m evals.harness   # deterministic
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from research_assistant.graph import build_graph

from .judge import judge
from .metrics import first_specialist_correct, keyword_recall, specialist_accuracy

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


async def _run_case(graph, case: dict) -> dict:
    t0 = time.time()
    try:
        final = await graph.ainvoke({"query": case["query"]})
        elapsed = time.time() - t0
        subtasks = final.get("subtasks", [])
        used = [s.specialist for s in subtasks]
        tool_call_count = sum(len(s.tool_calls or []) for s in subtasks)
        answer = final.get("answer", "")

        verdict = judge(case["query"], answer, case["rubric"])

        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "answer": answer,
            "elapsed_s": round(elapsed, 2),
            "specialists_used": used,
            "specialists_expected": case["expected_specialists"],
            "tool_call_count": tool_call_count,
            "metrics": {
                "judge_score": verdict.score,
                "judge_pass": verdict.pass_,
                "judge_rationale": verdict.rationale,
                "keyword_recall": round(keyword_recall(answer, case["expected_keywords"]), 3),
                "specialist_jaccard": round(
                    specialist_accuracy(used, case["expected_specialists"]), 3
                ),
                "first_specialist_correct": first_specialist_correct(
                    used, case["expected_specialists"]
                ),
            },
            "subtasks": [s.model_dump() for s in subtasks],
            "citations": final.get("citations", []),
            "error": None,
        }
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "elapsed_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        }


def _aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return {"n": len(results), "n_errors": len(results), "pass_rate": 0.0}

    scores = [r["metrics"]["judge_score"] for r in ok]
    passes = [int(r["metrics"]["judge_pass"]) for r in ok]
    recalls = [r["metrics"]["keyword_recall"] for r in ok]
    routings = [r["metrics"]["specialist_jaccard"] for r in ok]
    firsts = [r["metrics"]["first_specialist_correct"] for r in ok]
    latencies = [r["elapsed_s"] for r in ok]

    return {
        "n": len(results),
        "n_errors": len(results) - len(ok),
        "pass_rate": round(sum(passes) / len(passes), 3),
        "mean_judge_score": round(statistics.mean(scores), 2),
        "mean_keyword_recall": round(statistics.mean(recalls), 3),
        "mean_specialist_jaccard": round(statistics.mean(routings), 3),
        "first_specialist_accuracy": round(statistics.mean(firsts), 3),
        "p50_latency_s": round(statistics.median(latencies), 2),
        "p95_latency_s": round(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 5
            else max(latencies), 2,
        ),
    }


def _write_markdown(results: list[dict], agg: dict, path: Path) -> None:
    lines = ["# Research Assistant — Eval Report", ""]
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    for k, v in agg.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| ID | Category | Pass | Score | Recall | Routing | Latency | Specialists |")
    lines.append("|----|----------|------|-------|--------|---------|---------|-------------|")
    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['id']} | {r['category']} | ❌ ERROR | - | - | - | "
                f"{r['elapsed_s']}s | {r['error']} |"
            )
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['id']} | {r['category']} | "
            f"{'✅' if m['judge_pass'] else '❌'} | "
            f"{m['judge_score']}/5 | {m['keyword_recall']} | {m['specialist_jaccard']} | "
            f"{r['elapsed_s']}s | {','.join(r['specialists_used'])} |"
        )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    for r in results:
        if r.get("error") or not r["metrics"]["judge_pass"]:
            lines.append(f"### {r.get('id')} — {r.get('query', '')}")
            if r.get("error"):
                lines.append(f"`{r['error']}`")
            else:
                lines.append(f"**Answer:** {r.get('answer', '')[:500]}")
                lines.append(f"**Judge:** {r['metrics']['judge_rationale']}")
            lines.append("")

    path.write_text("\n".join(lines))


async def main() -> None:
    cases = json.loads(DATASET.read_text())["cases"]
    graph = build_graph()
    results: list[dict] = []
    for c in cases:
        print(f"▶ {c['id']}: {c['query'][:80]}")
        r = await _run_case(graph, c)
        if r.get("error"):
            print(f"  ✗ {r['error']}")
        else:
            m = r["metrics"]
            print(
                f"  {'✓' if m['judge_pass'] else '✗'} score={m['judge_score']}/5 "
                f"recall={m['keyword_recall']} routing={m['specialist_jaccard']} "
                f"({r['elapsed_s']}s)"
            )
        results.append(r)

    agg = _aggregate(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    md_path = RESULTS_DIR / f"eval_{stamp}.md"
    json_path.write_text(json.dumps({"aggregate": agg, "cases": results}, indent=2))
    _write_markdown(results, agg, md_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"\nAggregate: {agg}")


if __name__ == "__main__":
    asyncio.run(main())
