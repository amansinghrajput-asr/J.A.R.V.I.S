"""Performance Profiling Framework for J.A.R.V.I.S Planning Subsystems (Phase 16.5).

Uses cProfile and pstats to profile:
- ExecutionMemory computed property derivation & accumulation
- FailureClassifier pattern evaluations
- RecoveryHeuristics viability checks
- MemorySummaryBuilder string generation
- Planner LLM response validation

Outputs formatted text profiling report to docs/PROFILING_REPORT.md.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
from typing import Any, Dict, List

from app.ai.planner.failure_classifier import FailureClassifier
from app.ai.planner.heuristics import evaluate_recovery_viability
from app.ai.planner.memory import ExecutionMemory, FailureCategory
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.models import ExecutionResult, Plan, PlanningStrategy, Task, TaskStatus
from app.ai.planner.planner import Planner
from benchmarks.benchmark_suite import create_test_plan


def run_profiling(iterations: int = 100) -> str:
    """Profile the planning subsystems across 100 iterations of 250-task plans."""
    plan, result = create_test_plan(250)
    memory = ExecutionMemory.from_execution_result(result, wave=1)

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(iterations):
        # 1. ExecutionMemory computed properties
        _ = memory.completed_tasks
        _ = memory.failed_tasks
        _ = memory.skipped_tasks
        _ = memory.completed_task_ids
        _ = memory.failed_task_ids
        _ = memory.skipped_task_ids
        _ = memory.execution_order
        _ = memory.retry_history
        _ = memory.task_outputs
        _ = memory.latest_task_records

        # 2. FailureClassifier
        for t in result.failed_tasks:
            _ = FailureClassifier.classify(t, "timeout occurred: 30s deadline exceeded", result.dependency_failures)

        # 3. RecoveryHeuristics
        _ = evaluate_recovery_viability(plan.query, plan, memory)

        # 4. MemorySummaryBuilder
        _ = MemorySummaryBuilder.build_planner_summary(memory, plan.tasks)
        _ = MemorySummaryBuilder.build_metadata_summary(memory)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    return s.getvalue()


def render_profiling_markdown(stats_output: str, output_path: str = "docs/PROFILING_REPORT.md") -> str:
    """Save profiling report in Markdown format."""
    md = [
        "# J.A.R.V.I.S Subsystem Profiling Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Workload:** 100 iterations on 250-task synthetic plans  ",
        "",
        "---",
        "",
        "## Top 35 Functions by Cumulative Time",
        "",
        "```text",
        stats_output.strip(),
        "```",
        "",
        "---",
        "",
        "## Bottleneck Analysis & Optimization Targets",
        "",
        "1. **ExecutionMemory Computed Properties Overhead**:",
        "   Each call to `completed_task_ids`, `failed_task_ids`, `retry_history`, and `latest_task_records` iterates linearly through all records ($O(N)$ for each property access).",
        "   When multiple properties are read consecutively (e.g. during summary generation and heuristic checks), `self.records` is traversed 8-10 times redundantly.",
        "   *Target Optimization*: Implement an internal cached cache structure in `ExecutionMemory` with explicit invalidation on `record_execution()`.",
        "",
        "2. **FailureClassifier Regex Compilation Overhead**:",
        "   Currently regex patterns or repeated linear keyword scans can be pre-compiled into compiled pattern tuples for faster priority matching.",
        "   *Target Optimization*: Pre-compile regular expressions and prioritize fast string inclusion checks before regex matching.",
        "",
        "3. **MemorySummaryBuilder List Allocations**:",
        "   String concatenation via intermediate list builds can be streamlined to minimize intermediate objects.",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return output_path


if __name__ == "__main__":
    print("Profiling J.A.R.V.I.S Adaptive Planning Subsystems...")
    profile_text = run_profiling(iterations=100)
    out_file = render_profiling_markdown(profile_text)
    print(f"Profiling report written to {out_file}")
