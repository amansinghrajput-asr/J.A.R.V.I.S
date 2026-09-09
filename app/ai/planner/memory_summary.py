"""Memory Summary Builder for J.A.R.V.I.S Planner.

Formats concise summaries of ExecutionMemory for injection into planner prompts,
response metadata, and human-readable logging without causing token explosion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.ai.planner.memory import ExecutionMemory
from app.ai.planner.models import Task, TaskStatus


class MemorySummaryBuilder:
    """Formats structured summaries of execution memory."""

    @classmethod
    def build_planner_summary(
        cls,
        memory: ExecutionMemory,
        original_plan: Optional[List[Task]] = None,
        max_output_length: int = 120,
    ) -> str:
        """Construct a compact summary suitable for LLM prompt context."""
        sections: List[str] = []

        # 1. Successfully Completed Work
        completed = memory.completed_tasks
        if completed:
            lines = ["Completed Tasks (DO NOT REPEAT):"]
            task_outputs = memory.task_outputs
            for t in completed:
                out = task_outputs.get(t.id)
                out_snippet = ""
                if out is not None:
                    out_str = str(out).strip().replace("\n", " ")
                    if len(out_str) > max_output_length:
                        out_str = out_str[:max_output_length] + "..."
                    out_snippet = f" -> Output: \"{out_str}\""
                lines.append(f"- ID '{t.id}': {t.action} (target: '{t.target}'){out_snippet}")
            sections.append("\n".join(lines))
        else:
            sections.append("Completed Tasks: None")

        # 2. Failed Tasks with Root Cause
        latest_records = memory.latest_task_records
        failed = memory.failed_tasks
        if failed:
            lines = ["Failed Tasks (Require alternative or recovery):"]
            retry_history = memory.retry_history
            for t in failed:
                rec = latest_records.get(t.id)
                cat = rec.failure_category.value if rec else "unknown"
                attempts = retry_history.get(t.id, 1)
                err = rec.error if rec and rec.error else "Unknown error"
                if len(err) > max_output_length:
                    err = err[:max_output_length] + "..."
                lines.append(
                    f"- ID '{t.id}': {t.action} (target: '{t.target}') [Cause: {cat}, Attempts: {attempts}] Error: {err}"
                )
            sections.append("\n".join(lines))

        # 3. Skipped Tasks
        skipped = memory.skipped_tasks
        if skipped:
            lines = ["Skipped Tasks (Blocked by failed dependencies):"]
            dependency_failures = memory.dependency_failures
            for t in skipped:
                deps = dependency_failures.get(t.id, [])
                lines.append(f"- ID '{t.id}': {t.action} (target: '{t.target}') [Blocked by: {deps}]")
            sections.append("\n".join(lines))

        # 4. Remaining original tasks
        if original_plan:
            completed_ids = memory.completed_task_ids
            uncompleted_orig = [t for t in original_plan if t.id not in completed_ids]
            if uncompleted_orig:
                lines = ["Original Tasks Not Yet Completed:"]
                for t in uncompleted_orig:
                    lines.append(f"- ID '{t.id}': {t.action} (target: '{t.target}')")
                sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @classmethod
    def build_metadata_summary(cls, memory: ExecutionMemory) -> Dict[str, Any]:
        """Construct structured metadata summary of execution memory for API responses."""
        latest_records = memory.latest_task_records
        retry_history = memory.retry_history
        latest_failures: Dict[str, Dict[str, Any]] = {}
        for tid, r in latest_records.items():
            if r.status in (TaskStatus.FAILED, TaskStatus.SKIPPED):
                latest_failures[tid] = {
                    "action": r.action,
                    "target": r.target,
                    "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                    "failure_category": r.failure_category.value if hasattr(r.failure_category, "value") else str(r.failure_category),
                    "error": r.error,
                    "attempts": retry_history.get(tid, 1),
                }

        return {
            "total_records": len(memory.records),
            "execution_waves": memory.metrics.execution_waves,
            "completed_count": len(memory.completed_tasks),
            "failed_count": len(memory.failed_tasks),
            "skipped_count": len(memory.skipped_tasks),
            "retry_counts": dict(retry_history),
            "latest_failures": latest_failures,
        }

    @classmethod
    def build_human_readable_summary(cls, memory: ExecutionMemory) -> str:
        """Construct human-readable multi-line summary of overall execution history."""
        lines: List[str] = [
            f"Execution History: {len(memory.completed_tasks)} completed, {len(memory.failed_tasks)} failed, {len(memory.skipped_tasks)} skipped across {memory.metrics.execution_waves} wave(s)."
        ]
        for r in memory.records:
            status_symbol = "✓" if r.status == TaskStatus.COMPLETED else ("✗" if r.status == TaskStatus.FAILED else "○")
            err_str = f" ({r.error})" if r.error else ""
            lines.append(f"  [{status_symbol}] Wave {r.wave} - Task {r.task_id} ({r.action} -> {r.target}){err_str}")
        return "\n".join(lines)
