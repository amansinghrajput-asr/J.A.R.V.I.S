"""Memory Summary Builder for J.A.R.V.I.S Planner.

Formats concise summaries of ExecutionMemory for injection into planner prompts,
response metadata, and human-readable logging without causing token explosion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Final, List, Optional

from app.ai.planner.memory import ExecutionMemory, FailureCategory, sanitize_sensitive_data
from app.ai.planner.models import Task, TaskStatus

VISUAL_ACTIONS: Final[frozenset[str]] = frozenset({
    "visual_click",
    "visual_double_click",
    "visual_type",
    "visual_clear_and_type",
    "visual_select",
    "visual_toggle",
    "visual_dismiss_modal",
    "visual_interact",
})

_VISUAL_STATUS_TOKENS: Final[tuple[str, ...]] = (
    "PREFLIGHT_STALE_COORDINATES",
    "PREFLIGHT_WINDOW_MISMATCH",
    "PREFLIGHT_MODAL_CHANGED",
    "DISABLED_CONTROL",
    "SECURITY_BLOCKED",
    "SENSITIVE_PROTECTED",
    "CONFIRMATION_REQUIRED",
    "VERIFICATION_UNCERTAIN",
    "VERIFICATION_FAILED",
    "GROUNDING_FAILED",
    "TARGET_NOT_FOUND",
    "PRECONDITION_FAILED",
    "TIMEOUT",
)


def _sanitize_text(text: Optional[str]) -> str:
    """Redact sensitive credentials, passwords, raw input text, confirmation tokens, and coordinates."""
    if not text:
        return ""
    return sanitize_sensitive_data(str(text), redact_coordinates=True)


def _extract_visual_status(error_msg: str) -> str:
    """Extract structured visual failure status token if present in error message."""
    err_upper = error_msg.upper()
    for token in _VISUAL_STATUS_TOKENS:
        if token in err_upper:
            return token
    return "UNKNOWN"


class MemorySummaryBuilder:
    """Formats structured summaries of execution memory."""

    @classmethod
    def build_planner_summary(
        cls,
        memory: ExecutionMemory,
        original_plan: Optional[List[Task]] = None,
        max_output_length: int = 140,
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
                    out_str = _sanitize_text(str(out).strip().replace("\n", " "))
                    if len(out_str) > max_output_length:
                        out_str = out_str[:max_output_length] + "..."
                    out_snippet = f" -> Output: \"{out_str}\""
                target_disp = _sanitize_text(t.target)
                lines.append(f"- ID '{t.id}': {t.action} (target: '{target_disp}'){out_snippet}")
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
                raw_err = rec.error if rec and rec.error else "Unknown error"

                # Check if visual action task
                is_visual = (
                    t.action in VISUAL_ACTIONS
                    or (rec is not None and rec.action in VISUAL_ACTIONS)
                    or cat in (
                        FailureCategory.VISUAL_PRECONDITION_FAILURE.value,
                        FailureCategory.VISUAL_TOCTOU_FAILURE.value,
                        FailureCategory.VISUAL_VERIFICATION_FAILURE.value,
                    )
                )

                if is_visual:
                    status_token = getattr(rec, "visual_status", None) or _extract_visual_status(raw_err)
                    clean_err = _sanitize_text(raw_err)
                    if len(clean_err) > max_output_length:
                        clean_err = clean_err[:max_output_length] + "..."
                    has_input = bool(
                        (t.parameters and ("input_text" in t.parameters or "text" in t.parameters))
                        or t.action in ("visual_type", "visual_clear_and_type")
                    )
                    input_meta = " (input_present=True, input_redacted=True)" if has_input else ""
                    target_disp = _sanitize_text(t.target)
                    lines.append(
                        f"- ID '{t.id}': {t.action} (target: '{target_disp}'{input_meta}) "
                        f"[Visual Cause: {cat}, Status: {status_token}, Attempts: {attempts}] Reason: {clean_err}"
                    )
                else:
                    err = _sanitize_text(raw_err)
                    if len(err) > max_output_length:
                        err = err[:max_output_length] + "..."
                    target_disp = _sanitize_text(t.target)
                    lines.append(
                        f"- ID '{t.id}': {t.action} (target: '{target_disp}') [Cause: {cat}, Attempts: {attempts}] Error: {err}"
                    )
            sections.append("\n".join(lines))

        # 3. Skipped Tasks
        skipped = memory.skipped_tasks
        if skipped:
            lines = ["Skipped Tasks (Blocked by failed dependencies):"]
            dependency_failures = memory.dependency_failures
            for t in skipped:
                deps = dependency_failures.get(t.id, [])
                target_disp = _sanitize_text(t.target)
                lines.append(f"- ID '{t.id}': {t.action} (target: '{target_disp}') [Blocked by: {deps}]")
            sections.append("\n".join(lines))

        # 4. Remaining original tasks
        if original_plan:
            completed_ids = memory.completed_task_ids
            uncompleted_orig = [t for t in original_plan if t.id not in completed_ids]
            if uncompleted_orig:
                lines = ["Original Tasks Not Yet Completed:"]
                for t in uncompleted_orig:
                    target_disp = _sanitize_text(t.target)
                    lines.append(f"- ID '{t.id}': {t.action} (target: '{target_disp}')")
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
                is_visual = (
                    r.action in VISUAL_ACTIONS
                    or r.failure_category in (
                        FailureCategory.VISUAL_PRECONDITION_FAILURE,
                        FailureCategory.VISUAL_TOCTOU_FAILURE,
                        FailureCategory.VISUAL_VERIFICATION_FAILURE,
                    )
                )
                safe_err = _sanitize_text(r.error)
                failure_info: Dict[str, Any] = {
                    "action": r.action,
                    "target": _sanitize_text(r.target),
                    "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                    "failure_category": r.failure_category.value if hasattr(r.failure_category, "value") else str(r.failure_category),
                    "error": safe_err,
                    "attempts": retry_history.get(tid, 1),
                }
                if is_visual:
                    failure_info["is_visual"] = True
                    failure_info["visual_status"] = getattr(r, "visual_status", None) or _extract_visual_status(r.error or "")
                    failure_info["input_redacted"] = True
                    if getattr(r, "verification_confidence", None) is not None:
                        failure_info["verification_confidence"] = r.verification_confidence
                    if getattr(r, "evidence_summary", None):
                        failure_info["evidence_summary"] = _sanitize_text(r.evidence_summary)
                    if getattr(r, "preflight_status", None):
                        failure_info["preflight_status"] = r.preflight_status
                latest_failures[tid] = failure_info

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
