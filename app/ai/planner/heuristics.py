"""Recovery Heuristics for J.A.R.V.I.S Planner.

Provides lightweight deterministic checks to evaluate whether replanning is viable
before invoking the LLM provider, avoiding unnecessary or futile recovery attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Set, Union

from app.ai.planner.memory import ExecutionMemory, FailureCategory
from app.ai.planner.models import Plan, Task


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

PERMANENT_VISUAL_FAILURES: Final[frozenset[str]] = frozenset({
    "SECURITY_BLOCKED",
    "SENSITIVE_PROTECTED",
    "DISABLED_CONTROL",
    "CONFIRMATION_REQUIRED",
    "VERIFICATION_UNCERTAIN",
})


def is_permanent_visual_failure(error_msg: Optional[str]) -> Optional[str]:
    """Evaluate whether an error indicates a permanent non-recoverable visual failure."""
    if not error_msg:
        return None
    err_upper = str(error_msg).upper()
    for reason in PERMANENT_VISUAL_FAILURES:
        if reason in err_upper:
            return reason
    err_lower = str(error_msg).lower()
    if "disabled control" in err_lower:
        return "DISABLED_CONTROL"
    if "confirmation required" in err_lower or "requires explicit confirmation" in err_lower:
        return "CONFIRMATION_REQUIRED"
    if "sensitive protected" in err_lower or "strictly prohibited" in err_lower:
        return "SENSITIVE_PROTECTED"
    if "security policy" in err_lower or "security blocked" in err_lower:
        return "SECURITY_BLOCKED"
    if "verification uncertain" in err_lower:
        return "VERIFICATION_UNCERTAIN"
    return None


@dataclass(frozen=True)
class RecoveryDecision:
    """Structured evaluation outcome of recovery viability heuristics.

    Attributes:
        viable: Whether replanning should proceed.
        reason: Human-readable explanation of the heuristic decision.
        blocking_tasks: List of task IDs causing recovery to be non-viable, if any.
    """

    viable: bool
    reason: str
    blocking_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to dictionary."""
        return {
            "viable": self.viable,
            "reason": self.reason,
            "blocking_tasks": list(self.blocking_tasks),
        }


def evaluate_recovery_viability(
    original_query: str,
    original_plan: Union[Plan, List[Task]],
    memory: ExecutionMemory,
    max_task_retries: int = 3,
) -> RecoveryDecision:
    """Deterministically evaluate whether replanning is worthwhile without invoking LLM.

    Evaluations performed:
    1. Completion check: If all tasks are completed, recovery is unnecessary.
    2. Empty plan check: If no original tasks exist, recovery is impossible.
    3. Permanent visual failures: If any visual task failed due to permanent reasons
       (SECURITY_BLOCKED, SENSITIVE_PROTECTED, DISABLED_CONTROL, CONFIRMATION_REQUIRED,
       VERIFICATION_UNCERTAIN), halt visual recovery immediately.
    4. Max retries check: If any remaining failed task has exceeded its retry budget
       (max 1 recovery attempt for visual tasks; max_task_retries for ordinary tasks),
       halt to prevent infinite retry loops.
    5. Permanent validation errors: If all remaining tasks failed due to permanent
       VALIDATION_FAILURE (e.g. unknown action), replanning cannot resolve them.
    6. Impossible dependency chains: If remaining work is entirely skipped due to dependencies
       that have permanently failed and cannot be recovered.

    Args:
        original_query: The initial user query.
        original_plan: The plan initially scheduled (Plan object or List[Task]).
        memory: The current accumulated execution memory.
        max_task_retries: Maximum permitted failure attempts for a single task.

    Returns:
        RecoveryDecision indicating viability, rationale, and any blocking task IDs.
    """
    if isinstance(original_plan, Plan):
        original_tasks = list(original_plan.tasks)
    else:
        original_tasks = list(original_plan or [])

    # 1. Completion check: if original plan is provided and all tasks are completed
    if original_tasks and all(t.id in memory.completed_task_ids for t in original_tasks):
        return RecoveryDecision(
            viable=False,
            reason="All tasks completed successfully; no remaining work.",
        )

    latest_records = memory.latest_task_records

    # 2. Check for permanent non-recoverable visual failures (immediate halt)
    for task in memory.failed_tasks:
        record = latest_records.get(task.id)
        is_visual = (
            task.action in VISUAL_ACTIONS
            or (record is not None and record.action in VISUAL_ACTIONS)
            or (record is not None and record.failure_category in (
                FailureCategory.VISUAL_PRECONDITION_FAILURE,
                FailureCategory.VISUAL_TOCTOU_FAILURE,
                FailureCategory.VISUAL_VERIFICATION_FAILURE,
            ))
        )
        if is_visual and record and record.error:
            perm_reason = is_permanent_visual_failure(record.error)
            if perm_reason:
                return RecoveryDecision(
                    viable=False,
                    reason=f"Visual task '{task.id}' failed with permanent non-recoverable status: {perm_reason}.",
                    blocking_tasks=[task.id],
                )

    # 3. Repeated fatal failures on same task (retry exhaustion)
    retries = memory.retry_history
    exhausted_tasks: List[str] = []
    for task in memory.failed_tasks:
        record = latest_records.get(task.id)
        is_visual = (
            task.action in VISUAL_ACTIONS
            or (record is not None and record.action in VISUAL_ACTIONS)
            or (record is not None and record.failure_category in (
                FailureCategory.VISUAL_PRECONDITION_FAILURE,
                FailureCategory.VISUAL_TOCTOU_FAILURE,
                FailureCategory.VISUAL_VERIFICATION_FAILURE,
            ))
        )
        # Visual actions allow at most 1 recovery attempt (total attempts limit = 2)
        limit = 2 if is_visual else max_task_retries
        if retries.get(task.id, 0) >= limit:
            exhausted_tasks.append(task.id)

    if exhausted_tasks:
        return RecoveryDecision(
            viable=False,
            reason=f"Tasks {exhausted_tasks} exceeded maximum retry limit without progress.",
            blocking_tasks=exhausted_tasks,
        )

    # 4. Check for permanent validation failures on all remaining tasks
    remaining_tasks = list(memory.failed_tasks) + list(memory.skipped_tasks)
    permanent_failures: List[str] = []
    for t in remaining_tasks:
        record = latest_records.get(t.id)
        if record and record.failure_category == FailureCategory.VALIDATION_FAILURE:
            permanent_failures.append(t.id)

    if permanent_failures and len(permanent_failures) == len(remaining_tasks):
        return RecoveryDecision(
            viable=False,
            reason=f"All remaining tasks {permanent_failures} failed due to permanent validation errors.",
            blocking_tasks=permanent_failures,
        )

    # 5. Impossible dependency chains: all remaining work is skipped and all failed dependencies are exhausted
    if not memory.failed_tasks and memory.skipped_tasks:
        # Only skipped tasks remain, check if their dependencies were ever completed
        completed_ids = memory.completed_task_ids
        unresolvable: List[str] = []
        for t in memory.skipped_tasks:
            failed_deps = memory.dependency_failures.get(t.id, [])
            if any(dep not in completed_ids for dep in failed_deps):
                unresolvable.append(t.id)

        if len(unresolvable) == len(memory.skipped_tasks):
            return RecoveryDecision(
                viable=False,
                reason=f"All remaining tasks {unresolvable} are blocked by permanently unresolvable dependencies.",
                blocking_tasks=unresolvable,
            )

    return RecoveryDecision(
        viable=True,
        reason="Recovery is viable; uncompleted tasks remain within retry budget.",
    )
