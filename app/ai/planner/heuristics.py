"""Recovery Heuristics for J.A.R.V.I.S Planner.

Provides lightweight deterministic checks to evaluate whether replanning is viable
before invoking the LLM provider, avoiding unnecessary or futile recovery attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from app.ai.planner.memory import ExecutionMemory, FailureCategory
from app.ai.planner.models import Plan, Task


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
    3. Max retries check: If any remaining failed task has failed >= max_task_retries,
       halt to prevent infinite retry loops on fundamentally broken tasks.
    4. Permanent validation errors: If all remaining tasks failed due to permanent
       VALIDATION_FAILURE (e.g. unknown action), replanning cannot resolve them.
    5. Impossible dependency chains: If remaining work is entirely skipped due to dependencies
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

    # 3. Repeated fatal failures on same task (retry exhaustion)
    retries = memory.retry_history
    exhausted_tasks: List[str] = []
    for task in memory.failed_tasks:
        if retries.get(task.id, 0) >= max_task_retries:
            exhausted_tasks.append(task.id)

    if exhausted_tasks:
        return RecoveryDecision(
            viable=False,
            reason=f"Tasks {exhausted_tasks} exceeded maximum retry limit ({max_task_retries}) without progress.",
            blocking_tasks=exhausted_tasks,
        )

    # 4. Check for permanent validation failures on all remaining tasks
    remaining_tasks = list(memory.failed_tasks) + list(memory.skipped_tasks)
    latest_records = memory.latest_task_records
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
