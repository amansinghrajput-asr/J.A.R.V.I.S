"""Execution Memory Subsystem for J.A.R.V.I.S Planner.

Maintains canonical execution history across multi-wave execution and replanning cycles,
providing structured context for adaptive replanning, heuristics, and observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from app.ai.planner.models import ExecutionResult, Task, TaskStatus


class FailureCategory(str, Enum):
    """Categorization of task failure causes."""

    DEPENDENCY_FAILURE = "dependency_failure"
    TOOL_FAILURE = "tool_failure"
    VALIDATION_FAILURE = "validation_failure"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Immutable record of a single task execution attempt.

    Attributes:
        task_id: Identifier of the task executed.
        action: Action type executed.
        target: Target operand of the action.
        status: Final status for this attempt (COMPLETED, FAILED, SKIPPED).
        wave: Execution wave number (1 for initial, 2+ for recovery waves).
        attempt: Total attempt number for this specific task ID.
        output: Human-readable output or result if completed.
        error: Error message if failed or skipped.
        failure_category: Categorized root cause if failed or skipped.
        duration: Execution duration of this specific task attempt in seconds.
        timestamp: Unix timestamp when the record was created.
    """

    task_id: str
    action: str
    target: Optional[str]
    status: TaskStatus
    wave: int = 1
    attempt: int = 1
    output: Optional[Any] = None
    error: Optional[str] = None
    failure_category: FailureCategory = FailureCategory.UNKNOWN
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "task_id": self.task_id,
            "action": self.action,
            "target": self.target,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "wave": self.wave,
            "attempt": self.attempt,
            "output": self.output,
            "error": self.error,
            "failure_category": self.failure_category.value if hasattr(self.failure_category, "value") else str(self.failure_category),
            "duration": self.duration,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskExecutionRecord:
        """Deserialize dictionary to TaskExecutionRecord."""
        status_val = data.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_val)
        except Exception:
            status = TaskStatus.PENDING
        cat_val = data.get("failure_category", FailureCategory.UNKNOWN.value)
        try:
            cat = FailureCategory(cat_val)
        except Exception:
            cat = FailureCategory.UNKNOWN
        return cls(
            task_id=str(data.get("task_id", "")),
            action=str(data.get("action", "")),
            target=data.get("target"),
            status=status,
            wave=int(data.get("wave", 1)),
            attempt=int(data.get("attempt", 1)),
            output=data.get("output"),
            error=data.get("error"),
            failure_category=cat,
            duration=float(data.get("duration", 0.0)),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class ExecutionMetrics:
    """Observational metrics collected across planning and recovery lifecycles.

    Metrics are strictly observational and must never influence planning decisions.
    """

    planning_count: int = 0
    replan_count: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    total_recovery_duration: float = 0.0
    execution_waves: int = 0

    @property
    def average_recovery_duration(self) -> float:
        """Return average recovery duration across executed recovery replans."""
        if self.replan_count <= 0:
            return 0.0
        return self.total_recovery_duration / self.replan_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "planning_count": self.planning_count,
            "replan_count": self.replan_count,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "total_recovery_duration": self.total_recovery_duration,
            "average_recovery_duration": self.average_recovery_duration,
            "execution_waves": self.execution_waves,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionMetrics:
        """Deserialize dictionary to ExecutionMetrics."""
        return cls(
            planning_count=int(data.get("planning_count", 0)),
            replan_count=int(data.get("replan_count", 0)),
            successful_recoveries=int(data.get("successful_recoveries", 0)),
            failed_recoveries=int(data.get("failed_recoveries", 0)),
            total_recovery_duration=float(data.get("total_recovery_duration", 0.0)),
            execution_waves=int(data.get("execution_waves", 0)),
        )


@dataclass
class ExecutionMemory:
    """Canonical accumulated execution history across all execution waves.

    Stores sequential execution records, tracks dependency failures, and records
    execution metrics. State such as completed/failed tasks, execution order, and
    retry history are derived dynamically as computed properties to avoid duplicate sources of truth.
    """

    records: List[TaskExecutionRecord] = field(default_factory=list)
    dependency_failures: Dict[str, List[str]] = field(default_factory=dict)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    _cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def invalidate_cache(self) -> None:
        """Explicitly invalidate internal derived property cache."""
        self._cache.clear()

    # --------------------------------------------------------------------------
    # Computed Properties (Derived from records to avoid state duplication)
    # --------------------------------------------------------------------------

    @property
    def latest_task_records(self) -> Dict[str, TaskExecutionRecord]:
        """Map of task ID to its most recent execution record."""
        if "latest_task_records" not in self._cache:
            latest: Dict[str, TaskExecutionRecord] = {}
            for r in self.records:
                latest[r.task_id] = r
            self._cache["latest_task_records"] = latest
        return self._cache["latest_task_records"]

    @property
    def completed_task_ids(self) -> Set[str]:
        """Set of task IDs that have succeeded in any execution wave."""
        if "completed_task_ids" not in self._cache:
            self._cache["completed_task_ids"] = {r.task_id for r in self.records if r.status == TaskStatus.COMPLETED}
        return self._cache["completed_task_ids"]

    @property
    def failed_task_ids(self) -> Set[str]:
        """Set of task IDs whose latest attempt ended in failure."""
        if "failed_task_ids" not in self._cache:
            completed = self.completed_task_ids
            self._cache["failed_task_ids"] = {
                tid for tid, r in self.latest_task_records.items()
                if r.status == TaskStatus.FAILED and tid not in completed
            }
        return self._cache["failed_task_ids"]

    @property
    def skipped_task_ids(self) -> Set[str]:
        """Set of task IDs whose latest attempt was skipped."""
        if "skipped_task_ids" not in self._cache:
            completed = self.completed_task_ids
            self._cache["skipped_task_ids"] = {
                tid for tid, r in self.latest_task_records.items()
                if r.status == TaskStatus.SKIPPED and tid not in completed
            }
        return self._cache["skipped_task_ids"]

    @property
    def completed_tasks(self) -> List[Task]:
        """List of completed Task models derived from successful records."""
        if "completed_tasks" not in self._cache:
            tasks: List[Task] = []
            seen: Set[str] = set()
            for r in self.records:
                if r.status == TaskStatus.COMPLETED and r.task_id not in seen:
                    seen.add(r.task_id)
                    tasks.append(
                        Task(
                            id=r.task_id,
                            action=r.action,
                            target=r.target,
                            status=TaskStatus.COMPLETED,
                        )
                    )
            self._cache["completed_tasks"] = tasks
        return self._cache["completed_tasks"]

    @property
    def failed_tasks(self) -> List[Task]:
        """List of Task models currently in failed state."""
        if "failed_tasks" not in self._cache:
            completed = self.completed_task_ids
            tasks: List[Task] = []
            seen: Set[str] = set()
            for r in reversed(self.records):
                if r.status == TaskStatus.FAILED and r.task_id not in completed and r.task_id not in seen:
                    seen.add(r.task_id)
                    tasks.append(
                        Task(
                            id=r.task_id,
                            action=r.action,
                            target=r.target,
                            status=TaskStatus.FAILED,
                        )
                    )
            self._cache["failed_tasks"] = list(reversed(tasks))
        return self._cache["failed_tasks"]

    @property
    def skipped_tasks(self) -> List[Task]:
        """List of Task models currently in skipped state."""
        if "skipped_tasks" not in self._cache:
            completed = self.completed_task_ids
            tasks: List[Task] = []
            seen: Set[str] = set()
            for r in reversed(self.records):
                if r.status == TaskStatus.SKIPPED and r.task_id not in completed and r.task_id not in seen:
                    seen.add(r.task_id)
                    tasks.append(
                        Task(
                            id=r.task_id,
                            action=r.action,
                            target=r.target,
                            status=TaskStatus.SKIPPED,
                        )
                    )
            self._cache["skipped_tasks"] = list(reversed(tasks))
        return self._cache["skipped_tasks"]

    @property
    def execution_order(self) -> List[str]:
        """Deterministic chronological list of unique task IDs executed."""
        if "execution_order" not in self._cache:
            order: List[str] = []
            seen: Set[str] = set()
            for r in self.records:
                if r.task_id not in seen:
                    seen.add(r.task_id)
                    order.append(r.task_id)
            self._cache["execution_order"] = order
        return self._cache["execution_order"]

    @property
    def retry_history(self) -> Dict[str, int]:
        """Number of execution attempts made per task ID."""
        if "retry_history" not in self._cache:
            counts: Dict[str, int] = {}
            for r in self.records:
                counts[r.task_id] = counts.get(r.task_id, 0) + 1
            self._cache["retry_history"] = counts
        return self._cache["retry_history"]

    @property
    def task_outputs(self) -> Dict[str, Any]:
        """Map of task ID to latest non-empty output."""
        if "task_outputs" not in self._cache:
            outputs: Dict[str, Any] = {}
            for r in self.records:
                if r.output is not None:
                    outputs[r.task_id] = r.output
            self._cache["task_outputs"] = outputs
        return self._cache["task_outputs"]

    # --------------------------------------------------------------------------
    # Factory & Accumulation Methods
    # --------------------------------------------------------------------------

    @classmethod
    def from_execution_result(
        cls,
        result: ExecutionResult,
        wave: int = 1,
        classifier: Optional[Any] = None,
    ) -> ExecutionMemory:
        """Create a new ExecutionMemory initialized from an ExecutionResult wave."""
        memory = cls(
            records=[],
            dependency_failures={k: list(v) for k, v in result.dependency_failures.items()},
            metrics=ExecutionMetrics(execution_waves=wave),
        )
        return memory.record_execution(result, wave=wave, classifier=classifier)

    def record_execution(
        self,
        result: ExecutionResult,
        wave: Optional[int] = None,
        classifier: Optional[Any] = None,
    ) -> ExecutionMemory:
        """Record an ExecutionResult wave into the execution history, returning an updated ExecutionMemory.

        Derives failure categories using the provided classifier (or default FailureClassifier).
        """
        if classifier is None:
            from app.ai.planner.failure_classifier import FailureClassifier
            classifier = FailureClassifier

        target_wave = wave if wave is not None else (self.metrics.execution_waves + 1)
        new_records: List[TaskExecutionRecord] = list(self.records)
        current_retries = dict(self.retry_history)

        # 1. Record completed tasks
        for task in result.completed_tasks:
            attempt = current_retries.get(task.id, 0) + 1
            current_retries[task.id] = attempt
            out = result.output if len(result.completed_tasks) == 1 else None
            new_records.append(
                TaskExecutionRecord(
                    task_id=task.id,
                    action=task.action,
                    target=task.target,
                    status=TaskStatus.COMPLETED,
                    wave=target_wave,
                    attempt=attempt,
                    output=out,
                    error=None,
                    failure_category=FailureCategory.UNKNOWN,
                )
            )

        # 2. Record failed tasks
        for task in result.failed_tasks:
            attempt = current_retries.get(task.id, 0) + 1
            current_retries[task.id] = attempt
            err = result.output or "Task execution failed"
            category = classifier.classify(task, error=err, dependency_failures=result.dependency_failures)
            new_records.append(
                TaskExecutionRecord(
                    task_id=task.id,
                    action=task.action,
                    target=task.target,
                    status=TaskStatus.FAILED,
                    wave=target_wave,
                    attempt=attempt,
                    output=None,
                    error=err,
                    failure_category=category,
                )
            )

        # 3. Record skipped tasks
        for task in result.skipped_tasks:
            attempt = current_retries.get(task.id, 0) + 1
            current_retries[task.id] = attempt
            failed_deps = result.dependency_failures.get(task.id, [])
            err = f"Skipped due to failed dependencies: {failed_deps}"
            category = FailureCategory.DEPENDENCY_FAILURE
            new_records.append(
                TaskExecutionRecord(
                    task_id=task.id,
                    action=task.action,
                    target=task.target,
                    status=TaskStatus.SKIPPED,
                    wave=target_wave,
                    attempt=attempt,
                    output=None,
                    error=err,
                    failure_category=category,
                )
            )

        # 4. Merge dependency failures
        merged_deps: Dict[str, List[str]] = {k: list(v) for k, v in self.dependency_failures.items()}
        for k, v in result.dependency_failures.items():
            if k in merged_deps:
                seen = set(merged_deps[k])
                for dep in v:
                    if dep not in seen:
                        merged_deps[k].append(dep)
                        seen.add(dep)
            else:
                merged_deps[k] = list(v)

        # 5. Update metrics
        updated_metrics = ExecutionMetrics(
            planning_count=self.metrics.planning_count,
            replan_count=self.metrics.replan_count,
            successful_recoveries=self.metrics.successful_recoveries,
            failed_recoveries=self.metrics.failed_recoveries,
            total_recovery_duration=self.metrics.total_recovery_duration,
            execution_waves=max(self.metrics.execution_waves, target_wave),
        )

        return ExecutionMemory(
            records=new_records,
            dependency_failures=merged_deps,
            metrics=updated_metrics,
        )

    def merge(self, other: ExecutionMemory) -> ExecutionMemory:
        """Immutably merge this ExecutionMemory with another.

        Concatenates records, merges dependency failure mappings, and sums metrics.
        """
        merged_records = list(self.records) + list(other.records)
        merged_deps: Dict[str, List[str]] = {k: list(v) for k, v in self.dependency_failures.items()}
        for k, v in other.dependency_failures.items():
            if k in merged_deps:
                seen = set(merged_deps[k])
                for dep in v:
                    if dep not in seen:
                        merged_deps[k].append(dep)
                        seen.add(dep)
            else:
                merged_deps[k] = list(v)

        merged_metrics = ExecutionMetrics(
            planning_count=self.metrics.planning_count + other.metrics.planning_count,
            replan_count=self.metrics.replan_count + other.metrics.replan_count,
            successful_recoveries=self.metrics.successful_recoveries + other.metrics.successful_recoveries,
            failed_recoveries=self.metrics.failed_recoveries + other.metrics.failed_recoveries,
            total_recovery_duration=self.metrics.total_recovery_duration + other.metrics.total_recovery_duration,
            execution_waves=max(self.metrics.execution_waves, other.metrics.execution_waves),
        )

        return ExecutionMemory(
            records=merged_records,
            dependency_failures=merged_deps,
            metrics=merged_metrics,
        )

    def summarize_for_prompt(self, original_plan: Optional[List[Task]] = None) -> str:
        """Build concise execution summary suitable for injection into planner prompt."""
        from app.ai.planner.memory_summary import MemorySummaryBuilder
        return MemorySummaryBuilder.build_planner_summary(self, original_plan or [])

    def get_summary(self) -> Dict[str, Any]:
        """Construct structured metadata summary of execution memory."""
        from app.ai.planner.memory_summary import MemorySummaryBuilder
        return MemorySummaryBuilder.build_metadata_summary(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExecutionMemory to dictionary."""
        return {
            "records": [r.to_dict() for r in self.records],
            "dependency_failures": {k: list(v) for k, v in self.dependency_failures.items()},
            "metrics": self.metrics.to_dict(),
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "skipped_task_ids": list(self.skipped_task_ids),
            "execution_order": self.execution_order,
            "retry_history": self.retry_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionMemory:
        """Deserialize dictionary to ExecutionMemory, rebuilding caches automatically."""
        records = [TaskExecutionRecord.from_dict(r) for r in data.get("records", [])]
        dep_failures = {k: list(v) for k, v in data.get("dependency_failures", {}).items()}
        raw_metrics = data.get("metrics")
        metrics = ExecutionMetrics.from_dict(raw_metrics) if isinstance(raw_metrics, dict) else ExecutionMetrics()
        return cls(
            records=records,
            dependency_failures=dep_failures,
            metrics=metrics,
        )
