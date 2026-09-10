"""Data models for the J.A.R.V.I.S Planner Subsystem.

Defines TaskStatus enumeration, Task model, Plan container,
and ExecutionResult structures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TaskStatus(str, Enum):
    """Execution lifecycle status for planned tasks."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class PlanningStrategy(str, Enum):
    """Planning strategy modes for task decomposition."""

    RULE_BASED = "RULE_BASED"
    LLM = "LLM"
    HYBRID = "HYBRID"


@dataclass
class Task:
    """Represents a single atomic planned action.

    Attributes:
        action: Identifier for the action to execute (e.g. 'open_app', 'web_search').
        target: Optional primary target or operand of the action (e.g. 'chrome', 'weather').
        parameters: Optional dictionary of arguments or metadata.
        status: Current task lifecycle status.
        id: Unique identifier for tracking task execution.
        dependencies: Task IDs that must complete before this task executes.
    """

    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Task to a dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "target": self.target,
            "parameters": dict(self.parameters),
            "status": self.status.value,
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Deserialize dictionary to Task."""
        status_val = data.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_val)
        except Exception:
            status = TaskStatus.PENDING
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            action=str(data.get("action", "")),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            status=status,
            dependencies=list(data.get("dependencies", [])),
        )


@dataclass
class Plan:
    """Represents an ordered sequence of tasks created to satisfy a user query.

    Attributes:
        query: The original user query.
        tasks: Ordered list of tasks to execute.
        id: Unique identifier for the plan.
        created_at: Unix timestamp marking plan creation.
        strategy: Planning strategy utilized to generate this plan.
        metadata: Optional metadata dictionary associated with plan generation.
    """

    query: str
    tasks: List[Task] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    strategy: PlanningStrategy = PlanningStrategy.RULE_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Plan to a dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "tasks": [t.to_dict() for t in self.tasks],
            "strategy": self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Plan:
        """Deserialize dictionary to Plan."""
        strategy_val = data.get("strategy", PlanningStrategy.RULE_BASED.value)
        try:
            strategy = PlanningStrategy(strategy_val)
        except Exception:
            strategy = PlanningStrategy.RULE_BASED
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            query=str(data.get("query", "")),
            tasks=tasks,
            strategy=strategy,
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )

    def is_empty(self) -> bool:
        """Return True if plan contains no tasks."""
        return len(self.tasks) == 0


@dataclass
class ExecutionResult:
    """Outcome of executing a plan.

    Attributes:
        success: Whether all tasks completed successfully.
        completed_tasks: List of tasks that executed successfully.
        failed_tasks: List of tasks that failed.
        output: Human-readable output summary or return value.
        skipped_tasks: List of tasks skipped due to dependency failures.
        execution_order: Order of task IDs executed during plan processing.
        dependency_failures: Mapping of skipped task ID to failed dependency task IDs.
    """

    success: bool
    completed_tasks: List[Task] = field(default_factory=list)
    failed_tasks: List[Task] = field(default_factory=list)
    output: Optional[str] = None
    skipped_tasks: List[Task] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    dependency_failures: Dict[str, List[str]] = field(default_factory=dict)
    execution_duration: float = 0.0

    @property
    def completed_task_ids(self) -> Set[str]:
        """Return a set of task IDs that completed successfully."""
        return {t.id for t in self.completed_tasks}

    @property
    def failed_task_ids(self) -> Set[str]:
        """Return a set of task IDs that failed."""
        return {t.id for t in self.failed_tasks}

    @property
    def skipped_task_ids(self) -> Set[str]:
        """Return a set of task IDs that were skipped."""
        return {t.id for t in self.skipped_tasks}

    def merge(self, other: ExecutionResult) -> ExecutionResult:
        """Immutably merge this ExecutionResult with another (recovery) ExecutionResult.

        Merge Contract:
        1. Immutability: Returns a new ExecutionResult instance without mutating either
           `self` or `other`.
        2. Completed Tasks: Preserves all completed tasks from `self`, then appends completed
           tasks from `other` whose task IDs have not already been recorded, preventing duplicates.
        3. Execution Order: Preserves execution order from `self`, then appends task IDs from
           `other` that have not yet appeared, maintaining deterministic execution history.
        4. Dependency Failures: Merges failure mapping across both results. If a task ID exists
           in both, unique failed dependency IDs are unioned without duplicates.
        5. Execution Duration: Total duration is accumulated (self.execution_duration + other.execution_duration).
        6. Authoritative Final Outcome: `other` represents the recovery execution and provides the
           authoritative final values for `failed_tasks`, `skipped_tasks`, and `success`.
        7. Sensible Output: Uses `other.output` if non-empty; falls back to `self.output` otherwise.

        Args:
            other: The subsequent (e.g. recovery) ExecutionResult to merge.

        Returns:
            A new merged ExecutionResult instance.
        """
        # 1. Merge completed tasks preserving order without duplicate task IDs
        seen_completed: Set[str] = set()
        merged_completed: List[Task] = []
        for t in self.completed_tasks:
            if t.id not in seen_completed:
                seen_completed.add(t.id)
                merged_completed.append(t)
        for t in other.completed_tasks:
            if t.id not in seen_completed:
                seen_completed.add(t.id)
                merged_completed.append(t)

        # 2. Merge execution order deterministically without duplicate task IDs
        seen_order: Set[str] = set()
        merged_order: List[str] = []
        for tid in self.execution_order:
            if tid not in seen_order:
                seen_order.add(tid)
                merged_order.append(tid)
        for tid in other.execution_order:
            if tid not in seen_order:
                seen_order.add(tid)
                merged_order.append(tid)

        # 3. Merge dependency failures
        merged_deps: Dict[str, List[str]] = {k: list(v) for k, v in self.dependency_failures.items()}
        for k, v in other.dependency_failures.items():
            if k in merged_deps:
                seen_deps = set(merged_deps[k])
                for dep in v:
                    if dep not in seen_deps:
                        merged_deps[k].append(dep)
                        seen_deps.add(dep)
            else:
                merged_deps[k] = list(v)

        # 4. Resolve output sensibly
        resolved_output = (
            other.output
            if (other.output is not None and str(other.output).strip())
            else self.output
        )

        return ExecutionResult(
            success=other.success,
            completed_tasks=merged_completed,
            failed_tasks=list(other.failed_tasks),
            skipped_tasks=list(other.skipped_tasks),
            execution_order=merged_order,
            dependency_failures=merged_deps,
            execution_duration=self.execution_duration + other.execution_duration,
            output=resolved_output,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExecutionResult to a dictionary."""
        return {
            "success": self.success,
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "failed_tasks": [t.to_dict() for t in self.failed_tasks],
            "skipped_tasks": [t.to_dict() for t in self.skipped_tasks],
            "execution_order": list(self.execution_order),
            "dependency_failures": {k: list(v) for k, v in self.dependency_failures.items()},
            "execution_duration": self.execution_duration,
            "output": self.output,
        }
