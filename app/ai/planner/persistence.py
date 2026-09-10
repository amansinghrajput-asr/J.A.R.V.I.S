"""Execution Persistence Subsystem for J.A.R.V.I.S. Adaptive Planner.

Provides serialization, deserialization, and deterministic execution resumption
for planned executions and DAG progress without persisting transient caches,
event buses, or synchronization primitives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Set, TextIO, Union
import time

from app.ai.planner.memory import ExecutionMemory
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus


@dataclass
class PersistedExecutionState:
    """Serializable snapshot of a plan's execution state.

    Captures execution memory, DAG progress, retry history, and lifecycle metadata.
    Explicitly excludes runtime synchronization primitives, event buses, and lazy caches.

    Attributes:
        execution_id: Unique identifier for the execution instance.
        plan_id: Identifier of the plan being executed.
        query: Original user query.
        memory: Canonical ExecutionMemory containing execution records and metrics.
        dag_state: Detailed graph execution state (task definitions, in-degrees, output map).
        completed_tasks: Tasks that finished with COMPLETED status.
        failed_tasks: Tasks that finished with FAILED status.
        skipped_tasks: Tasks that were SKIPPED.
        retry_history: Mapping of task IDs to attempt counts.
        recovery_attempts: Number of replanning recovery attempts performed.
        metadata: Contextual planner and execution metadata.
        created_at: Unix timestamp when the execution was initiated.
        updated_at: Unix timestamp when this snapshot was created.
        schema_version: Version for state schema compatibility.
    """

    execution_id: str
    plan_id: str
    query: str
    memory: ExecutionMemory
    dag_state: Dict[str, Any] = field(default_factory=dict)
    completed_tasks: List[Task] = field(default_factory=list)
    failed_tasks: List[Task] = field(default_factory=list)
    skipped_tasks: List[Task] = field(default_factory=list)
    retry_history: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PersistedExecutionState to dictionary."""
        # Ensure DAG state tasks are properly serialized
        serialized_dag: Dict[str, Any] = dict(self.dag_state)
        if "task_map" in serialized_dag and isinstance(serialized_dag["task_map"], dict):
            serialized_dag["task_map"] = {
                tid: (t.to_dict() if isinstance(t, Task) else t)
                for tid, t in serialized_dag["task_map"].items()
            }

        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "plan_id": self.plan_id,
            "query": self.query,
            "memory": self.memory.to_dict(),
            "dag_state": serialized_dag,
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "failed_tasks": [t.to_dict() for t in self.failed_tasks],
            "skipped_tasks": [t.to_dict() for t in self.skipped_tasks],
            "retry_history": dict(self.retry_history),
            "recovery_attempts": self.recovery_attempts,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export state to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistedExecutionState:
        """Construct PersistedExecutionState from dictionary, rebuilding caches lazily."""
        raw_memory = data.get("memory", {})
        memory = ExecutionMemory.from_dict(raw_memory) if isinstance(raw_memory, dict) else ExecutionMemory()

        completed_tasks = [Task.from_dict(t) for t in data.get("completed_tasks", [])]
        failed_tasks = [Task.from_dict(t) for t in data.get("failed_tasks", [])]
        skipped_tasks = [Task.from_dict(t) for t in data.get("skipped_tasks", [])]

        raw_dag = dict(data.get("dag_state", {}))
        if "task_map" in raw_dag and isinstance(raw_dag["task_map"], dict):
            raw_dag["task_map"] = {
                tid: (Task.from_dict(t) if isinstance(t, dict) else t)
                for tid, t in raw_dag["task_map"].items()
            }

        return cls(
            execution_id=str(data.get("execution_id", "")),
            plan_id=str(data.get("plan_id", "")),
            query=str(data.get("query", "")),
            memory=memory,
            dag_state=raw_dag,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=skipped_tasks,
            retry_history=dict(data.get("retry_history", {})),
            recovery_attempts=int(data.get("recovery_attempts", 0)),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PersistedExecutionState:
        """Construct PersistedExecutionState from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def save(
    state: PersistedExecutionState,
    target: Optional[Union[str, Path, IO[str], TextIO]] = None,
) -> Optional[str]:
    """Save PersistedExecutionState to a JSON string, file path, or file-like object.

    Args:
        state: The PersistedExecutionState instance to persist.
        target: Optional target. If None, returns the JSON string. If str or Path,
            writes to the specified file path. If file-like (has .write()), writes to it.

    Returns:
        JSON string representation if target is None; otherwise None.
    """
    state.updated_at = time.time()
    json_data = state.to_json(indent=2)

    if target is None:
        return json_data

    if isinstance(target, (str, Path)):
        path = Path(target)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json_data)
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return None

    if hasattr(target, "write"):
        target.write(json_data)
        return None

    raise TypeError(f"Unsupported target type for save: {type(target).__name__}")


def load(
    source: Union[str, Path, IO[str], TextIO, Dict[str, Any]],
) -> PersistedExecutionState:
    """Load PersistedExecutionState from a JSON string, file path, file-like object, or dict.

    Args:
        source: JSON string, Path/str to a file, stream with .read(), or dictionary.

    Returns:
        Reconstructed PersistedExecutionState instance with caches ready to lazily recompute.
    """
    if isinstance(source, dict):
        return PersistedExecutionState.from_dict(source)

    if hasattr(source, "read"):
        content = source.read()
        return PersistedExecutionState.from_json(content)

    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            if path.exists() and path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return PersistedExecutionState.from_json(content)
        except OSError:
            pass

        if isinstance(source, str) and source.strip().startswith("{"):
            return PersistedExecutionState.from_json(source)

    raise ValueError(f"Unable to load PersistedExecutionState from source of type {type(source).__name__}")


def resume(
    state: PersistedExecutionState,
    executor: Optional[Any] = None,
    planner: Optional[Any] = None,
    **kwargs: Any,
) -> ExecutionResult:
    """Resume execution of a persisted state, guaranteeing completed tasks never execute again.

    Identifies all tasks that already reached COMPLETED status, constructs a sub-plan
    containing only the remaining uncompleted tasks with their satisfied dependencies resolved,
    executes them through the executor, and merges the new results with prior completed tasks.

    Args:
        state: The PersistedExecutionState to resume.
        executor: Optional Executor instance. If None, resolves from default singleton.
        planner: Optional Planner instance.
        **kwargs: Extra parameters passed to executor.execute_plan.

    Returns:
        Unified ExecutionResult reflecting the complete execution across all phases.
    """
    if executor is None:
        from app.ai.planner.executor import executor as default_exec
        executor = default_exec

    # 1. Collect all completed task IDs (MUST NEVER EXECUTE AGAIN)
    completed_task_ids: Set[str] = set(state.memory.completed_task_ids)
    completed_task_ids.update(t.id for t in state.completed_tasks)
    if "completed_ids" in state.dag_state:
        completed_task_ids.update(state.dag_state["completed_ids"])

    # 2. Reconstruct the full task catalog
    all_tasks: Dict[str, Task] = {}

    if "task_map" in state.dag_state and isinstance(state.dag_state["task_map"], dict):
        for tid, t in state.dag_state["task_map"].items():
            all_tasks[tid] = t if isinstance(t, Task) else Task.from_dict(t)

    for t in state.completed_tasks:
        all_tasks.setdefault(t.id, t)
    for t in state.failed_tasks:
        all_tasks.setdefault(t.id, t)
    for t in state.skipped_tasks:
        all_tasks.setdefault(t.id, t)

    # 3. Identify remaining uncompleted tasks
    remaining_tasks: List[Task] = []
    for tid, task in all_tasks.items():
        if tid in completed_task_ids:
            continue  # CRITICAL INVARIANT: Completed tasks never run again

        # Strip dependencies that have already completed
        uncompleted_deps = [dep for dep in task.dependencies if dep not in completed_task_ids]

        resumed_task = Task(
            id=task.id,
            action=task.action,
            target=task.target,
            parameters=dict(task.parameters),
            status=TaskStatus.PENDING,
            dependencies=uncompleted_deps,
        )
        remaining_tasks.append(resumed_task)

    if not remaining_tasks:
        return ExecutionResult(
            success=len(state.failed_tasks) == 0 and len(state.skipped_tasks) == 0,
            completed_tasks=list(state.completed_tasks),
            failed_tasks=list(state.failed_tasks),
            skipped_tasks=list(state.skipped_tasks),
            output="Resumed execution: all tasks already completed.",
            execution_order=state.memory.execution_order,
            dependency_failures=dict(state.memory.dependency_failures),
        )

    # 4. Construct resume plan
    resume_plan = Plan(
        query=state.query,
        tasks=remaining_tasks,
        metadata={
            "is_resume": True,
            "original_plan_id": state.plan_id,
            "resumed_execution_id": state.execution_id,
        },
    )

    # 5. Execute remaining tasks
    new_result = executor.execute_plan(resume_plan, **kwargs)

    # 6. Merge prior completed tasks with new execution result
    merged_completed = list(state.completed_tasks)
    seen_completed = {t.id for t in merged_completed}
    for t in new_result.completed_tasks:
        if t.id not in seen_completed:
            merged_completed.append(t)
            seen_completed.add(t.id)

    merged_order = list(state.memory.execution_order)
    for tid in new_result.execution_order:
        if tid not in merged_order:
            merged_order.append(tid)

    merged_dep_failures = dict(state.memory.dependency_failures)
    for k, v in new_result.dependency_failures.items():
        merged_dep_failures.setdefault(k, []).extend([d for d in v if d not in merged_dep_failures.get(k, [])])

    updated_memory = state.memory.record_execution(new_result)
    state.memory = updated_memory
    state.completed_tasks = merged_completed
    state.failed_tasks = new_result.failed_tasks
    state.skipped_tasks = new_result.skipped_tasks

    return ExecutionResult(
        success=new_result.success,
        completed_tasks=merged_completed,
        failed_tasks=new_result.failed_tasks,
        skipped_tasks=new_result.skipped_tasks,
        output=new_result.output,
        execution_order=merged_order,
        dependency_failures=merged_dep_failures,
        execution_duration=new_result.execution_duration,
    )
