"""Metrics Collector for the J.A.R.V.I.S. Planner Subsystem.

Subscribes to PlannerEventBus and aggregates operational performance metrics,
including latencies, DAG characteristics, task failure categories, retries,
and recovery success rate, with export to JSON and Prometheus exposition format.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEvent,
    PlannerEventBus,
    PlanStarted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
)


class PlannerMetricsCollector:
    """Collects and aggregates real-time planner metrics via event subscriptions."""

    def __init__(self, bus: Optional[PlannerEventBus] = None) -> None:
        """Initialize metrics collector, optionally attaching to an event bus.

        Args:
            bus: Optional PlannerEventBus to subscribe to.
        """
        self._lock = threading.RLock()
        self._bus: Optional[PlannerEventBus] = bus
        self._unsubscribe_fn: Optional[Callable[[], bool]] = None

        # Counters
        self.plans_started: int = 0
        self.plans_completed: int = 0
        self.plans_failed: int = 0
        self.plans_cancelled: int = 0

        self.tasks_started: int = 0
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self.tasks_retried: int = 0

        self.recoveries_started: int = 0
        self.recoveries_successful: int = 0
        self.recoveries_failed: int = 0

        # Latency lists (in seconds)
        self.planning_latencies: List[float] = []
        self.execution_latencies: List[float] = []
        self.recovery_latencies: List[float] = []

        # Task duration lists (by action)
        self.task_durations: List[float] = []
        self.task_durations_by_action: Dict[str, List[float]] = {}

        # DAG metrics
        self.dag_sizes: List[int] = []
        self.dependency_depths: List[int] = []

        # Token metrics
        self.estimated_prompt_tokens: int = 0

        # Failure categories
        self.failure_categories: Dict[str, int] = {}

        # Retries per task
        self.retries_per_task: Dict[str, int] = {}

        if bus is not None:
            self.attach(bus)

    def attach(self, bus: PlannerEventBus) -> None:
        """Attach to a PlannerEventBus and start collecting metrics."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                self.detach()
            self._bus = bus
            self._unsubscribe_fn = bus.subscribe(PlannerEvent, self.on_event)

    def detach(self) -> None:
        """Detach from the active PlannerEventBus."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                try:
                    self._unsubscribe_fn()
                except Exception:
                    pass
                self._unsubscribe_fn = None
            self._bus = None

    def on_event(self, event: PlannerEvent) -> None:
        """Process an incoming planner event and update internal aggregations."""
        with self._lock:
            if isinstance(event, PlanStarted):
                self.plans_started += 1
                if event.task_count > 0:
                    self.dag_sizes.append(event.task_count)
                depth = event.metadata.get("dependency_depth")
                if depth is not None and isinstance(depth, int):
                    self.dependency_depths.append(depth)
                tokens = event.metadata.get("estimated_prompt_tokens")
                if tokens is not None and isinstance(tokens, (int, float)):
                    self.estimated_prompt_tokens += int(tokens)
                p_latency = event.metadata.get("planning_latency")
                if p_latency is not None and isinstance(p_latency, (int, float)):
                    self.planning_latencies.append(float(p_latency))

            elif isinstance(event, PlanCompleted):
                self.plans_completed += 1
                if event.duration > 0:
                    self.execution_latencies.append(event.duration)

            elif isinstance(event, PlanFailed):
                self.plans_failed += 1

            elif isinstance(event, PlanCancelled):
                self.plans_cancelled += 1

            elif isinstance(event, TaskStarted):
                self.tasks_started += 1

            elif isinstance(event, TaskCompleted):
                self.tasks_completed += 1
                if event.duration > 0:
                    self.task_durations.append(event.duration)
                    action = event.action or "unknown"
                    self.task_durations_by_action.setdefault(action, []).append(event.duration)

            elif isinstance(event, TaskFailed):
                self.tasks_failed += 1
                if event.duration > 0:
                    self.task_durations.append(event.duration)
                    action = event.action or "unknown"
                    self.task_durations_by_action.setdefault(action, []).append(event.duration)
                cat = event.metadata.get("failure_category", "UNKNOWN")
                self.failure_categories[cat] = self.failure_categories.get(cat, 0) + 1

            elif isinstance(event, TaskRetried):
                self.tasks_retried += 1
                self.retries_per_task[event.task_id] = self.retries_per_task.get(event.task_id, 0) + 1

            elif isinstance(event, RecoveryStarted):
                self.recoveries_started += 1
                tokens = event.metadata.get("estimated_prompt_tokens")
                if tokens is not None and isinstance(tokens, (int, float)):
                    self.estimated_prompt_tokens += int(tokens)

            elif isinstance(event, RecoveryCompleted):
                if event.success:
                    self.recoveries_successful += 1
                else:
                    self.recoveries_failed += 1
                if event.duration > 0:
                    self.recovery_latencies.append(event.duration)

            elif isinstance(event, RecoveryFailed):
                self.recoveries_failed += 1
                if event.duration > 0:
                    self.recovery_latencies.append(event.duration)

    # --------------------------------------------------------------------------
    # Derived Statistics
    # --------------------------------------------------------------------------

    def get_average_planning_latency(self) -> float:
        """Return average planning latency in seconds."""
        with self._lock:
            return sum(self.planning_latencies) / len(self.planning_latencies) if self.planning_latencies else 0.0

    def get_average_execution_latency(self) -> float:
        """Return average execution latency in seconds."""
        with self._lock:
            return sum(self.execution_latencies) / len(self.execution_latencies) if self.execution_latencies else 0.0

    def get_average_recovery_latency(self) -> float:
        """Return average recovery latency in seconds."""
        with self._lock:
            return sum(self.recovery_latencies) / len(self.recovery_latencies) if self.recovery_latencies else 0.0

    def get_average_task_duration(self, action: Optional[str] = None) -> float:
        """Return average task duration in seconds, optionally filtered by action."""
        with self._lock:
            if action is not None:
                durations = self.task_durations_by_action.get(action, [])
            else:
                durations = self.task_durations
            return sum(durations) / len(durations) if durations else 0.0

    def get_recovery_success_rate(self) -> float:
        """Return recovery success rate as a ratio between 0.0 and 1.0."""
        with self._lock:
            total = self.recoveries_successful + self.recoveries_failed
            return (self.recoveries_successful / total) if total > 0 else 0.0

    def get_average_dag_size(self) -> float:
        """Return average task count across all plans."""
        with self._lock:
            return sum(self.dag_sizes) / len(self.dag_sizes) if self.dag_sizes else 0.0

    def get_average_dependency_depth(self) -> float:
        """Return average DAG dependency depth."""
        with self._lock:
            return sum(self.dependency_depths) / len(self.dependency_depths) if self.dependency_depths else 0.0

    # --------------------------------------------------------------------------
    # Serialization & Export
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export all aggregated metrics to a dictionary."""
        with self._lock:
            return {
                "plans": {
                    "started": self.plans_started,
                    "completed": self.plans_completed,
                    "failed": self.plans_failed,
                    "cancelled": self.plans_cancelled,
                },
                "tasks": {
                    "started": self.tasks_started,
                    "completed": self.tasks_completed,
                    "failed": self.tasks_failed,
                    "retried": self.tasks_retried,
                    "average_duration_seconds": round(self.get_average_task_duration(), 4),
                    "action_averages": {
                        act: round(sum(d) / len(d), 4)
                        for act, d in self.task_durations_by_action.items()
                        if d
                    },
                },
                "recovery": {
                    "started": self.recoveries_started,
                    "successful": self.recoveries_successful,
                    "failed": self.recoveries_failed,
                    "success_rate": round(self.get_recovery_success_rate(), 4),
                    "average_latency_seconds": round(self.get_average_recovery_latency(), 4),
                },
                "latencies": {
                    "average_planning_seconds": round(self.get_average_planning_latency(), 4),
                    "average_execution_seconds": round(self.get_average_execution_latency(), 4),
                    "average_recovery_seconds": round(self.get_average_recovery_latency(), 4),
                },
                "dag": {
                    "average_size": round(self.get_average_dag_size(), 2),
                    "average_depth": round(self.get_average_dependency_depth(), 2),
                },
                "tokens": {
                    "estimated_prompt_tokens_total": self.estimated_prompt_tokens,
                },
                "failure_categories": dict(self.failure_categories),
                "retries_per_task": dict(self.retries_per_task),
            }

    def to_json(self, indent: int = 2) -> str:
        """Export metrics to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_prometheus(self) -> str:
        """Export metrics in standard Prometheus text exposition format."""
        with self._lock:
            lines = [
                "# HELP jarvis_planner_plans_total Total number of plans by terminal status",
                "# TYPE jarvis_planner_plans_total counter",
                f'jarvis_planner_plans_total{{status="completed"}} {self.plans_completed}',
                f'jarvis_planner_plans_total{{status="failed"}} {self.plans_failed}',
                f'jarvis_planner_plans_total{{status="cancelled"}} {self.plans_cancelled}',
                "",
                "# HELP jarvis_planner_tasks_total Total number of task executions by status",
                "# TYPE jarvis_planner_tasks_total counter",
                f'jarvis_planner_tasks_total{{status="completed"}} {self.tasks_completed}',
                f'jarvis_planner_tasks_total{{status="failed"}} {self.tasks_failed}',
                f'jarvis_planner_tasks_total{{status="retried"}} {self.tasks_retried}',
                "",
                "# HELP jarvis_planner_planning_latency_seconds_average Average latency for planning",
                "# TYPE jarvis_planner_planning_latency_seconds_average gauge",
                f"jarvis_planner_planning_latency_seconds_average {self.get_average_planning_latency():.4f}",
                "",
                "# HELP jarvis_planner_execution_latency_seconds_average Average latency for execution",
                "# TYPE jarvis_planner_execution_latency_seconds_average gauge",
                f"jarvis_planner_execution_latency_seconds_average {self.get_average_execution_latency():.4f}",
                "",
                "# HELP jarvis_planner_recovery_success_rate Success rate of recovery attempts",
                "# TYPE jarvis_planner_recovery_success_rate gauge",
                f"jarvis_planner_recovery_success_rate {self.get_recovery_success_rate():.4f}",
                "",
                "# HELP jarvis_planner_dag_size_average Average number of tasks per plan",
                "# TYPE jarvis_planner_dag_size_average gauge",
                f"jarvis_planner_dag_size_average {self.get_average_dag_size():.2f}",
                "",
                "# HELP jarvis_planner_prompt_tokens_estimated_total Total estimated tokens in prompts",
                "# TYPE jarvis_planner_prompt_tokens_estimated_total counter",
                f"jarvis_planner_prompt_tokens_estimated_total {self.estimated_prompt_tokens}",
            ]
            return "\n".join(lines) + "\n"
