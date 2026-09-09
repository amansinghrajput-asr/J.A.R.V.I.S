"""Unit tests for the Planner Metrics Collector (Phase 17.0).

Verifies:
- Operational latency calculations (planning, execution, recovery)
- Task duration distributions and action-level averages
- Retry metrics and failure category breakdowns
- Recovery success rate calculation
- JSON serialization
- Prometheus text exposition format validation
"""

from __future__ import annotations

import json
import unittest

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
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
from app.ai.planner.metrics_collector import PlannerMetricsCollector


class TestPlannerMetricsCollector(unittest.TestCase):
    """Test suite for PlannerMetricsCollector."""

    def test_plan_and_task_counters(self) -> None:
        """Collector must accurately count started, completed, failed, and cancelled plans/tasks."""
        bus = PlannerEventBus()
        metrics = PlannerMetricsCollector(bus=bus)

        bus.publish(PlanStarted(plan_id="p1", task_count=3, metadata={"planning_latency": 0.15, "dependency_depth": 2}))
        bus.publish(TaskStarted(task_id="t1", action="open_app"))
        bus.publish(TaskCompleted(task_id="t1", action="open_app", duration=0.10))
        bus.publish(TaskStarted(task_id="t2", action="web_search"))
        bus.publish(TaskFailed(task_id="t2", action="web_search", duration=0.20, metadata={"failure_category": "TIMEOUT"}))
        bus.publish(PlanCompleted(plan_id="p1", success=False, duration=0.35))

        self.assertEqual(metrics.plans_started, 1)
        self.assertEqual(metrics.plans_completed, 1)
        self.assertEqual(metrics.tasks_started, 2)
        self.assertEqual(metrics.tasks_completed, 1)
        self.assertEqual(metrics.tasks_failed, 1)
        self.assertEqual(metrics.failure_categories.get("TIMEOUT"), 1)
        self.assertAlmostEqual(metrics.get_average_planning_latency(), 0.15)
        self.assertAlmostEqual(metrics.get_average_execution_latency(), 0.35)
        self.assertAlmostEqual(metrics.get_average_task_duration(), 0.15)  # (0.10 + 0.20) / 2
        self.assertAlmostEqual(metrics.get_average_dag_size(), 3.0)
        self.assertAlmostEqual(metrics.get_average_dependency_depth(), 2.0)

    def test_recovery_metrics_and_success_rate(self) -> None:
        """Collector must calculate recovery success rates and average recovery latencies."""
        bus = PlannerEventBus()
        metrics = PlannerMetricsCollector(bus=bus)

        # Attempt 1: succeeds
        bus.publish(RecoveryStarted(query="q1", attempt=1))
        bus.publish(RecoveryCompleted(query="q1", attempt=1, success=True, duration=0.40))

        # Attempt 2: fails
        bus.publish(RecoveryStarted(query="q2", attempt=1))
        bus.publish(RecoveryCompleted(query="q2", attempt=1, success=False, duration=0.60))

        self.assertEqual(metrics.recoveries_started, 2)
        self.assertEqual(metrics.recoveries_successful, 1)
        self.assertEqual(metrics.recoveries_failed, 1)
        self.assertAlmostEqual(metrics.get_recovery_success_rate(), 0.50)
        self.assertAlmostEqual(metrics.get_average_recovery_latency(), 0.50)

    def test_retry_metrics_per_task(self) -> None:
        """Collector must record retry occurrences per task ID."""
        bus = PlannerEventBus()
        metrics = PlannerMetricsCollector(bus=bus)

        bus.publish(TaskRetried(task_id="task-A", attempt=1))
        bus.publish(TaskRetried(task_id="task-A", attempt=2))
        bus.publish(TaskRetried(task_id="task-B", attempt=1))

        self.assertEqual(metrics.tasks_retried, 3)
        self.assertEqual(metrics.retries_per_task.get("task-A"), 2)
        self.assertEqual(metrics.retries_per_task.get("task-B"), 1)

    def test_json_and_prometheus_export(self) -> None:
        """Collector must generate valid JSON and standard Prometheus exposition text."""
        bus = PlannerEventBus()
        metrics = PlannerMetricsCollector(bus=bus)

        bus.publish(PlanStarted(plan_id="p1", task_count=2, metadata={"planning_latency": 0.1, "estimated_prompt_tokens": 150}))
        bus.publish(TaskStarted(task_id="t1", action="calculate"))
        bus.publish(TaskCompleted(task_id="t1", action="calculate", duration=0.05))
        bus.publish(PlanCompleted(plan_id="p1", duration=0.08))

        # JSON verification
        json_str = metrics.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["plans"]["started"], 1)
        self.assertEqual(data["tokens"]["estimated_prompt_tokens_total"], 150)

        # Prometheus verification
        prom_str = metrics.to_prometheus()
        self.assertIn("# TYPE jarvis_planner_plans_total counter", prom_str)
        self.assertIn('jarvis_planner_plans_total{status="completed"} 1', prom_str)
        self.assertIn("# TYPE jarvis_planner_planning_latency_seconds_average gauge", prom_str)
        self.assertIn("jarvis_planner_planning_latency_seconds_average 0.1000", prom_str)
        self.assertIn("jarvis_planner_prompt_tokens_estimated_total 150", prom_str)


if __name__ == "__main__":
    unittest.main()
