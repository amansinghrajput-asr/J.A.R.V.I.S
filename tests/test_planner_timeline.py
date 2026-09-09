"""Unit tests for the Execution Timeline Recorder (Phase 17.0).

Verifies:
- Deterministic event ordering
- Filtering by task ID, event type, and timestamp window
- Duration calculations for plans and individual tasks
- Recovery replanning wave extraction
- Structured JSON export
- Formatted Markdown timeline export
"""

from __future__ import annotations

import json
import time
import unittest

from app.ai.planner.events import (
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
from app.ai.planner.timeline import ExecutionTimeline


class TestPlannerTimeline(unittest.TestCase):
    """Test suite for ExecutionTimeline."""

    def test_timeline_recording_and_ordering(self) -> None:
        """Timeline must record events and maintain sequence/timestamp ordering."""
        bus = PlannerEventBus()
        timeline = ExecutionTimeline(bus=bus)

        e1 = PlanStarted(plan_id="p1", query="search news", task_count=2)
        e2 = TaskStarted(task_id="t1", action="open_app")
        e3 = TaskCompleted(task_id="t1", action="open_app", result="opened", duration=0.1)

        bus.publish(e1)
        bus.publish(e2)
        bus.publish(e3)

        self.assertEqual(len(timeline), 3)
        self.assertEqual(timeline.events[0].sequence_id, 1)
        self.assertEqual(timeline.events[1].sequence_id, 2)
        self.assertEqual(timeline.events[2].sequence_id, 3)

    def test_filter_by_task(self) -> None:
        """filter_by_task must isolate events for a specific task ID."""
        bus = PlannerEventBus()
        timeline = ExecutionTimeline(bus=bus)

        bus.publish(TaskStarted(task_id="t1", action="open_app"))
        bus.publish(TaskStarted(task_id="t2", action="web_search"))
        bus.publish(TaskCompleted(task_id="t1", action="open_app", result="OK"))
        bus.publish(TaskFailed(task_id="t2", action="web_search", error="timeout"))

        t1_events = timeline.filter_by_task("t1")
        t2_events = timeline.filter_by_task("t2")

        self.assertEqual(len(t1_events), 2)
        self.assertEqual(len(t2_events), 2)
        self.assertTrue(all(getattr(e, "task_id", "") == "t1" for e in t1_events))
        self.assertTrue(all(getattr(e, "task_id", "") == "t2" for e in t2_events))

    def test_filter_by_type_and_time(self) -> None:
        """filter_by_type and filter_by_time must accurately subset events."""
        timeline = ExecutionTimeline()

        t0 = 1000.0
        e1 = PlanStarted(plan_id="p1", sequence_id=1, timestamp=t0)
        e2 = TaskStarted(task_id="t1", sequence_id=2, timestamp=t0 + 5.0)
        e3 = TaskCompleted(task_id="t1", sequence_id=3, timestamp=t0 + 10.0)

        timeline.record(e1)
        timeline.record(e2)
        timeline.record(e3)

        completed = timeline.filter_by_type(TaskCompleted)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].task_id, "t1")

        window = timeline.filter_by_time(t0 + 1.0, t0 + 8.0)
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0].sequence_id, 2)

    def test_durations_and_retries(self) -> None:
        """get_plan_duration, get_task_durations, and get_task_retries must return correct analytics."""
        bus = PlannerEventBus()
        timeline = ExecutionTimeline(bus=bus)

        bus.publish(PlanStarted(plan_id="p1"))
        bus.publish(TaskStarted(task_id="t1"))
        bus.publish(TaskFailed(task_id="t1", error="net fail", duration=0.25))
        bus.publish(TaskRetried(task_id="t1", attempt=2, reason="transient"))
        bus.publish(TaskCompleted(task_id="t1", result="success", duration=0.30))
        bus.publish(PlanCompleted(plan_id="p1", duration=0.55))

        self.assertEqual(timeline.get_plan_duration(), 0.55)
        durations = timeline.get_task_durations()
        self.assertEqual(durations.get("t1"), 0.30)

        retries = timeline.get_task_retries("t1")
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0].attempt, 2)

    def test_recovery_waves_extraction(self) -> None:
        """get_recovery_waves must accurately assemble recovery attempts into structured summaries."""
        bus = PlannerEventBus()
        timeline = ExecutionTimeline(bus=bus)

        bus.publish(RecoveryStarted(query="test", attempt=1, failed_task_ids=["t1"]))
        bus.publish(RecoveryCompleted(query="test", attempt=1, success=True, new_plan_id="p_rec", task_count=1, duration=0.42))

        waves = timeline.get_recovery_waves()
        self.assertEqual(len(waves), 1)
        self.assertEqual(waves[0]["attempt"], 1)
        self.assertEqual(waves[0]["status"], "SUCCESS")
        self.assertEqual(waves[0]["duration"], 0.42)
        self.assertEqual(waves[0]["failed_tasks"], ["t1"])

    def test_json_and_markdown_exports(self) -> None:
        """to_json and to_markdown must produce non-empty, valid formats."""
        bus = PlannerEventBus()
        timeline = ExecutionTimeline(bus=bus)

        bus.publish(PlanStarted(plan_id="p1", query="calc 5+5", task_count=1))
        bus.publish(TaskStarted(task_id="t1", action="calculate"))
        bus.publish(TaskCompleted(task_id="t1", action="calculate", result="10", duration=0.05))
        bus.publish(PlanCompleted(plan_id="p1", success=True, duration=0.06))

        # JSON validation
        json_str = timeline.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["total_events"], 4)
        self.assertEqual(len(parsed["events"]), 4)

        # Markdown validation
        md_str = timeline.to_markdown()
        self.assertIn("# Planner Execution Timeline", md_str)
        self.assertIn("| Seq | Relative Time | Event |", md_str)
        self.assertIn("calc 5+5", md_str)
        self.assertIn("calculate", md_str)


if __name__ == "__main__":
    unittest.main()
