"""Unit Tests for Offline Planner Replay Engine (Sprint 17.2).

Verifies step forward, step backward, seek, seek_time, reset,
snapshot accuracy, and offline isolation invariants.
"""

from __future__ import annotations

import unittest

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    PlanStarted,
    RecoveryCompleted,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
)
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord
from app.ai.planner.models import Task, TaskStatus
from app.ai.planner.persistence import PersistedExecutionState
from app.ai.planner.replay import PlannerReplayEngine
from app.ai.planner.timeline import ExecutionTimeline


class TestPlannerReplayEngine(unittest.TestCase):
    """Test suite for offline replay engine functionality."""

    def setUp(self) -> None:
        """Construct timeline with rich event sequence."""
        self.bus = PlannerEventBus()
        self.timeline = ExecutionTimeline(self.bus)

        # Emit simulated event timeline
        self.bus.publish(PlanStarted(plan_id="p1", query="test query", task_count=2))
        self.bus.publish(TaskStarted(plan_id="p1", task_id="t1", action="open_app"))
        self.bus.publish(TaskCompleted(plan_id="p1", task_id="t1", action="open_app", result="OK", duration=0.02))
        self.bus.publish(TaskStarted(plan_id="p1", task_id="t2", action="web_search"))
        self.bus.publish(TaskFailed(plan_id="p1", task_id="t2", action="web_search", error="timeout", duration=0.05))
        self.bus.publish(RecoveryStarted(query="test query", attempt=2, failed_task_ids=["t2"], skipped_task_ids=["t3"]))
        self.bus.publish(TaskStarted(plan_id="p2", task_id="t2_retry", action="web_search"))
        self.bus.publish(TaskCompleted(plan_id="p2", task_id="t2_retry", action="web_search", result="Found", duration=0.03))
        self.bus.publish(RecoveryCompleted(query="test query", attempt=2, success=True, duration=0.05))
        self.bus.publish(PlanCompleted(plan_id="p1", success=True, completed_count=2, failed_count=0, skipped_count=0, duration=0.15))

    def test_step_and_step_back_navigation(self) -> None:
        """Verify sequential forward and backward stepping through timeline."""
        engine = PlannerReplayEngine(self.timeline)
        self.assertEqual(engine.total_steps, 10)
        self.assertEqual(engine.current_cursor, -1)

        # Initial snapshot before step
        snap = engine.get_snapshot()
        self.assertEqual(snap["cursor"], -1)
        self.assertEqual(snap["current_execution_progress"], 0.0)
        self.assertEqual(len(snap["completed_tasks"]), 0)

        # Step 1: PlanStarted
        s1 = engine.step()
        self.assertIsNotNone(s1)
        self.assertEqual(s1["cursor"], 0)
        self.assertEqual(s1["current_event"]["event_type"], "PlanStarted")

        # Step 2: TaskStarted t1
        s2 = engine.step()
        self.assertEqual(s2["task_lifecycle"].get("t1"), TaskStatus.RUNNING.value)

        # Step 3: TaskCompleted t1
        s3 = engine.step()
        self.assertEqual(s3["task_lifecycle"].get("t1"), TaskStatus.COMPLETED.value)
        self.assertIn("t1", s3["completed_tasks"])

        # Step back to Step 2
        b2 = engine.step_back()
        self.assertIsNotNone(b2)
        self.assertEqual(b2["cursor"], 1)
        self.assertEqual(b2["task_lifecycle"].get("t1"), TaskStatus.RUNNING.value)
        self.assertNotIn("t1", b2["completed_tasks"])

        # Step back to Step 1
        b1 = engine.step_back()
        self.assertEqual(b1["cursor"], 0)

        # Step back to initial state
        b0 = engine.step_back()
        self.assertEqual(b0["cursor"], -1)

        # Step back past beginning returns None
        self.assertIsNone(engine.step_back())

    def test_seek_and_reset(self) -> None:
        """Verify seek jumps directly to arbitrary index and reset clears cursor."""
        engine = PlannerReplayEngine(self.timeline)

        # Seek to step 4 (TaskFailed t2)
        s4 = engine.seek(4)
        self.assertEqual(s4["cursor"], 4)
        self.assertEqual(s4["current_event"]["event_type"], "TaskFailed")
        self.assertIn("t2", s4["failed_tasks"])

        # Seek to end
        s_end = engine.seek(9)
        self.assertEqual(s_end["cursor"], 9)
        self.assertEqual(s_end["current_execution_progress"], 1.0)
        self.assertTrue(engine.is_at_end)

        # Seek out of bounds is clamped
        s_clamp = engine.seek(999)
        self.assertEqual(s_clamp["cursor"], 9)

        # Reset
        s_reset = engine.reset()
        self.assertEqual(s_reset["cursor"], -1)
        self.assertEqual(s_reset["current_execution_progress"], 0.0)

    def test_seek_time(self) -> None:
        """Verify seek_time navigates to the event closest to target timestamp."""
        engine = PlannerReplayEngine(self.timeline)
        events = self.timeline.events
        target_ts = events[3].timestamp

        snap = engine.seek_time(target_ts)
        self.assertEqual(snap["cursor"], 3)
        self.assertEqual(snap["current_event"]["event_type"], "TaskStarted")

    def test_pause_resume_and_cancellation_tracking(self) -> None:
        """Verify replay captures pause, resume, and cancellation states."""
        bus = PlannerEventBus()
        tl = ExecutionTimeline(bus)

        bus.publish(PlanStarted(plan_id="p_ctrl"))
        bus.publish(PlanPaused(plan_id="p_ctrl", reason="User paused"))
        bus.publish(PlanResumed(plan_id="p_ctrl", reason="User resumed"))
        bus.publish(PlanCancelled(plan_id="p_ctrl", reason="Aborted by user"))

        engine = PlannerReplayEngine(tl)
        engine.seek(1)
        self.assertTrue(engine.get_snapshot()["is_paused"])
        self.assertFalse(engine.get_snapshot()["is_cancelled"])

        engine.seek(2)
        self.assertFalse(engine.get_snapshot()["is_paused"])
        self.assertFalse(engine.get_snapshot()["is_cancelled"])

        engine.seek(3)
        self.assertTrue(engine.get_snapshot()["is_cancelled"])

    def test_replay_from_persisted_execution_state(self) -> None:
        """Verify replay engine can initialize directly from PersistedExecutionState."""
        t1 = Task(id="task_1", action="calculate", target="1+1", status=TaskStatus.COMPLETED)
        state = PersistedExecutionState(
            execution_id="replay_exec_1",
            plan_id="replay_plan_1",
            query="calc query",
            memory=ExecutionMemory(
                records=[
                    TaskExecutionRecord(
                        task_id="task_1",
                        action="calculate",
                        target="1+1",
                        status=TaskStatus.COMPLETED,
                        duration=0.01,
                    )
                ]
            ),
            completed_tasks=[t1],
            dag_state={"task_map": {"task_1": t1}},
        )

        engine = PlannerReplayEngine(state)
        self.assertGreater(engine.total_steps, 0)
        engine.seek(engine.total_steps - 1)
        snap = engine.get_snapshot()
        self.assertIn("task_1", snap["completed_tasks"])


if __name__ == "__main__":
    unittest.main()
