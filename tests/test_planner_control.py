"""Unit Tests for Execution Controller and Cancellation (Sprint 17.3).

Verifies pause, resume, cancellation, checkpoints (sync & async),
recovery abortion, thread safety, and event notifications.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import List

from app.ai.planner.control import (
    CancellationToken,
    ExecutionCancelledError,
    ExecutionController,
)
from app.ai.planner.events import (
    PlanCancelled,
    PlannerEvent,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    RecoveryAborted,
)


from app.ai.planner.executor import Executor
from app.ai.planner.models import Plan, Task, TaskStatus
from app.ai.planner.timeouts import TimeoutConfig, TimeoutPolicy


class TestPlannerControl(unittest.TestCase):
    """Test suite for execution control and cancellation mechanisms."""

    def test_cancellation_token_basic(self) -> None:
        """Verify cancellation token status and error raising."""
        token = CancellationToken()
        self.assertFalse(token.is_cancelled)
        self.assertIsNone(token.reason)

        # Should not raise when active
        token.raise_if_cancelled()

        # Cancel
        token.cancel("User requested stop")
        self.assertTrue(token.is_cancelled)
        self.assertEqual(token.reason, "User requested stop")

        with self.assertRaises(ExecutionCancelledError) as ctx:
            token.raise_if_cancelled()
        self.assertIn("User requested stop", str(ctx.exception))

    def test_sync_pause_and_resume(self) -> None:
        """Verify synchronous checkpoint blocks on pause and unblocks on resume."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        controller = ExecutionController(event_bus=bus, plan_id="p1")
        self.assertFalse(controller.is_paused)

        # Thread to wait at checkpoint
        checkpoint_reached = threading.Event()
        checkpoint_completed = threading.Event()

        def worker() -> None:
            checkpoint_reached.set()
            controller.check_checkpoint("test_worker")
            checkpoint_completed.set()

        # Pause first
        controller.pause("Maintenance")
        self.assertTrue(controller.is_paused)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        checkpoint_reached.wait(timeout=1.0)
        # Give worker time to block
        time.sleep(0.05)
        self.assertFalse(checkpoint_completed.is_set(), "Worker should be blocked at checkpoint while paused.")

        # Resume
        controller.resume("All clear")
        self.assertFalse(controller.is_paused)

        checkpoint_completed.wait(timeout=1.0)
        self.assertTrue(checkpoint_completed.is_set(), "Worker should proceed after resume.")
        t.join(timeout=1.0)

        # Verify emitted events
        event_types = [type(e) for e in events]
        self.assertIn(PlanPaused, event_types)
        self.assertIn(PlanResumed, event_types)

    def test_async_pause_and_resume(self) -> None:
        """Verify async checkpoint awaits on pause and unblocks on resume without blocking event loop."""
        async def run_test() -> None:
            controller = ExecutionController(plan_id="p_async")
            controller.pause("async hold")

            resumed_flag = False

            async def async_worker() -> None:
                nonlocal resumed_flag
                await controller.check_checkpoint_async("async_worker")
                resumed_flag = True

            task = asyncio.create_task(async_worker())

            # Yield control to allow worker to hit checkpoint
            await asyncio.sleep(0.02)
            self.assertFalse(resumed_flag, "Async worker must be suspended.")

            controller.resume("async continue")
            await asyncio.sleep(0.02)
            await task
            self.assertTrue(resumed_flag, "Async worker must complete after resume.")

        asyncio.run(run_test())

    def test_cancel_while_running(self) -> None:
        """Verify cancellation immediately raises ExecutionCancelledError at checkpoint."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        controller = ExecutionController(event_bus=bus, plan_id="p_cancel")
        controller.cancel("Emergency stop")
        self.assertTrue(controller.is_cancelled)

        with self.assertRaises(ExecutionCancelledError) as ctx:
            controller.check_checkpoint("running_task")
        self.assertIn("Emergency stop", str(ctx.exception))

        event_types = [type(e) for e in events]
        self.assertIn(PlanCancelled, event_types)

    def test_cancel_while_paused_unblocks_waiting_threads(self) -> None:
        """Verify cancellation unblocks paused threads so they can exit cleanly."""
        controller = ExecutionController(plan_id="p_pause_cancel")
        controller.pause("Suspended")

        caught_error: List[ExecutionCancelledError] = []

        def worker() -> None:
            try:
                controller.check_checkpoint("blocked_worker")
            except ExecutionCancelledError as exc:
                caught_error.append(exc)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        time.sleep(0.05)

        # Cancel while paused
        controller.cancel("Terminated during pause")
        t.join(timeout=1.0)

        self.assertEqual(len(caught_error), 1)
        self.assertIn("Terminated during pause", str(caught_error[0]))

    def test_abort_recovery(self) -> None:
        """Verify recovery abortion flags and events."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        controller = ExecutionController(event_bus=bus, execution_id="e_abort")
        self.assertFalse(controller.is_recovery_aborted)

        controller.abort_recovery("Exceeded max retry threshold")
        self.assertTrue(controller.is_recovery_aborted)
        self.assertEqual(controller.recovery_abort_reason, "Exceeded max retry threshold")

        event_types = [type(e) for e in events]
        self.assertIn(RecoveryAborted, event_types)

    def test_concurrent_control_thread_safety(self) -> None:
        """Verify thread-safety of controller under concurrent operations."""
        controller = ExecutionController()
        errors: List[Exception] = []

        def toggler() -> None:
            try:
                for i in range(50):
                    if i % 2 == 0:
                        controller.pause(f"pause {i}")
                    else:
                        controller.resume(f"resume {i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggler) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, "No exceptions should occur during concurrent toggling.")

    def test_executor_sync_cancellation(self) -> None:
        """Verify that cancelling during sync execution marks remaining tasks as CANCELLED."""
        bus = PlannerEventBus()
        ctrl = ExecutionController(event_bus=bus, plan_id="p_exec_cancel")

        plan = Plan(query="test_cancel")
        t1 = Task(action="step_one", target="first")
        t2 = Task(action="step_two", target="second", dependencies=[t1.id])
        plan.tasks.append(t1)
        plan.tasks.append(t2)

        exec_instance = Executor(planner_event_bus=bus)

        def handle_step_one(t: Task) -> str:
            ctrl.cancel("Cancelled mid-execution")
            return "step_one_done"

        def handle_step_two(t: Task) -> str:
            return "step_two_done"

        exec_instance.register_handler("step_one", handle_step_one)
        exec_instance.register_handler("step_two", handle_step_two)

        result = exec_instance.execute_plan(plan, controller=ctrl)

        self.assertFalse(result.success)
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.CANCELLED)
        self.assertEqual(len(result.completed_tasks), 1)
        self.assertIn(t2, result.skipped_tasks)

    def test_executor_async_cancellation(self) -> None:
        """Verify that cancelling during async execution cleanly terminates DAG."""
        async def run_async_test() -> None:
            bus = PlannerEventBus()
            ctrl = ExecutionController(event_bus=bus, plan_id="p_exec_async_cancel")

            plan = Plan(query="test_async_cancel")
            t1 = Task(action="async_step_one")
            t2 = Task(action="async_step_two", dependencies=[t1.id])
            plan.tasks.append(t1)
            plan.tasks.append(t2)

            exec_instance = Executor(planner_event_bus=bus)

            async def handle_one(t: Task) -> str:
                ctrl.cancel("Cancelled in async step one")
                return "async_one_done"

            async def handle_two(t: Task) -> str:
                return "async_two_done"

            exec_instance.register_handler("async_step_one", handle_one)
            exec_instance.register_handler("async_step_two", handle_two)

            result = await exec_instance.execute_plan_async(plan, controller=ctrl)

            self.assertFalse(result.success)
            self.assertEqual(t1.status, TaskStatus.COMPLETED)
            self.assertEqual(t2.status, TaskStatus.CANCELLED)
            self.assertEqual(len(result.completed_tasks), 1)
            self.assertIn(t2, result.skipped_tasks)

        asyncio.run(run_async_test())

    def test_executor_pause_and_resume(self) -> None:
        """Verify executor pauses before next task and resumes when instructed."""
        bus = PlannerEventBus()
        ctrl = ExecutionController(event_bus=bus, plan_id="p_exec_pause")

        plan = Plan(query="test_pause")
        t1 = Task(action="pause_step_one")
        t2 = Task(action="pause_step_two", dependencies=[t1.id])
        plan.tasks.append(t1)
        plan.tasks.append(t2)

        exec_instance = Executor(planner_event_bus=bus)

        def handle_one(t: Task) -> str:
            # Trigger pause immediately upon step one finishing
            ctrl.pause("Paused between steps")
            # Schedule a background thread to resume after a short delay
            def resume_worker() -> None:
                time.sleep(0.05)
                ctrl.resume("Resuming after pause")

            threading.Thread(target=resume_worker, daemon=True).start()
            return "one_ok"

        def handle_two(t: Task) -> str:
            return "two_ok"

        exec_instance.register_handler("pause_step_one", handle_one)
        exec_instance.register_handler("pause_step_two", handle_two)

        result = exec_instance.execute_plan(plan, controller=ctrl)

        self.assertTrue(result.success)
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(len(result.completed_tasks), 2)

    def test_executor_timeout_policy_skip(self) -> None:
        """Verify executor with TimeoutPolicy.SKIP skips timed out task and continues independent tasks."""
        plan = Plan(query="test_timeout_skip")
        t1 = Task(action="timeout_slow")
        t2 = Task(action="timeout_fast")  # Independent
        plan.tasks.append(t1)
        plan.tasks.append(t2)

        exec_instance = Executor()

        def handle_slow(t: Task) -> str:
            time.sleep(0.1)
            return "slow_done"

        def handle_fast(t: Task) -> str:
            return "fast_done"

        exec_instance.register_handler("timeout_slow", handle_slow)
        exec_instance.register_handler("timeout_fast", handle_fast)

        timeout_config = TimeoutConfig(task_timeout=0.03, policy=TimeoutPolicy.SKIP)
        result = exec_instance.execute_plan(plan, timeout_config=timeout_config)

        self.assertEqual(t1.status, TaskStatus.SKIPPED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(len(result.completed_tasks), 1)

    def test_executor_timeout_policy_abort(self) -> None:
        """Verify executor with TimeoutPolicy.ABORT cancels remaining execution on timeout."""
        ctrl = ExecutionController()
        plan = Plan(query="test_timeout_abort")
        t1 = Task(action="abort_slow")
        t2 = Task(action="abort_fast", dependencies=[t1.id])
        plan.tasks.append(t1)
        plan.tasks.append(t2)

        exec_instance = Executor()

        def handle_slow(t: Task) -> str:
            time.sleep(0.1)
            return "slow_done"

        def handle_fast(t: Task) -> str:
            return "fast_done"

        exec_instance.register_handler("abort_slow", handle_slow)
        exec_instance.register_handler("abort_fast", handle_fast)

        timeout_config = TimeoutConfig(task_timeout=0.03, policy=TimeoutPolicy.ABORT)
        result = exec_instance.execute_plan(plan, controller=ctrl, timeout_config=timeout_config)

        self.assertFalse(result.success)
        self.assertTrue(ctrl.is_cancelled)
        self.assertEqual(t2.status, TaskStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
