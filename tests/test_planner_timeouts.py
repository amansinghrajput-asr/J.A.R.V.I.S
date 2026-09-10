"""Unit Tests for Timeout Management and Policies (Sprint 17.4).

Verifies sync and async task timeouts, policy enforcement (RETRY, SKIP, ABORT, ESCALATE),
event publication, and overall plan deadline checks.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any, List

from app.ai.planner.control import ExecutionController
from app.ai.planner.events import PlannerEvent, PlannerEventBus, TaskRetried, TaskTimeout
from app.ai.planner.timeouts import (
    TaskTimeoutError,
    TimeoutConfig,
    TimeoutManager,
    TimeoutPolicy,
)


class TestPlannerTimeouts(unittest.TestCase):
    """Test suite for planner timeout management."""

    def test_timeout_config_resolution(self) -> None:
        """Verify per-action override resolution and dictionary serialization."""
        config = TimeoutConfig(
            task_timeout=1.0,
            policy=TimeoutPolicy.RETRY,
            max_retries=2,
            action_timeouts={"web_search": 5.0},
        )
        self.assertEqual(config.get_task_timeout("open_app"), 1.0)
        self.assertEqual(config.get_task_timeout("web_search"), 5.0)

        data = config.to_dict()
        self.assertEqual(data["policy"], "RETRY")
        self.assertEqual(data["max_retries"], 2)

        reloaded = TimeoutConfig.from_dict(data)
        self.assertEqual(reloaded.policy, TimeoutPolicy.RETRY)
        self.assertEqual(reloaded.get_task_timeout("web_search"), 5.0)

    def test_sync_timeout_success(self) -> None:
        """Verify task completes normally when running well within timeout."""
        config = TimeoutConfig(task_timeout=1.0)
        mgr = TimeoutManager(config)

        res = mgr.run_sync("t1", "calculate", lambda: 42)
        self.assertEqual(res, 42)

    def test_sync_timeout_breach_and_event(self) -> None:
        """Verify sync task timeout raises TaskTimeoutError and emits TaskTimeout."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        config = TimeoutConfig(task_timeout=0.05, policy=TimeoutPolicy.SKIP)
        mgr = TimeoutManager(config, event_bus=bus)

        def slow_func() -> str:
            time.sleep(0.15)
            return "done"

        with self.assertRaises(TaskTimeoutError) as ctx:
            mgr.run_sync("t_slow", "web_search", slow_func, plan_id="p1")

        self.assertEqual(ctx.exception.task_id, "t_slow")
        self.assertEqual(ctx.exception.policy, TimeoutPolicy.SKIP)

        timeout_events = [e for e in events if isinstance(e, TaskTimeout)]
        self.assertEqual(len(timeout_events), 1)
        self.assertEqual(timeout_events[0].task_id, "t_slow")
        self.assertEqual(timeout_events[0].policy, "SKIP")

    def test_async_timeout_breach_and_event(self) -> None:
        """Verify async task timeout with asyncio.wait_for."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        config = TimeoutConfig(task_timeout=0.05, policy=TimeoutPolicy.SKIP)
        mgr = TimeoutManager(config, event_bus=bus)

        async def slow_coro() -> str:
            await asyncio.sleep(0.15)
            return "async done"

        async def test_runner() -> None:
            with self.assertRaises(TaskTimeoutError):
                await mgr.run_async("t_async_slow", "calculate", slow_coro, plan_id="p_async")

        asyncio.run(test_runner())

        timeout_events = [e for e in events if isinstance(e, TaskTimeout)]
        self.assertEqual(len(timeout_events), 1)
        self.assertEqual(timeout_events[0].task_id, "t_async_slow")

    def test_retry_policy_recovers(self) -> None:
        """Verify RETRY policy retries and succeeds if subsequent attempt is fast."""
        bus = PlannerEventBus()
        events: List[Any] = []
        bus.subscribe(PlannerEvent, events.append)

        config = TimeoutConfig(task_timeout=0.05, policy=TimeoutPolicy.RETRY, max_retries=2)
        mgr = TimeoutManager(config, event_bus=bus)

        attempt_count = 0

        def flaky_func() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                time.sleep(0.12)
            return "succeeded on attempt 2"

        res = mgr.run_sync("t_flaky", "fetch", flaky_func)
        self.assertEqual(res, "succeeded on attempt 2")
        self.assertEqual(attempt_count, 2)

        retried_events = [e for e in events if isinstance(e, TaskRetried)]
        self.assertEqual(len(retried_events), 1)
        self.assertEqual(retried_events[0].attempt, 2)

    def test_abort_policy_triggers_cancellation(self) -> None:
        """Verify ABORT policy cancels attached execution controller."""
        controller = ExecutionController(plan_id="p_abort")
        config = TimeoutConfig(task_timeout=0.05, policy=TimeoutPolicy.ABORT)
        mgr = TimeoutManager(config, controller=controller)

        def stalling_func() -> None:
            time.sleep(0.12)

        with self.assertRaises(TaskTimeoutError):
            mgr.run_sync("t_abort", "heavy_op", stalling_func)

        self.assertTrue(controller.is_cancelled)
        self.assertIn("ABORT policy", str(controller.cancellation_reason))

    def test_escalate_policy_raises_timeout_error(self) -> None:
        """Verify ESCALATE policy raises standard TimeoutError for outer recovery handling."""
        config = TimeoutConfig(task_timeout=0.05, policy=TimeoutPolicy.ESCALATE)
        mgr = TimeoutManager(config)

        def slow_func() -> None:
            time.sleep(0.12)

        with self.assertRaises(TimeoutError) as ctx:
            mgr.run_sync("t_esc", "model_call", slow_func)

        self.assertIn("escalated", str(ctx.exception))

    def test_total_execution_timeout(self) -> None:
        """Verify overall plan execution total deadline check."""
        controller = ExecutionController(plan_id="p_total")
        config = TimeoutConfig(total_timeout=0.10)
        mgr = TimeoutManager(config, controller=controller)

        start = time.perf_counter()
        time.sleep(0.15)

        with self.assertRaises(TimeoutError):
            mgr.check_total_timeout(start, plan_id="p_total")

        self.assertTrue(controller.is_cancelled)


if __name__ == "__main__":
    unittest.main()
