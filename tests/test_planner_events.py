"""Unit tests for the Planner Event System (Phase 17.0).

Verifies:
- Immutable dataclass behavior (FrozenInstanceError on mutation)
- Schema versioning
- Sequence ordering & deterministic monotonicity
- Typed subscriptions vs. wildcard subscriptions
- Unsubscription mechanics
- Asynchronous publishing
- Exception isolation (subscriber failure does not crash publisher)
- Thread safety across concurrent threads
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import threading
import time
import unittest
from typing import List

from app.ai.planner.events import (
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


class TestPlannerEvents(unittest.TestCase):
    """Test suite for strongly-typed immutable events and PlannerEventBus."""

    def test_event_immutability(self) -> None:
        """Planner events must be immutable (@dataclass(frozen=True))."""
        event = PlanStarted(plan_id="p1", query="test query", task_count=2)
        self.assertEqual(event.plan_id, "p1")
        self.assertEqual(event.schema_version, 1)

        with self.assertRaises(FrozenInstanceError):
            event.plan_id = "p2"  # type: ignore

    def test_event_to_dict_serialization(self) -> None:
        """Event to_dict() must return a serializable dictionary containing core and subclass fields."""
        event = TaskCompleted(
            execution_id="e1",
            plan_id="p1",
            task_id="t1",
            action="web_search",
            result="Found weather",
            duration=0.123,
        )
        d = event.to_dict()
        self.assertEqual(d["event_type"], "TaskCompleted")
        self.assertEqual(d["execution_id"], "e1")
        self.assertEqual(d["task_id"], "t1")
        self.assertEqual(d["action"], "web_search")
        self.assertEqual(d["duration"], 0.123)
        self.assertEqual(d["schema_version"], 1)

    def test_bus_sequence_id_monotonicity(self) -> None:
        """PlannerEventBus must assign monotonically increasing sequence IDs."""
        bus = PlannerEventBus()
        captured_seqs: List[int] = []

        bus.subscribe(PlannerEvent, lambda e: captured_seqs.append(e.sequence_id))

        e1 = PlanStarted(plan_id="p1")
        e2 = TaskStarted(task_id="t1")
        e3 = TaskCompleted(task_id="t1", result="OK")

        bus.publish(e1)
        bus.publish(e2)
        bus.publish(e3)

        self.assertEqual(captured_seqs, [1, 2, 3])

    def test_typed_vs_wildcard_subscriptions(self) -> None:
        """Typed subscribers must only receive their target event; wildcard receives all."""
        bus = PlannerEventBus()
        all_events: List[PlannerEvent] = []
        task_completed_events: List[TaskCompleted] = []

        bus.subscribe(PlannerEvent, lambda e: all_events.append(e))
        bus.subscribe(TaskCompleted, lambda e: task_completed_events.append(e))

        bus.publish(PlanStarted(plan_id="p1"))
        bus.publish(TaskStarted(task_id="t1"))
        bus.publish(TaskCompleted(task_id="t1", result="done"))
        bus.publish(PlanCompleted(plan_id="p1", success=True))

        self.assertEqual(len(all_events), 4)
        self.assertEqual(len(task_completed_events), 1)
        self.assertEqual(task_completed_events[0].task_id, "t1")

    def test_unsubscription(self) -> None:
        """Unsubscribe callable and method must stop handler invocation."""
        bus = PlannerEventBus()
        calls: List[str] = []

        def handler(e: PlanStarted) -> None:
            calls.append(e.plan_id)

        unsub = bus.subscribe(PlanStarted, handler)
        bus.publish(PlanStarted(plan_id="1"))
        self.assertEqual(calls, ["1"])

        # Unsubscribe via returned callable
        removed = unsub()
        self.assertTrue(removed)
        bus.publish(PlanStarted(plan_id="2"))
        self.assertEqual(calls, ["1"])

    def test_exception_isolation(self) -> None:
        """Failing subscribers must not disrupt publisher or other subscribers."""
        bus = PlannerEventBus()
        good_called: List[str] = []

        def faulty_subscriber(e: PlannerEvent) -> None:
            raise RuntimeError("Subscriber explosion!")

        def healthy_subscriber(e: PlannerEvent) -> None:
            good_called.append(e.execution_id)

        bus.subscribe(PlannerEvent, faulty_subscriber)
        bus.subscribe(PlannerEvent, healthy_subscriber)

        # Must not raise RuntimeError
        bus.publish(PlanStarted(execution_id="exec-123"))
        self.assertEqual(good_called, ["exec-123"])

    def test_async_publishing(self) -> None:
        """publish_async must properly invoke async coroutine handlers."""
        bus = PlannerEventBus()
        async_called: List[str] = []

        async def async_handler(e: TaskStarted) -> None:
            await asyncio.sleep(0.01)
            async_called.append(e.task_id)

        bus.subscribe(TaskStarted, async_handler)

        async def run_test() -> None:
            await bus.publish_async(TaskStarted(task_id="async-task-1"))
            await bus.publish_async(TaskStarted(task_id="async-task-2"))

        asyncio.run(run_test())
        self.assertEqual(async_called, ["async-task-1", "async-task-2"])

    def test_thread_safety_concurrent_publish(self) -> None:
        """Concurrent publishes across multiple threads must maintain unique, monotonic sequence IDs."""
        bus = PlannerEventBus()
        captured_seqs: List[int] = []
        lock = threading.Lock()

        def on_event(e: PlannerEvent) -> None:
            with lock:
                captured_seqs.append(e.sequence_id)

        bus.subscribe(PlannerEvent, on_event)

        threads: List[threading.Thread] = []
        events_per_thread = 50
        thread_count = 10

        def worker(tid: int) -> None:
            for i in range(events_per_thread):
                bus.publish(TaskStarted(task_id=f"t-{tid}-{i}"))

        for t_idx in range(thread_count):
            t = threading.Thread(target=worker, args=(t_idx,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(captured_seqs), thread_count * events_per_thread)
        # Sequence IDs must be unique and span from 1 to total_events
        self.assertEqual(len(set(captured_seqs)), thread_count * events_per_thread)
        self.assertEqual(min(captured_seqs), 1)
        self.assertEqual(max(captured_seqs), thread_count * events_per_thread)


if __name__ == "__main__":
    unittest.main()
