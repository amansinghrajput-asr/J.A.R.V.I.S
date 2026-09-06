"""Unit tests for the J.A.R.V.I.S Event Bus Subsystem."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.event_bus import (
    Event,
    EventBus,
    EventPriority,
    clear,
    event_bus,
    listener_count,
    publish,
    publish_async,
    subscribe,
    unsubscribe,
)


class TestEventModel(unittest.TestCase):
    """Test suite for Event dataclass validation and initialization."""

    def test_event_creation_with_defaults(self) -> None:
        """Verify Event initializes defaults correctly."""
        before = time.time()
        evt = Event(name="SYSTEM_INITIALIZED")
        after = time.time()

        self.assertEqual(evt.name, "SYSTEM_INITIALIZED")
        self.assertEqual(evt.payload, {})
        self.assertGreaterEqual(evt.timestamp, before)
        self.assertLessEqual(evt.timestamp, after)
        self.assertEqual(evt.source, "system")

    def test_event_creation_with_explicit_fields(self) -> None:
        """Verify Event handles explicit payload, source, and timestamp."""
        custom_time = 1700000000.0
        evt = Event(
            name="VOICE_COMMAND",
            payload={"text": "open terminal", "confidence": 0.95},
            timestamp=custom_time,
            source="perception.stt",
        )

        self.assertEqual(evt.name, "VOICE_COMMAND")
        self.assertEqual(evt.payload["text"], "open terminal")
        self.assertEqual(evt.payload["confidence"], 0.95)
        self.assertEqual(evt.timestamp, custom_time)
        self.assertEqual(evt.source, "perception.stt")

    def test_event_name_validation(self) -> None:
        """Verify invalid event names raise ValueError."""
        for invalid_name in ["", "   ", None, 123]:  # type: ignore[list-item]
            with self.assertRaises(ValueError):
                Event(name=invalid_name)  # type: ignore[arg-type]


class TestEventBusOperations(unittest.TestCase):
    """Test suite for EventBus subscription, prioritization, and dispatching."""

    def setUp(self) -> None:
        """Create fresh isolated EventBus instance for each test."""
        self.bus = EventBus()
        clear()

    def tearDown(self) -> None:
        """Clean up state after each test."""
        self.bus.clear()
        clear()

    def test_subscribe_and_publish_synchronous(self) -> None:
        """Verify single listener receives published event synchronously."""
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        self.bus.subscribe("TEST_EVENT", handler)
        self.assertEqual(self.bus.listener_count("TEST_EVENT"), 1)

        published = self.bus.publish("TEST_EVENT", payload={"data": 42}, source="test")

        self.assertEqual(len(received_events), 1)
        self.assertIs(received_events[0], published)
        self.assertEqual(received_events[0].payload["data"], 42)
        self.assertEqual(received_events[0].source, "test")

    def test_subscribe_multiple_listeners(self) -> None:
        """Verify multiple listeners receive the same published event."""
        results: list[str] = []

        def listener_one(evt: Event) -> None:
            results.append("one")

        def listener_two(evt: Event) -> None:
            results.append("two")

        self.bus.subscribe("MULTI_EVENT", listener_one)
        self.bus.subscribe("MULTI_EVENT", listener_two)

        self.assertEqual(self.bus.listener_count("MULTI_EVENT"), 2)

        self.bus.publish("MULTI_EVENT")

        self.assertEqual(results, ["one", "two"])

    def test_subscribe_decorator_syntax(self) -> None:
        """Verify subscription using decorator syntax."""
        calls: list[str] = []

        @self.bus.subscribe("DECORATOR_EVENT")
        def decorated_handler(evt: Event) -> None:
            calls.append(evt.name)

        self.assertEqual(self.bus.listener_count("DECORATOR_EVENT"), 1)

        self.bus.publish("DECORATOR_EVENT")
        self.assertEqual(calls, ["DECORATOR_EVENT"])

    def test_subscribe_listener_with_no_parameters(self) -> None:
        """Verify listeners requiring zero arguments are invoked cleanly."""
        invoked = []

        def no_arg_handler() -> None:
            invoked.append(True)

        self.bus.subscribe("NO_ARG_EVENT", no_arg_handler)
        self.bus.publish("NO_ARG_EVENT")

        self.assertTrue(invoked)

    def test_unsubscribe_via_method(self) -> None:
        """Verify unregistering a listener removes it from future events."""
        received = []

        def handler(evt: Event) -> None:
            received.append(evt)

        self.bus.subscribe("UNSUB_EVENT", handler)
        self.assertEqual(self.bus.listener_count("UNSUB_EVENT"), 1)

        self.bus.publish("UNSUB_EVENT")
        self.assertEqual(len(received), 1)

        removed = self.bus.unsubscribe("UNSUB_EVENT", handler)
        self.assertTrue(removed)
        self.assertEqual(self.bus.listener_count("UNSUB_EVENT"), 0)

        self.bus.publish("UNSUB_EVENT")
        self.assertEqual(len(received), 1)

        # Unsubscribing again returns False
        self.assertFalse(self.bus.unsubscribe("UNSUB_EVENT", handler))

    def test_unsubscribe_via_returned_callback(self) -> None:
        """Verify unsubscription using returned closure callback."""
        received = []

        def handler(evt: Event) -> None:
            received.append(evt)

        unsub = self.bus.subscribe("CLOSURE_EVENT", handler)
        self.assertEqual(self.bus.listener_count("CLOSURE_EVENT"), 1)

        self.bus.publish("CLOSURE_EVENT")
        self.assertEqual(len(received), 1)

        self.assertTrue(unsub())
        self.assertEqual(self.bus.listener_count("CLOSURE_EVENT"), 0)

        self.bus.publish("CLOSURE_EVENT")
        self.assertEqual(len(received), 1)

    def test_event_priorities_execution_order(self) -> None:
        """Verify listeners execute strictly according to priority descending."""
        execution_log: list[str] = []

        def low_listener(evt: Event) -> None:
            execution_log.append("low")

        def normal_listener(evt: Event) -> None:
            execution_log.append("normal")

        def high_listener(evt: Event) -> None:
            execution_log.append("high")

        def critical_listener(evt: Event) -> None:
            execution_log.append("critical")

        # Register in arbitrary order
        self.bus.subscribe("PRIORITY_EVENT", low_listener, priority=EventPriority.LOW)
        self.bus.subscribe("PRIORITY_EVENT", critical_listener, priority=EventPriority.CRITICAL)
        self.bus.subscribe("PRIORITY_EVENT", normal_listener, priority=EventPriority.NORMAL)
        self.bus.subscribe("PRIORITY_EVENT", high_listener, priority=EventPriority.HIGH)

        self.bus.publish("PRIORITY_EVENT")

        # Critical (200) -> High (100) -> Normal (50) -> Low (10)
        self.assertEqual(execution_log, ["critical", "high", "normal", "low"])

    def test_fifo_ordering_for_equal_priorities(self) -> None:
        """Verify listeners with identical priorities execute in registration FIFO order."""
        log: list[int] = []

        for i in range(5):
            self.bus.subscribe(
                "FIFO_EVENT",
                lambda evt, idx=i: log.append(idx),
                priority=EventPriority.NORMAL,
            )

        self.bus.publish("FIFO_EVENT")
        self.assertEqual(log, [0, 1, 2, 3, 4])

    def test_wildcard_subscription(self) -> None:
        """Verify wildcard '*' listener receives all events in priority order."""
        wildcard_events: list[str] = []
        specific_events: list[str] = []

        self.bus.subscribe(
            "*",
            lambda evt: wildcard_events.append(evt.name),
            priority=EventPriority.HIGH,
        )
        self.bus.subscribe(
            "EVENT_A",
            lambda evt: specific_events.append(evt.name),
            priority=EventPriority.NORMAL,
        )

        self.bus.publish("EVENT_A")
        self.bus.publish("EVENT_B")

        self.assertEqual(wildcard_events, ["EVENT_A", "EVENT_B"])
        self.assertEqual(specific_events, ["EVENT_A"])

    def test_error_isolation_synchronous(self) -> None:
        """Verify a failing listener does not prevent remaining listeners from running."""
        executed: list[str] = []

        def first_ok(evt: Event) -> None:
            executed.append("first")

        def faulty_listener(evt: Event) -> None:
            raise RuntimeError("Intentional listener crash!")

        def second_ok(evt: Event) -> None:
            executed.append("second")

        self.bus.subscribe("ISOLATION_TEST", first_ok, priority=EventPriority.HIGH)
        self.bus.subscribe("ISOLATION_TEST", faulty_listener, priority=EventPriority.NORMAL)
        self.bus.subscribe("ISOLATION_TEST", second_ok, priority=EventPriority.LOW)

        # Should not raise exception
        self.bus.publish("ISOLATION_TEST")

        self.assertEqual(executed, ["first", "second"])

    def test_clear_specific_event(self) -> None:
        """Verify clear() for a single event topic."""
        self.bus.subscribe("EVENT_1", lambda evt: None)
        self.bus.subscribe("EVENT_2", lambda evt: None)

        self.assertEqual(self.bus.listener_count(), 2)

        self.bus.clear("EVENT_1")

        self.assertEqual(self.bus.listener_count("EVENT_1"), 0)
        self.assertEqual(self.bus.listener_count("EVENT_2"), 1)
        self.assertEqual(self.bus.listener_count(), 1)

    def test_clear_all_events(self) -> None:
        """Verify clear() without arguments removes all listeners."""
        self.bus.subscribe("EVENT_1", lambda evt: None)
        self.bus.subscribe("EVENT_2", lambda evt: None)
        self.bus.subscribe("*", lambda evt: None)

        self.assertEqual(self.bus.listener_count(), 3)

        self.bus.clear()

        self.assertEqual(self.bus.listener_count(), 0)
        self.assertEqual(len(self.bus), 0)
        self.assertFalse(self.bus.has_listeners("EVENT_1"))

    def test_listener_count_and_helpers(self) -> None:
        """Verify listener_count, has_listeners, and dunder methods."""
        self.assertFalse(self.bus.has_listeners("TOPIC"))
        self.assertNotIn("TOPIC", self.bus)

        self.bus.subscribe("TOPIC", lambda evt: None)

        self.assertTrue(self.bus.has_listeners("TOPIC"))
        self.assertIn("TOPIC", self.bus)
        self.assertEqual(self.bus.listener_count("TOPIC"), 1)
        self.assertEqual(self.bus.listener_count(), 1)
        self.assertEqual(len(self.bus), 1)

        self.assertEqual(self.bus.registered_events(), ["TOPIC"])
        self.assertIn("EventBus", repr(self.bus))


class TestAsyncEventBusOperations(unittest.IsolatedAsyncioTestCase):
    """Test suite for asynchronous publishing and coroutine listeners."""

    def setUp(self) -> None:
        """Prepare isolated bus."""
        self.bus = EventBus()

    def tearDown(self) -> None:
        """Clean up."""
        self.bus.clear()

    async def test_publish_async_with_async_listener(self) -> None:
        """Verify async listeners are awaited during publish_async."""
        executed: list[str] = []

        async def async_handler(evt: Event) -> None:
            await asyncio.sleep(0.01)
            executed.append(f"async:{evt.payload['key']}")

        self.bus.subscribe("ASYNC_EVENT", async_handler)

        published = await self.bus.publish_async(
            "ASYNC_EVENT", payload={"key": "test_async"}
        )

        self.assertEqual(executed, ["async:test_async"])
        self.assertEqual(published.name, "ASYNC_EVENT")

    async def test_publish_async_mixed_sync_and_async_listeners(self) -> None:
        """Verify publish_async handles mixed sync and async listeners in priority order."""
        log: list[str] = []

        async def async_critical(evt: Event) -> None:
            await asyncio.sleep(0.005)
            log.append("async_critical")

        def sync_normal(evt: Event) -> None:
            log.append("sync_normal")

        async def async_low(evt: Event) -> None:
            await asyncio.sleep(0.005)
            log.append("async_low")

        self.bus.subscribe("MIXED_EVENT", sync_normal, priority=EventPriority.NORMAL)
        self.bus.subscribe("MIXED_EVENT", async_critical, priority=EventPriority.CRITICAL)
        self.bus.subscribe("MIXED_EVENT", async_low, priority=EventPriority.LOW)

        await self.bus.publish_async("MIXED_EVENT")

        self.assertEqual(log, ["async_critical", "sync_normal", "async_low"])

    async def test_error_isolation_async(self) -> None:
        """Verify publish_async isolates errors in async listeners."""
        executed: list[str] = []

        async def failing_async(evt: Event) -> None:
            await asyncio.sleep(0.005)
            raise ValueError("Async explosion!")

        async def succeeding_async(evt: Event) -> None:
            await asyncio.sleep(0.005)
            executed.append("success")

        self.bus.subscribe("ASYNC_FAULT", failing_async, priority=EventPriority.HIGH)
        self.bus.subscribe("ASYNC_FAULT", succeeding_async, priority=EventPriority.LOW)

        # Should not raise exception
        await self.bus.publish_async("ASYNC_FAULT")

        self.assertEqual(executed, ["success"])


class TestGlobalEventBusSingleton(unittest.TestCase):
    """Test suite for global event_bus singleton and module functions."""

    def setUp(self) -> None:
        """Ensure global singleton is clean."""
        clear()

    def tearDown(self) -> None:
        """Reset global singleton."""
        clear()

    def test_global_singleton_functions(self) -> None:
        """Verify module-level publish, subscribe, unsubscribe, and clear."""
        received: list[str] = []

        def handler(evt: Event) -> None:
            received.append(evt.name)

        subscribe("GLOBAL_EVENT", handler)
        self.assertEqual(listener_count("GLOBAL_EVENT"), 1)

        publish("GLOBAL_EVENT")
        self.assertEqual(received, ["GLOBAL_EVENT"])

        unsubscribe("GLOBAL_EVENT", handler)
        self.assertEqual(listener_count("GLOBAL_EVENT"), 0)

    def test_global_singleton_instance(self) -> None:
        """Verify event_bus is an instance of EventBus."""
        self.assertIsInstance(event_bus, EventBus)


class TestEventBusThreadSafety(unittest.TestCase):
    """Stress tests verifying concurrency and thread-safety of EventBus."""

    def setUp(self) -> None:
        """Create fresh EventBus."""
        self.bus = EventBus()

    def tearDown(self) -> None:
        """Clean up."""
        self.bus.clear()

    def test_concurrent_publishing(self) -> None:
        """Verify concurrent publishing from multiple threads without data loss."""
        received_count = 0
        lock = threading.Lock()

        def counter_listener(evt: Event) -> None:
            nonlocal received_count
            with lock:
                received_count += 1

        self.bus.subscribe("CONCURRENT_EVENT", counter_listener)

        num_threads = 20
        events_per_thread = 50

        def publish_worker(thread_idx: int) -> None:
            for i in range(events_per_thread):
                self.bus.publish(
                    "CONCURRENT_EVENT",
                    payload={"thread": thread_idx, "idx": i},
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(publish_worker, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        expected = num_threads * events_per_thread
        self.assertEqual(received_count, expected)

    def test_concurrent_subscribe_unsubscribe_and_publish(self) -> None:
        """Verify dynamic subscribe/unsubscribe during concurrent publishing causes no deadlocks."""
        num_workers = 16
        stop_flag = threading.Event()

        def dynamic_subscriber(worker_id: int) -> None:
            while not stop_flag.is_set():
                def dummy_handler(evt: Event) -> None:
                    pass

                topic = f"TOPIC_{worker_id % 4}"
                unsub = self.bus.subscribe(topic, dummy_handler)
                time.sleep(0.001)
                unsub()

        def dynamic_publisher(worker_id: int) -> None:
            for i in range(50):
                topic = f"TOPIC_{worker_id % 4}"
                self.bus.publish(topic, payload={"val": i})
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            sub_futures = [
                executor.submit(dynamic_subscriber, i) for i in range(num_workers // 2)
            ]
            pub_futures = [
                executor.submit(dynamic_publisher, i)
                for i in range(num_workers // 2, num_workers)
            ]

            # Wait for all publishers to finish
            for f in as_completed(pub_futures):
                f.result()

            # Signal subscribers to stop
            stop_flag.set()
            for f in as_completed(sub_futures):
                f.result()


if __name__ == "__main__":
    unittest.main()
