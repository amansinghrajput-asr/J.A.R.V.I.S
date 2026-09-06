"""Comprehensive unit tests for the J.A.R.V.I.S Memory Subsystem."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.memory.models import (
    DEFAULT_CONVERSATION_ID,
    DEFAULT_MEMORY_CAPACITY,
    VALID_ROLES,
    ConversationMemory,
    InvalidMemoryError,
    MemoryCapacityError,
    MemoryError,
)
from app.memory.store import MemoryStore
from app.memory.manager import MemoryManager, memory_manager


class TestConversationMemory(unittest.TestCase):
    """Unit tests for the ConversationMemory dataclass."""

    def test_memory_initialization_defaults(self) -> None:
        mem = ConversationMemory(content="Hello Jarvis")
        self.assertEqual(mem.content, "Hello Jarvis")
        self.assertEqual(mem.role, "user")
        self.assertEqual(mem.conversation_id, DEFAULT_CONVERSATION_ID)
        self.assertEqual(mem.source, "text")
        self.assertEqual(mem.importance, 1)
        self.assertIsInstance(mem.timestamp, float)
        self.assertGreater(mem.timestamp, 0)
        self.assertEqual(mem.metadata, {})
        self.assertEqual(mem.tags, [])
        # Verify valid UUID string
        self.assertTrue(uuid.UUID(mem.id))

    def test_memory_custom_attributes(self) -> None:
        custom_id = str(uuid.uuid4())
        mem = ConversationMemory(
            content="I am JARVIS, your AI assistant.",
            role="assistant",
            id=custom_id,
            conversation_id="session_42",
            source="voice",
            importance=8,
            timestamp=12345.67,
            metadata={"intent_id": "test-123", "model": "gemini-2.0-flash"},
            tags=["greeting", "persona"],
        )
        self.assertEqual(mem.id, custom_id)
        self.assertEqual(mem.role, "assistant")
        self.assertEqual(mem.conversation_id, "session_42")
        self.assertEqual(mem.source, "voice")
        self.assertEqual(mem.importance, 8)
        self.assertEqual(mem.timestamp, 12345.67)
        self.assertEqual(mem.metadata["model"], "gemini-2.0-flash")
        self.assertEqual(mem.tags, ["greeting", "persona"])

    def test_memory_importance_clamping(self) -> None:
        mem_high = ConversationMemory(content="High", importance=15)
        self.assertEqual(mem_high.importance, 10)

        mem_low = ConversationMemory(content="Low", importance=-2)
        self.assertEqual(mem_low.importance, 1)

        mem_invalid = ConversationMemory(content="Invalid", importance="invalid")  # type: ignore
        self.assertEqual(mem_invalid.importance, 1)

    def test_memory_invalid_empty_content(self) -> None:
        with self.assertRaises(InvalidMemoryError):
            ConversationMemory(content="")

        with self.assertRaises(InvalidMemoryError):
            ConversationMemory(content="   ")

        with self.assertRaises(InvalidMemoryError):
            ConversationMemory(content=None)  # type: ignore

    def test_memory_role_normalization_and_fallback(self) -> None:
        mem1 = ConversationMemory(content="test", role="ASSISTANT")
        self.assertEqual(mem1.role, "assistant")

        mem2 = ConversationMemory(content="test", role="SYSTEM")
        self.assertEqual(mem2.role, "system")

        # Invalid role falls back to 'user'
        mem3 = ConversationMemory(content="test", role="unknown_alien_role")
        self.assertEqual(mem3.role, "user")

    def test_memory_to_dict_and_from_dict(self) -> None:
        original = ConversationMemory(
            content="What is my schedule?",
            role="user",
            conversation_id="conv_abc",
            source="voice",
            importance=7,
            metadata={"source_device": "headset"},
            tags=["calendar", "query"],
        )
        data = original.to_dict()
        self.assertEqual(data["id"], original.id)
        self.assertEqual(data["content"], "What is my schedule?")
        self.assertEqual(data["role"], "user")
        self.assertEqual(data["conversation_id"], "conv_abc")
        self.assertEqual(data["source"], "voice")
        self.assertEqual(data["importance"], 7)
        self.assertEqual(data["metadata"], {"source_device": "headset"})
        self.assertEqual(data["tags"], ["calendar", "query"])

        restored = ConversationMemory.from_dict(data)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.role, original.role)
        self.assertEqual(restored.conversation_id, original.conversation_id)
        self.assertEqual(restored.source, original.source)
        self.assertEqual(restored.importance, original.importance)
        self.assertEqual(restored.metadata, original.metadata)
        self.assertEqual(restored.tags, original.tags)

    def test_memory_json_serialization(self) -> None:
        original = ConversationMemory(
            content="Serialized to JSON string",
            role="assistant",
            conversation_id="session_99",
            source="automation",
            importance=5,
            metadata={"status": "active"},
            tags=["json", "test"],
        )
        json_str = original.to_json()
        self.assertIsInstance(json_str, str)

        restored = ConversationMemory.from_json(json_str)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.conversation_id, "session_99")
        self.assertEqual(restored.source, "automation")
        self.assertEqual(restored.importance, 5)

    def test_memory_from_dict_invalid(self) -> None:
        with self.assertRaises(TypeError):
            ConversationMemory.from_dict("not a dict")  # type: ignore

        with self.assertRaises(InvalidMemoryError):
            ConversationMemory.from_dict({"role": "user"})

    def test_memory_matches(self) -> None:
        mem = ConversationMemory(
            content="Turn on the living room lights",
            role="user",
            conversation_id="home_auto",
            metadata={"device_type": "smart_plug"},
            tags=["home", "iot"],
        )
        self.assertTrue(mem.matches(query="living room"))
        self.assertTrue(mem.matches(query="LIGHTS"))
        self.assertTrue(mem.matches(query="home"))
        self.assertTrue(mem.matches(role="user"))
        self.assertTrue(mem.matches(conversation_id="home_auto"))
        self.assertTrue(mem.matches(metadata_filter={"device_type": "smart_plug"}))

        # Negative checks
        self.assertFalse(mem.matches(query="kitchen"))
        self.assertFalse(mem.matches(role="assistant"))
        self.assertFalse(mem.matches(conversation_id="other_session"))
        self.assertFalse(mem.matches(metadata_filter={"device_type": "camera"}))

    def test_memory_repr(self) -> None:
        mem = ConversationMemory(content="Short note")
        self.assertIn("Short note", repr(mem))


class TestMemoryStore(unittest.TestCase):
    """Unit tests for the MemoryStore class."""

    def test_store_initialization(self) -> None:
        store = MemoryStore(capacity=20)
        self.assertEqual(store.capacity, 20)
        self.assertEqual(len(store), 0)
        self.assertIsNone(store.get_last())

    def test_store_invalid_capacity(self) -> None:
        with self.assertRaises(MemoryCapacityError):
            MemoryStore(capacity=0)

        with self.assertRaises(MemoryCapacityError):
            MemoryStore(capacity=-5)

        with self.assertRaises(MemoryCapacityError):
            MemoryStore(capacity="invalid")  # type: ignore

    def test_store_add_and_get(self) -> None:
        store = MemoryStore(capacity=10)
        mem = ConversationMemory(content="Note 1")
        stored = store.add(mem)

        self.assertEqual(len(store), 1)
        self.assertIs(stored, mem)
        self.assertIs(store.get(mem.id), mem)
        self.assertTrue(store.exists(mem.id))
        self.assertIn(mem.id, store)

    def test_store_get_last(self) -> None:
        store = MemoryStore(capacity=10)
        self.assertIsNone(store.get_last())

        mem1 = ConversationMemory(content="First")
        mem2 = ConversationMemory(content="Second")
        store.add(mem1)
        self.assertEqual(store.get_last().content, "First")

        store.add(mem2)
        self.assertEqual(store.get_last().content, "Second")

    def test_store_exists_and_remove(self) -> None:
        store = MemoryStore(capacity=10)
        mem = ConversationMemory(content="Deletable")
        store.add(mem)

        self.assertTrue(store.exists(mem.id))
        self.assertTrue(store.remove(mem.id))
        self.assertFalse(store.exists(mem.id))
        self.assertFalse(store.remove(mem.id))
        self.assertEqual(len(store), 0)

        # Non-existent or invalid id
        self.assertFalse(store.exists("non-existent"))
        self.assertFalse(store.remove("non-existent"))
        self.assertFalse(store.remove(""))

    def test_store_add_invalid(self) -> None:
        store = MemoryStore(capacity=10)
        with self.assertRaises(InvalidMemoryError):
            store.add("not a ConversationMemory")  # type: ignore

    def test_store_capacity_eviction(self) -> None:
        capacity = 3
        store = MemoryStore(capacity=capacity)

        mem1 = ConversationMemory(content="Item 1")
        mem2 = ConversationMemory(content="Item 2")
        mem3 = ConversationMemory(content="Item 3")
        mem4 = ConversationMemory(content="Item 4")

        store.add(mem1)
        store.add(mem2)
        store.add(mem3)
        self.assertEqual(len(store), 3)

        # Adding 4th item evicts 1st (FIFO)
        store.add(mem4)
        self.assertEqual(len(store), 3)
        self.assertFalse(store.exists(mem1.id))
        self.assertTrue(store.exists(mem2.id))
        self.assertTrue(store.exists(mem3.id))
        self.assertTrue(store.exists(mem4.id))
        self.assertEqual(store.get_last().content, "Item 4")

    def test_store_dynamic_capacity_resize(self) -> None:
        store = MemoryStore(capacity=5)
        for i in range(5):
            store.add(ConversationMemory(content=f"Item {i}"))

        self.assertEqual(len(store), 5)

        # Expand capacity
        store.capacity = 10
        self.assertEqual(store.capacity, 10)
        self.assertEqual(len(store), 5)

        # Reduce capacity to 2 (should trim oldest 3)
        store.capacity = 2
        self.assertEqual(store.capacity, 2)
        self.assertEqual(len(store), 2)
        recent = store.all()
        self.assertEqual(recent[0].content, "Item 3")
        self.assertEqual(recent[1].content, "Item 4")

        with self.assertRaises(MemoryCapacityError):
            store.capacity = 0

    def test_store_get_recent_with_conversation_filter(self) -> None:
        store = MemoryStore(capacity=10)
        store.add(ConversationMemory(content="A1", conversation_id="conv_A"))
        store.add(ConversationMemory(content="B1", conversation_id="conv_B"))
        store.add(ConversationMemory(content="A2", conversation_id="conv_A"))

        conv_a = store.get_recent(conversation_id="conv_A")
        self.assertEqual(len(conv_a), 2)
        self.assertEqual(conv_a[0].content, "A1")
        self.assertEqual(conv_a[1].content, "A2")

        conv_b = store.get_recent(conversation_id="conv_B")
        self.assertEqual(len(conv_b), 1)
        self.assertEqual(conv_b[0].content, "B1")

    def test_store_search_comprehensive(self) -> None:
        store = MemoryStore(capacity=10)
        store.add(
            ConversationMemory(
                content="Play jazz music",
                role="user",
                conversation_id="music_session",
                metadata={"genre": "jazz"},
                tags=["media", "audio"],
            )
        )
        store.add(
            ConversationMemory(
                content="Playing miles davis",
                role="assistant",
                conversation_id="music_session",
                metadata={"genre": "jazz"},
                tags=["media", "playing"],
            )
        )
        store.add(
            ConversationMemory(
                content="Check weather",
                role="user",
                conversation_id="weather_session",
                metadata={"city": "Tokyo"},
                tags=["weather"],
            )
        )

        # Search by query
        res1 = store.search(query="jazz")
        self.assertEqual(len(res1), 1)

        # Search by role and metadata
        res2 = store.search(role="assistant", metadata={"genre": "jazz"})
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0].content, "Playing miles davis")

        # Search by conversation_id
        res3 = store.search(conversation_id="weather_session")
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0].content, "Check weather")

    def test_store_clear(self) -> None:
        store = MemoryStore(capacity=10)
        store.add(ConversationMemory(content="M1"))
        store.add(ConversationMemory(content="M2"))
        self.assertEqual(len(store), 2)

        cleared = store.clear()
        self.assertEqual(cleared, 2)
        self.assertEqual(len(store), 0)
        self.assertEqual(store.all(), [])


class TestMemoryManager(unittest.TestCase):
    """Unit tests for the MemoryManager coordinator."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.manager = MemoryManager(
            capacity=20,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        self.manager.clear()

    def test_manager_dependency_injection(self) -> None:
        self.assertIs(self.manager.container, self.container)
        self.assertIs(self.manager.event_bus, self.event_bus)
        self.assertIsNotNone(self.manager.logger)
        self.assertIsNotNone(self.manager.config)
        self.assertEqual(self.manager.capacity, 20)

    def test_manager_auto_registration_in_container(self) -> None:
        test_container = ServiceContainer()
        mgr = MemoryManager(
            container_instance=test_container,
            auto_register_in_container=True,
        )
        self.assertTrue(test_container.exists("memory_manager"))
        self.assertTrue(test_container.exists("memory"))
        self.assertIs(test_container.resolve("memory_manager"), mgr)
        self.assertIs(test_container.resolve("memory"), mgr)

    def test_global_singleton_instance(self) -> None:
        self.assertIsInstance(memory_manager, MemoryManager)

    def test_add_memory_with_conversation_id_source_importance(self) -> None:
        events = []
        self.event_bus.subscribe("memory.added", lambda e: events.append(e))

        mem = self.manager.add(
            content="Remember my favorite editor is VS Code",
            role="user",
            conversation_id="conv_pref",
            source="voice",
            importance=9,
            tags=["preference", "editor"],
        )
        self.assertEqual(self.manager.count(), 1)
        self.assertEqual(mem.content, "Remember my favorite editor is VS Code")
        self.assertEqual(mem.conversation_id, "conv_pref")
        self.assertEqual(mem.source, "voice")
        self.assertEqual(mem.importance, 9)

        # Verify event
        self.assertEqual(len(events), 1)
        payload = events[0].payload
        self.assertEqual(payload["memory"]["conversation_id"], "conv_pref")
        self.assertEqual(payload["memory"]["source"], "voice")
        self.assertEqual(payload["memory"]["importance"], 9)
        self.assertEqual(payload["total_count"], 1)

    def test_get_last(self) -> None:
        self.assertIsNone(self.manager.get_last())

        self.manager.add("Alpha")
        self.assertEqual(self.manager.get_last().content, "Alpha")

        self.manager.add("Beta")
        self.assertEqual(self.manager.get_last().content, "Beta")

    def test_exists_and_remove_event(self) -> None:
        remove_events = []
        self.event_bus.subscribe("memory.removed", lambda e: remove_events.append(e))

        mem = self.manager.add("To be deleted")
        self.assertTrue(self.manager.exists(mem.id))

        removed = self.manager.remove(mem.id)
        self.assertTrue(removed)
        self.assertFalse(self.manager.exists(mem.id))

        # Check memory.removed event
        self.assertEqual(len(remove_events), 1)
        self.assertEqual(remove_events[0].payload["memory_id"], mem.id)
        self.assertIn("timestamp", remove_events[0].payload)

    def test_get_recent_and_search_events(self) -> None:
        events = []
        self.event_bus.subscribe("memory.retrieved", lambda e: events.append(e))

        self.manager.add("Open browser", tags=["system"])
        self.manager.add("Play music", tags=["media"])

        # 1. get_recent
        recent = self.manager.get_recent(limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(events[-1].name, "memory.retrieved")
        self.assertEqual(events[-1].payload["count"], 2)

        # 2. search
        results = self.manager.search(query="browser")
        self.assertEqual(len(results), 1)
        self.assertEqual(events[-1].payload["query"], "browser")

    def test_clear_and_event(self) -> None:
        events = []
        self.event_bus.subscribe("memory.cleared", lambda e: events.append(e))

        self.manager.add("One")
        self.manager.add("Two")
        self.assertEqual(self.manager.count(), 2)

        cleared = self.manager.clear()
        self.assertEqual(cleared, 2)
        self.assertEqual(self.manager.count(), 0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["cleared_count"], 2)

    # --------------------------------------------------------------------------
    # Async Methods & Async Event Dispatching
    # --------------------------------------------------------------------------

    def test_async_operations(self) -> None:
        async def run_test() -> None:
            added_events = []
            removed_events = []
            retrieved_events = []
            cleared_events = []

            self.event_bus.subscribe("memory.added", lambda e: added_events.append(e.name))
            self.event_bus.subscribe("memory.removed", lambda e: removed_events.append(e.name))
            self.event_bus.subscribe("memory.retrieved", lambda e: retrieved_events.append(e.name))
            self.event_bus.subscribe("memory.cleared", lambda e: cleared_events.append(e.name))

            # 1. Async Add
            mem = await self.manager.add_async(
                "Async turn",
                role="assistant",
                conversation_id="async_conv",
                source="tool",
                importance=5,
            )
            self.assertEqual(mem.content, "Async turn")
            self.assertEqual(self.manager.count(), 1)
            self.assertIn("memory.added", added_events)

            # 2. Async Get Recent
            recent = await self.manager.get_recent_async(limit=10, conversation_id="async_conv")
            self.assertEqual(len(recent), 1)
            self.assertIn("memory.retrieved", retrieved_events)

            # 3. Async Search
            search_res = await self.manager.search_async(query="turn")
            self.assertEqual(len(search_res), 1)

            # 4. Async Remove
            removed = await self.manager.remove_async(mem.id)
            self.assertTrue(removed)
            self.assertIn("memory.removed", removed_events)

            # 5. Async Clear
            await self.manager.add_async("Temp item")
            cleared = await self.manager.clear_async()
            self.assertEqual(cleared, 1)
            self.assertIn("memory.cleared", cleared_events)

        asyncio.run(run_test())

    # --------------------------------------------------------------------------
    # Thread Safety & Concurrency
    # --------------------------------------------------------------------------

    def test_concurrent_memory_operations(self) -> None:
        total_workers = 30

        def write_worker(idx: int) -> None:
            mem = self.manager.add(f"Thread message {idx}", role="user")
            self.manager.exists(mem.id)
            self.manager.get_last()

        def read_worker() -> None:
            self.manager.get_recent(limit=5)
            self.manager.search(query="Thread")

        with ThreadPoolExecutor(max_workers=8) as executor:
            write_futures = [executor.submit(write_worker, i) for i in range(total_workers)]
            read_futures = [executor.submit(read_worker) for _ in range(total_workers)]
            for f in as_completed(write_futures + read_futures):
                f.result()

        self.assertEqual(self.manager.count(), min(total_workers, self.manager.capacity))


if __name__ == "__main__":
    unittest.main()
