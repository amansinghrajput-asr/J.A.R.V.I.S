"""Unit tests for the J.A.R.V.I.S Production Conversational AI Skill.

Verifies:
- Skill discovery via SkillManager
- Direct registration and container/bus binding
- can_handle() evaluation across strings, dicts, Intents, and empty inputs
- Synchronous execute() invoking AIManager.generate(query=text)
- Asynchronous execute_async() invoking await AIManager.generate_async(query=text)
- Mocked AIManager resolution via ServiceContainer and explicit injection
- Lifecycle event emissions: ai.skill.started, ai.skill.completed, ai.skill.failed
- Execution priority (-100) acting as fallback behind higher-priority skills
- Thread-safe concurrent execution under multi-threaded load
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.models import AIResponse
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.router.intent import Intent
from app.skills.ai_skill import (
    AISkill,
    DEFAULT_AI_SKILL_DESCRIPTION,
    DEFAULT_AI_SKILL_NAME,
    DEFAULT_AI_SKILL_PRIORITY,
    ai_skill,
)
from app.skills.base import BaseSkill, SkillExecutionError
from app.skills.manager import SkillManager


class HigherPrioritySkill(BaseSkill):
    """Test helper skill with higher priority."""

    name = "specific_calc"
    description = "Handles specific calculator queries"
    priority = 100

    def can_handle(self, command: any) -> bool:
        if isinstance(command, str) and command.startswith("calculate"):
            return True
        return False

    def execute(self, command: any) -> str:
        return "calc_result"


class TestAISkill(unittest.TestCase):
    """Test suite for AISkill specification, execution, and integration."""

    def setUp(self) -> None:
        """Initialize clean service container, event bus, and mock AIManager."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()

        self.mock_ai_manager = MagicMock()
        self.mock_response = AIResponse(
            content="Yes, Sir. Systems are online.",
            model="gemini-2.5-flash",
            total_tokens=42,
            duration=0.15,
        )
        self.mock_ai_manager.generate.return_value = self.mock_response
        self.mock_ai_manager.generate_async = AsyncMock(return_value=self.mock_response)

        # Register mock AIManager into test container
        self.container.register_singleton("ai_manager", self.mock_ai_manager)

        # Instantiate skill with container and event bus
        self.skill = AISkill(
            container=self.container,
            event_bus=self.event_bus,
        )

    def test_default_metadata_and_attributes(self) -> None:
        """Verify AISkill default metadata conforms to specification."""
        self.assertEqual(self.skill.name, DEFAULT_AI_SKILL_NAME)
        self.assertEqual(self.skill.name, "ai")
        self.assertEqual(self.skill.description, DEFAULT_AI_SKILL_DESCRIPTION)
        self.assertEqual(self.skill.priority, DEFAULT_AI_SKILL_PRIORITY)
        self.assertEqual(self.skill.priority, -100)
        self.assertTrue(self.skill.enabled)
        self.assertIn("ai", self.skill.tags)
        self.assertIn("fallback", self.skill.tags)

        meta = self.skill.metadata()
        self.assertEqual(meta["name"], "ai")
        self.assertEqual(meta["priority"], -100)
        self.assertTrue(meta["enabled"])

    def test_singleton_instance_export(self) -> None:
        """Verify module-level ai_skill instance is accessible."""
        self.assertIsInstance(ai_skill, AISkill)
        self.assertEqual(ai_skill.name, "ai")

    def test_resolve_ai_manager_from_container(self) -> None:
        """Verify ai_manager is dynamically resolved from ServiceContainer."""
        self.assertIs(self.skill.ai_manager, self.mock_ai_manager)

    def test_resolve_ai_manager_via_ai_alias(self) -> None:
        """Verify ai_manager is resolved from container 'ai' alias if 'ai_manager' is absent."""
        alt_container = ServiceContainer()
        alt_mgr = MagicMock()
        alt_container.register_singleton("ai", alt_mgr)

        skill = AISkill(container=alt_container)
        self.assertIs(skill.ai_manager, alt_mgr)

    def test_explicit_ai_manager_injection_and_binding(self) -> None:
        """Verify explicit AIManager injection overrides container resolution."""
        explicit_mgr = MagicMock()
        skill = AISkill(ai_manager_instance=explicit_mgr, container=self.container)
        self.assertIs(skill.ai_manager, explicit_mgr)

        # Re-bind test
        new_mgr = MagicMock()
        skill.bind(ai_manager=new_mgr)
        self.assertIs(skill.ai_manager, new_mgr)

    def test_can_handle_strings(self) -> None:
        """Verify can_handle evaluates string commands accurately."""
        self.assertTrue(self.skill.can_handle("What is the capital of France?"))
        self.assertTrue(self.skill.can_handle("Hello Jarvis"))
        self.assertFalse(self.skill.can_handle(""))
        self.assertFalse(self.skill.can_handle("    "))
        self.assertFalse(self.skill.can_handle(None))

    def test_can_handle_dictionaries(self) -> None:
        """Verify can_handle extracts text from query/command/text/raw_command dictionary keys."""
        self.assertTrue(self.skill.can_handle({"query": "Tell me a joke"}))
        self.assertTrue(self.skill.can_handle({"command": "Status report"}))
        self.assertTrue(self.skill.can_handle({"text": "Hello"}))
        self.assertTrue(self.skill.can_handle({"raw_command": "Diagnostics"}))
        self.assertFalse(self.skill.can_handle({}))
        self.assertFalse(self.skill.can_handle({"other": 123}))

    def test_can_handle_intent_objects(self) -> None:
        """Verify can_handle evaluates Intent dataclass instances."""
        intent = Intent(raw_command="Search for recent AI news")
        self.assertTrue(self.skill.can_handle(intent))

        intent_empty = Intent(raw_command="")
        self.assertFalse(self.skill.can_handle(intent_empty))

    def test_execute_sync_calls_generate(self) -> None:
        """Verify execute() calls ai_manager.generate(query=text) and returns response."""
        result = self.skill.execute("What is quantum computing?")

        self.mock_ai_manager.generate.assert_called_once_with(query="What is quantum computing?")
        self.assertEqual(result, self.mock_response)
        self.assertEqual(result.content, "Yes, Sir. Systems are online.")
        self.assertEqual(str(result), "Yes, Sir. Systems are online.")
        self.assertEqual(self.skill.execution_count, 1)

    def test_execute_sync_with_conversation_context(self) -> None:
        """Verify execute() forwards conversation_id if specified."""
        intent = Intent(
            raw_command="Remember my name",
            context={"conversation_id": "session_abc"},
        )
        self.skill.execute(intent)

        self.mock_ai_manager.generate.assert_called_once_with(
            query="Remember my name",
            conversation_id="session_abc",
        )

    def test_execute_async_calls_generate_async(self) -> None:
        """Verify execute_async() awaits ai_manager.generate_async(query=text)."""
        async def _run() -> any:
            return await self.skill.execute_async("Explain general relativity")

        result = asyncio.run(_run())

        self.mock_ai_manager.generate_async.assert_awaited_once_with(
            query="Explain general relativity"
        )
        self.assertEqual(result, self.mock_response)
        self.assertEqual(self.skill.execution_count, 1)

    def test_execute_async_with_conversation_context(self) -> None:
        """Verify execute_async() forwards conversation_id if specified."""
        async def _run() -> any:
            return await self.skill.execute_async({
                "query": "Follow up question",
                "conversation_id": "conv_999",
            })

        asyncio.run(_run())

        self.mock_ai_manager.generate_async.assert_awaited_once_with(
            query="Follow up question",
            conversation_id="conv_999",
        )

    def test_sync_event_publication_success(self) -> None:
        """Verify ai.skill.started and ai.skill.completed events are published synchronously."""
        started_events: list[Event] = []
        completed_events: list[Event] = []

        self.event_bus.subscribe("ai.skill.started", lambda e: started_events.append(e))
        self.event_bus.subscribe("ai.skill.completed", lambda e: completed_events.append(e))

        self.skill.execute("Hello Jarvis")

        # Verify ai.skill.started
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].payload["query"], "Hello Jarvis")
        self.assertEqual(started_events[0].payload["skill_name"], "ai")

        # Verify ai.skill.completed
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].payload["query"], "Hello Jarvis")
        self.assertEqual(completed_events[0].payload["response"], "Yes, Sir. Systems are online.")
        self.assertEqual(completed_events[0].payload["result"], self.mock_response)
        self.assertGreater(completed_events[0].payload["duration"], 0.0)

    def test_async_event_publication_success(self) -> None:
        """Verify ai.skill.started and ai.skill.completed events are published asynchronously."""
        started_events: list[Event] = []
        completed_events: list[Event] = []

        self.event_bus.subscribe("ai.skill.started", lambda e: started_events.append(e))
        self.event_bus.subscribe("ai.skill.completed", lambda e: completed_events.append(e))

        async def _run() -> any:
            return await self.skill.execute_async("Async test query")

        asyncio.run(_run())

        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].payload["query"], "Async test query")
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].payload["response"], "Yes, Sir. Systems are online.")

    def test_sync_event_publication_failure(self) -> None:
        """Verify ai.skill.failed event is published and SkillExecutionError raised on error."""
        self.mock_ai_manager.generate.side_effect = RuntimeError("API service unavailable")
        failed_events: list[Event] = []
        self.event_bus.subscribe("ai.skill.failed", lambda e: failed_events.append(e))

        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute("Crash query")

        self.assertIn("API service unavailable", str(ctx.exception))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["query"], "Crash query")
        self.assertEqual(failed_events[0].payload["error"], "API service unavailable")
        self.assertEqual(failed_events[0].payload["exception_type"], "RuntimeError")
        self.assertEqual(self.skill.error_count, 1)

    def test_async_event_publication_failure(self) -> None:
        """Verify ai.skill.failed event is published and SkillExecutionError raised on async error."""
        self.mock_ai_manager.generate_async.side_effect = TimeoutError("Request timed out")
        failed_events: list[Event] = []
        self.event_bus.subscribe("ai.skill.failed", lambda e: failed_events.append(e))

        async def _run() -> any:
            return await self.skill.execute_async("Async crash query")

        with self.assertRaises(SkillExecutionError) as ctx:
            asyncio.run(_run())

        self.assertIn("Request timed out", str(ctx.exception))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["query"], "Async crash query")
        self.assertEqual(failed_events[0].payload["exception_type"], "TimeoutError")
        self.assertEqual(self.skill.error_count, 1)

    def test_execute_empty_query_raises_skill_execution_error(self) -> None:
        """Verify execute() raises SkillExecutionError for blank query input."""
        with self.assertRaises(SkillExecutionError):
            self.skill.execute("")

        with self.assertRaises(SkillExecutionError):
            self.skill.execute("   ")

    def test_skill_manager_discovery(self) -> None:
        """Verify AISkill is discoverable by SkillManager from 'app.skills'."""
        mgr = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        discovered = mgr.discover("app.skills")
        self.assertIn("ai", discovered)

        skill = mgr.get("ai")
        self.assertIsInstance(skill, AISkill)
        self.assertEqual(skill.priority, -100)

    def test_skill_manager_registration(self) -> None:
        """Verify manual registration of AISkill into SkillManager."""
        mgr = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        mgr.register(self.skill)

        self.assertTrue(mgr.has_skill("ai"))
        self.assertIs(mgr.get("ai"), self.skill)

    def test_priority_fallback_behavior_in_skill_manager(self) -> None:
        """Verify AISkill (-100) acts as fallback behind higher priority skills (100)."""
        mgr = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        high_skill = HigherPrioritySkill()
        mgr.register(high_skill)
        mgr.register(self.skill)

        # 1. Specific calculator query goes to higher priority skill
        res1 = mgr.execute("calculate 2 + 2")
        self.assertEqual(res1, "calc_result")
        self.mock_ai_manager.generate.assert_not_called()

        # 2. General conversational query falls back to AISkill
        res2 = mgr.execute("Who was the first person on the moon?")
        self.assertEqual(res2, self.mock_response)
        self.mock_ai_manager.generate.assert_called_once_with(
            query="Who was the first person on the moon?"
        )

    def test_thread_safety_under_concurrent_executions(self) -> None:
        """Verify AISkill maintains thread safety under parallel worker execution."""
        total_workers = 12
        queries = [f"Concurrent test query {i}" for i in range(total_workers)]

        def worker(q: str) -> any:
            return self.skill.execute(q)

        results: list[any] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(worker, q) for q in queries]
            for f in as_completed(futures):
                results.append(f.result())

        self.assertEqual(len(results), total_workers)
        for res in results:
            self.assertEqual(res, self.mock_response)
        self.assertEqual(self.skill.execution_count, total_workers)
        self.assertEqual(self.mock_ai_manager.generate.call_count, total_workers)


if __name__ == "__main__":
    unittest.main()
