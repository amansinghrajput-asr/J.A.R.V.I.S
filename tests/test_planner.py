"""Unit tests for the J.A.R.V.I.S Planner Foundation."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any, Optional

from app.ai.manager import AIManager
from app.ai.models import AIResponse
from app.ai.planner.executor import Executor, executor
from app.ai.planner.failure_classifier import FailureClassifier
from app.ai.planner.heuristics import RecoveryDecision, evaluate_recovery_viability
from app.ai.planner.memory import (
    ExecutionMemory,
    ExecutionMetrics,
    FailureCategory,
    TaskExecutionRecord,
)
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.models import ExecutionResult, Plan, PlanningStrategy, Task, TaskStatus
from app.ai.planner.planner import Planner, planner
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.memory.manager import MemoryManager


class TestPlannerModels(unittest.TestCase):
    """Test suite for Planner models and TaskStatus enumeration."""

    def test_task_status_enum(self) -> None:
        """Verify TaskStatus enumeration members."""
        self.assertEqual(TaskStatus.PENDING.value, "PENDING")
        self.assertEqual(TaskStatus.RUNNING.value, "RUNNING")
        self.assertEqual(TaskStatus.COMPLETED.value, "COMPLETED")
        self.assertEqual(TaskStatus.FAILED.value, "FAILED")

    def test_task_model_defaults_and_serialization(self) -> None:
        """Verify Task creation, default status, UUID generation, and dictionary export."""
        task = Task(action="open_app", target="chrome")
        self.assertEqual(task.action, "open_app")
        self.assertEqual(task.target, "chrome")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.parameters, {})
        self.assertTrue(task.id)

        d = task.to_dict()
        self.assertEqual(d["action"], "open_app")
        self.assertEqual(d["target"], "chrome")
        self.assertEqual(d["status"], "PENDING")
        self.assertEqual(d["id"], task.id)

    def test_plan_model_and_serialization(self) -> None:
        """Verify Plan structure, task list, and is_empty helper."""
        p_empty = Plan(query="test query")
        self.assertTrue(p_empty.is_empty())
        self.assertEqual(len(p_empty.tasks), 0)

        t1 = Task(action="open_app", target="chrome")
        t2 = Task(action="web_search", target="weather")
        p = Plan(query="open chrome and search weather", tasks=[t1, t2])
        self.assertFalse(p.is_empty())
        self.assertEqual(len(p.tasks), 2)

        d = p.to_dict()
        self.assertEqual(d["query"], "open chrome and search weather")
        self.assertEqual(len(d["tasks"]), 2)

    def test_execution_result_model(self) -> None:
        """Verify ExecutionResult structure and serialization."""
        t1 = Task(action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        res = ExecutionResult(success=True, completed_tasks=[t1], output="Done")
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(res.output, "Done")
        d = res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["output"], "Done")


class TestPlanner(unittest.TestCase):
    """Test suite for Planner rule-based task decomposition."""

    def setUp(self) -> None:
        self.planner = Planner()

    def test_empty_and_whitespace_query(self) -> None:
        """Verify empty and whitespace queries return an empty Plan without errors."""
        p1 = self.planner.create_plan("")
        self.assertTrue(p1.is_empty())
        self.assertEqual(p1.query, "")

        p2 = self.planner.create_plan("   ")
        self.assertTrue(p2.is_empty())

    def test_unknown_query(self) -> None:
        """Verify unknown or unclassifiable queries return an empty Plan without raising."""
        p = self.planner.create_plan("tell me a random bedtime story about dragons")
        self.assertTrue(p.is_empty())
        self.assertEqual(len(p.tasks), 0)

    def test_single_task_open_app(self) -> None:
        """Verify 'open chrome' maps to open_app task with target chrome."""
        p = self.planner.create_plan("open chrome")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "open_app")
        self.assertEqual(t.target, "chrome")

    def test_single_task_web_search(self) -> None:
        """Verify 'search weather' maps to web_search task with target weather."""
        p = self.planner.create_plan("search weather")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "web_search")
        self.assertEqual(t.target, "weather")

    def test_single_task_summarize_file(self) -> None:
        """Verify 'summarize pdf' maps to summarize_file task with target pdf."""
        p = self.planner.create_plan("summarize pdf")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "summarize_file")
        self.assertEqual(t.target, "pdf")

    def test_single_task_calculate(self) -> None:
        """Verify 'calculate 5+5' maps to calculate task."""
        p = self.planner.create_plan("calculate 5+5")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "calculate")
        self.assertEqual(t.target, "5+5")
        self.assertEqual(t.parameters.get("expression"), "5+5")

    def test_single_task_save_memory(self) -> None:
        """Verify 'remember my favorite color is blue' maps to save_memory task."""
        p = self.planner.create_plan("remember my favorite color is blue")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "save_memory")
        self.assertEqual(t.target, "my favorite color is blue")
        self.assertEqual(t.parameters.get("content"), "my favorite color is blue")

    def test_single_task_clear_memory(self) -> None:
        """Verify 'forget everything' maps to clear_memory task."""
        p = self.planner.create_plan("forget everything")
        self.assertEqual(len(p.tasks), 1)
        t = p.tasks[0]
        self.assertEqual(t.action, "clear_memory")

    def test_multiple_tasks_ordered_decomposition(self) -> None:
        """Verify composite queries produce ordered multiple tasks.

        Example:
        Input: 'Open Chrome and search ChatGPT'
        Output: Plan(tasks=[Task(action='open_app', target='chrome'), Task(action='web_search', target='ChatGPT')])
        """
        p = self.planner.create_plan("Open Chrome and search ChatGPT")
        self.assertEqual(len(p.tasks), 2)
        self.assertEqual(p.tasks[0].action, "open_app")
        self.assertEqual(p.tasks[0].target, "Chrome")
        self.assertEqual(p.tasks[1].action, "web_search")
        self.assertEqual(p.tasks[1].target, "ChatGPT")

    def test_multiple_tasks_with_various_conjunctions(self) -> None:
        """Verify queries with multiple conjunctions (comma, then, and)."""
        p = self.planner.create_plan("open chrome, then search weather, and calculate 5+5")
        self.assertEqual(len(p.tasks), 3)
        self.assertEqual(p.tasks[0].action, "open_app")
        self.assertEqual(p.tasks[0].target, "chrome")
        self.assertEqual(p.tasks[1].action, "web_search")
        self.assertEqual(p.tasks[1].target, "weather")
        self.assertEqual(p.tasks[2].action, "calculate")
        self.assertEqual(p.tasks[2].target, "5+5")

    def test_singleton_export(self) -> None:
        """Verify module-level planner singleton is available."""
        self.assertIsInstance(planner, Planner)


class TestExecutor(unittest.TestCase):
    """Test suite for Executor task execution engine (Phase 9)."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def test_empty_plan_execution(self) -> None:
        """Verify empty plan returns successful ExecutionResult immediately."""
        plan = Plan(query="empty query", tasks=[])
        res = self.executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 0)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertIn("no tasks", res.output.lower())

    def test_handler_registration_and_execution(self) -> None:
        """Verify registering an action handler executes successfully."""
        self.executor.register_handler("open_app", lambda t: f"Launched {t.target}")
        self.assertIsNotNone(self.executor.get_handler("open_app"))

        plan = Plan(query="open chrome", tasks=[Task(action="open_app", target="chrome")])
        res = self.executor.execute_plan(plan)

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(res.completed_tasks[0].status, TaskStatus.COMPLETED)
        self.assertIn("Launched chrome", res.output)

    def test_handler_unregister(self) -> None:
        """Verify unregistering an action handler."""
        self.executor.register_handler("calc", lambda t: "10")
        self.assertTrue(self.executor.unregister_handler("calc"))
        self.assertIsNone(self.executor.get_handler("calc"))
        self.assertFalse(self.executor.unregister_handler("non_existent"))

    def test_missing_handler_marks_failed(self) -> None:
        """Verify task with unhandled action transitions to FAILED without crashing."""
        plan = Plan(query="do magic", tasks=[Task(action="magic_wand", target="rabbit")])
        res = self.executor.execute_plan(plan)

        self.assertFalse(res.success)
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(res.failed_tasks[0].status, TaskStatus.FAILED)
        self.assertIn("No handler registered", res.output)

    def test_handler_raising_exception(self) -> None:
        """Verify exceptions during task execution mark task as FAILED."""
        def faulty_handler(t: Task) -> None:
            raise RuntimeError("Connection failed")

        self.executor.register_handler("bad_action", faulty_handler)
        plan = Plan(query="fail", tasks=[Task(action="bad_action", target="foo")])
        res = self.executor.execute_plan(plan)

        self.assertFalse(res.success)
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(res.failed_tasks[0].status, TaskStatus.FAILED)
        self.assertIn("Connection failed", res.output)

    def test_sequential_multi_task_execution(self) -> None:
        """Verify sequential execution of user multi-step plan."""
        execution_order = []

        def open_handler(t: Task) -> str:
            execution_order.append(f"open:{t.target}")
            return f"Opened {t.target}"

        def search_handler(t: Task) -> str:
            execution_order.append(f"search:{t.target}")
            return f"Searched {t.target}"

        self.executor.register_handler("open_app", open_handler)
        self.executor.register_handler("web_search", search_handler)

        t1 = Task(action="open_app", target="chrome")
        t2 = Task(action="web_search", target="ChatGPT")
        plan = Plan(query="Open Chrome and search ChatGPT", tasks=[t1, t2])

        res = self.executor.execute_plan(plan)

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 2)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(execution_order, ["open:chrome", "search:ChatGPT"])
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertIn("Opened chrome", res.output)
        self.assertIn("Searched ChatGPT", res.output)

    def test_async_plan_execution(self) -> None:
        """Verify execute_plan_async supports async coroutine handlers."""
        async def async_calc(t: Task) -> str:
            await asyncio.sleep(0.001)
            return f"Result of {t.target} is 10"

        self.executor.register_handler("calculate", async_calc)
        plan = Plan(query="calculate 5+5", tasks=[Task(action="calculate", target="5+5")])

        res = asyncio.run(self.executor.execute_plan_async(plan))

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(res.completed_tasks[0].status, TaskStatus.COMPLETED)
        self.assertIn("Result of 5+5 is 10", res.output)

    def test_container_dynamic_skill_resolution(self) -> None:
        """Verify Executor resolves actions dynamically from ServiceContainer skills."""
        class MockSkill:
            def execute(self, query: str) -> str:
                return f"MockSkill handled: {query}"

        self.container.register_singleton("web_search", MockSkill())
        plan = Plan(query="search weather", tasks=[Task(action="web_search", target="weather")])

        res = self.executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertIn("MockSkill handled", res.output)

    def test_executor_singleton_export(self) -> None:
        """Verify module-level executor singleton is available."""
        self.assertIsInstance(executor, Executor)


class TestExecutorSkillRegistryAndRouting(unittest.TestCase):
    """Test suite for Phase 11: Skill Registry & Automatic Action Routing."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()

    def test_skill_manager_injection_explicit(self) -> None:
        """Verify SkillManager can be explicitly injected into Executor."""
        mock_sm = object()
        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=mock_sm,
            auto_register_in_container=False,
        )
        self.assertIs(ex.skill_manager, mock_sm)

    def test_skill_manager_injection_from_container(self) -> None:
        """Verify SkillManager is resolved from ServiceContainer when not explicitly passed."""
        mock_sm = object()
        self.container.register_singleton("skill_manager", mock_sm, allow_override=True)
        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=None,
            auto_register_in_container=False,
        )
        self.assertIs(ex.skill_manager, mock_sm)

    def test_priority_1_registered_handler_overrides_all(self) -> None:
        """Verify Priority 1: Registered handler overrides SkillManager and ServiceContainer."""
        class MockSkill:
            def execute(self, cmd: str) -> str:
                return "from_skill_manager"

        class MockContainerService:
            def execute(self, cmd: str) -> str:
                return "from_container"

        class MockSkillManager:
            def get(self, name: str) -> Any:
                return MockSkill() if name == "test_action" else None

        mock_sm = MockSkillManager()
        self.container.register_singleton("test_action", MockContainerService())

        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=mock_sm,
            auto_register_in_container=False,
        )
        ex.register_handler("test_action", lambda t: "from_registered_handler")

        task = Task(action="test_action", target="foo")
        handler = ex.resolve_handler(task)
        self.assertIsNotNone(handler)
        self.assertEqual(handler(task), "from_registered_handler")

    def test_priority_2_skill_manager_overrides_container(self) -> None:
        """Verify Priority 2: SkillManager overrides ServiceContainer when no registered handler exists."""
        class MockSkill:
            def execute(self, cmd: str) -> str:
                return f"from_skill_manager:{cmd}"

        class MockContainerService:
            def execute(self, cmd: str) -> str:
                return "from_container"

        class MockSkillManager:
            def get(self, name: str) -> Any:
                return MockSkill() if name == "test_action" else None

        mock_sm = MockSkillManager()
        self.container.register_singleton("test_action", MockContainerService())

        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=mock_sm,
            auto_register_in_container=False,
        )

        task = Task(action="test_action", target="bar")
        handler = ex.resolve_handler(task)
        self.assertIsNotNone(handler)
        self.assertEqual(handler(task), "from_skill_manager:test_action bar")

    def test_priority_3_container_fallback(self) -> None:
        """Verify Priority 3: ServiceContainer is used when neither registered handler nor SkillManager has the action."""
        class MockContainerService:
            def execute(self, cmd: str) -> str:
                return "from_container"

        class MockSkillManager:
            def get(self, name: str) -> Any:
                return None

        mock_sm = MockSkillManager()
        self.container.register_singleton("container_only_action", MockContainerService())

        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=mock_sm,
            auto_register_in_container=False,
        )

        task = Task(action="container_only_action")
        handler = ex.resolve_handler(task)
        self.assertIsNotNone(handler)
        self.assertEqual(handler(task), "from_container")

    def test_priority_4_unknown_action_returns_none_and_marks_failed(self) -> None:
        """Verify Priority 4: Unknown action returns None from resolve_handler and FAILED in execute_plan."""
        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=None,
            auto_register_in_container=False,
        )
        task = Task(action="unknown_galaxy_action", target="warp9")
        self.assertIsNone(ex.resolve_handler(task))
        self.assertIsNone(ex.resolve_handler("unknown_galaxy_action"))

        plan = Plan(query="warp", tasks=[task])
        res = ex.execute_plan(plan)
        self.assertFalse(res.success)
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(res.failed_tasks[0].status, TaskStatus.FAILED)
        self.assertIn("No handler registered", res.output)

    def test_multiple_skills_routing_through_skill_manager(self) -> None:
        """Verify multiple skills in SkillManager resolve and execute in a composite plan."""
        class SystemSkillMock:
            def execute(self, cmd: str) -> str:
                return f"App launched: {cmd}"

        class WebSearchSkillMock:
            def execute(self, cmd: str) -> str:
                return f"Web search results for: {cmd}"

        class MockSkillManager:
            def __init__(self) -> None:
                self.skills = {
                    "system": SystemSkillMock(),
                    "web_search": WebSearchSkillMock(),
                }

            def get(self, name: str) -> Any:
                return self.skills.get(name)

        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=MockSkillManager(),
            auto_register_in_container=False,
        )

        t1 = Task(action="open_app", target="chrome")
        t2 = Task(action="web_search", target="ChatGPT")
        plan = Plan(query="Open Chrome and search ChatGPT", tasks=[t1, t2])

        res = ex.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 2)
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertIn("App launched: open chrome", res.output)
        self.assertIn("Web search results for: search ChatGPT", res.output)

    def test_async_skill_execution(self) -> None:
        """Verify async skills with execute_async are properly awaited in execute_plan_async."""
        class AsyncSkillMock:
            async def execute_async(self, cmd: str) -> str:
                await asyncio.sleep(0.001)
                return f"Async result for: {cmd}"

        class MockSkillManager:
            def get(self, name: str) -> Any:
                return AsyncSkillMock() if name == "async_action" else None

        ex = Executor(
            container_instance=self.container,
            skill_manager_instance=MockSkillManager(),
            auto_register_in_container=False,
        )

        task = Task(action="async_action", target="hello")
        plan = Plan(query="run async", tasks=[task])

        res = asyncio.run(ex.execute_plan_async(plan))
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(res.completed_tasks[0].status, TaskStatus.COMPLETED)
        self.assertIn("Async result for: async_action hello", res.output)


class _DummyProvider:
    """Mock provider recording calls for planner integration tests."""

    def __init__(self, reply: str = "LLM response") -> None:
        self.reply = reply
        self.model = "mock-model"
        self.call_count = 0

    def generate(self, payload: Any, config: Any = None) -> AIResponse:
        self.call_count += 1
        return AIResponse(content=self.reply, model=self.model)

    async def generate_async(self, payload: Any, config: Any = None) -> AIResponse:
        self.call_count += 1
        return AIResponse(content=self.reply, model=self.model)


class TestAIManagerPlannerIntegration(unittest.TestCase):
    """Test suite for Planner and Executor integration into AIManager (Phase 8 & 10)."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.memory = MemoryManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.provider = _DummyProvider()
        self.executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        # Register standard handlers on test executor
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: f"Searched {t.target}")
        self.executor.register_handler("calculate", lambda t: f"Calculated {t.target}")
        self.executor.register_handler("save_memory", lambda t: f"Saved {t.target}")
        self.executor.register_handler("clear_memory", lambda t: "Cleared")

    _SENTINEL = object()

    def _create_manager(
        self,
        planner_inst: Optional[Planner] = None,
        executor_inst: Any = _SENTINEL,
    ) -> AIManager:
        resolved_executor = self.executor if executor_inst is self._SENTINEL else executor_inst
        return AIManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            memory_manager_instance=self.memory,
            provider_instance=self.provider,
            planner_instance=planner_inst,
            executor_instance=resolved_executor,
            auto_register_in_container=False,
        )

    def test_planner_injection_explicit(self) -> None:
        """Verify Planner can be explicitly injected via constructor."""
        custom_planner = Planner()
        mgr = self._create_manager(planner_inst=custom_planner)
        self.assertIs(mgr.planner, custom_planner)

    def test_planner_injection_from_container(self) -> None:
        """Verify Planner is resolved from ServiceContainer if registered."""
        custom_planner = Planner()
        self.container.register_singleton("planner", custom_planner, allow_override=True)
        mgr = self._create_manager(planner_inst=None)
        self.assertIs(mgr.planner, custom_planner)

    def test_planner_injection_fallback_to_global_singleton(self) -> None:
        """Verify Planner falls back to global singleton when not injected or in container."""
        empty_container = ServiceContainer()
        mgr = AIManager(
            container_instance=empty_container,
            event_bus_instance=self.event_bus,
            memory_manager_instance=self.memory,
            provider_instance=self.provider,
            planner_instance=None,
            auto_register_in_container=False,
        )
        self.assertIs(mgr.planner, planner)

    def test_executor_injection_explicit(self) -> None:
        """Verify Executor can be explicitly injected via constructor."""
        custom_executor = Executor(container_instance=self.container, auto_register_in_container=False)
        mgr = self._create_manager(executor_inst=custom_executor)
        self.assertIs(mgr.executor, custom_executor)

    def test_executor_injection_from_container(self) -> None:
        """Verify Executor is resolved from ServiceContainer if registered."""
        custom_executor = Executor(container_instance=self.container, auto_register_in_container=False)
        self.container.register_singleton("executor", custom_executor, allow_override=True)
        mgr = self._create_manager(executor_inst=None)
        self.assertIs(mgr.executor, custom_executor)

    def test_executor_injection_fallback_to_global_singleton(self) -> None:
        """Verify Executor falls back to global singleton when not injected or in container."""
        empty_container = ServiceContainer()
        mgr = AIManager(
            container_instance=empty_container,
            event_bus_instance=self.event_bus,
            memory_manager_instance=self.memory,
            provider_instance=self.provider,
            executor_instance=None,
            auto_register_in_container=False,
        )
        self.assertIs(mgr.executor, executor)

    def test_last_plan_property_initial_and_updated(self) -> None:
        """Verify last_plan property is None initially and updated after generation."""
        mgr = self._create_manager()
        self.assertIsNone(mgr.last_plan)

        mgr.generate("open chrome")
        self.assertIsNotNone(mgr.last_plan)
        self.assertIsInstance(mgr.last_plan, Plan)
        self.assertEqual(len(mgr.last_plan.tasks), 1)
        self.assertEqual(mgr.last_plan.tasks[0].action, "open_app")

    def test_empty_plan_fallback(self) -> None:
        """Verify 0-task (empty) plan falls back to normal LLM flow without error."""
        mgr = self._create_manager()
        resp = mgr.generate("tell me an interesting fact about space")

        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(resp.content, "LLM response")
        self.assertIsNotNone(mgr.last_plan)
        self.assertTrue(mgr.last_plan.is_empty())
        self.assertEqual(len(mgr.last_plan.tasks), 0)

    def test_single_task_plan_behavior(self) -> None:
        """Verify 1-task plan proceeds normally through dispatch/LLM."""
        mgr = self._create_manager()
        resp = mgr.generate("open chrome")

        # 1-task plan does not return early multi-task plan message; it proceeds through normal dispatch/LLM
        self.assertEqual(self.provider.call_count, 1)
        self.assertIsNotNone(mgr.last_plan)
        self.assertEqual(len(mgr.last_plan.tasks), 1)
        self.assertEqual(mgr.last_plan.tasks[0].action, "open_app")
        self.assertEqual(mgr.last_plan.tasks[0].target, "chrome")

    def test_multi_task_plan_execution_sync(self) -> None:
        """Verify multi-task query executes all tasks through Executor and returns formatted AIResponse."""
        mgr = self._create_manager()
        query = "Open Chrome and search ChatGPT and calculate 5+5"
        resp = mgr.generate(query)

        # Provider must NOT have been called
        self.assertEqual(self.provider.call_count, 0)
        self.assertIsNotNone(mgr.last_plan)
        self.assertEqual(len(mgr.last_plan.tasks), 3)

        expected_response = (
            "Completed:\n"
            "✓ Open Chrome\n"
            "✓ Search ChatGPT\n"
            "✓ Calculate 5+5\n\n"
            "Overall:\n"
            "3 completed\n"
            "0 failed"
        )
        self.assertEqual(resp.content, expected_response)
        self.assertEqual(resp.model, "executor")

        # Check metadata
        self.assertIn("plan", resp.metadata)
        self.assertIn("execution_result", resp.metadata)
        self.assertEqual(resp.metadata["completed_count"], 3)
        self.assertEqual(resp.metadata["failed_count"], 0)
        self.assertIn("execution_duration", resp.metadata)

    def test_multi_task_plan_execution_async(self) -> None:
        """Verify multi-task query in generate_async executes tasks asynchronously through Executor."""
        mgr = self._create_manager()
        query = "open chrome, then search weather, and calculate 5+5"

        async def _test() -> AIResponse:
            return await mgr.generate_async(query)

        resp = asyncio.run(_test())

        self.assertEqual(self.provider.call_count, 0)
        self.assertIsNotNone(mgr.last_plan)
        self.assertEqual(len(mgr.last_plan.tasks), 3)
        self.assertIn("Completed:\n✓ Open Chrome\n✓ Search weather\n✓ Calculate 5+5", resp.content)
        self.assertIn("Overall:\n3 completed\n0 failed", resp.content)
        self.assertEqual(resp.model, "executor")
        self.assertEqual(resp.metadata["completed_count"], 3)
        self.assertEqual(resp.metadata["failed_count"], 0)

    def test_multi_task_partial_failure(self) -> None:
        """Verify multi-task plan with partial failures formats both Completed and Failed sections."""
        # Create an executor without handler for calculate
        failing_executor = Executor(container_instance=self.container, auto_register_in_container=False)
        failing_executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        failing_executor.register_handler("web_search", lambda t: f"Searched {t.target}")
        # calculate has no handler -> will fail

        mgr = self._create_manager(executor_inst=failing_executor)
        query = "open chrome and search ChatGPT and calculate 5+5"
        resp = mgr.generate(query)

        self.assertEqual(self.provider.call_count, 0)
        self.assertIn("Completed:\n✓ Open Chrome\n✓ Search ChatGPT", resp.content)
        self.assertIn("Failed:\n✗ Calculate 5+5", resp.content)
        self.assertIn("Overall:\n2 completed\n1 failed", resp.content)
        self.assertEqual(resp.metadata["completed_count"], 2)
        self.assertEqual(resp.metadata["failed_count"], 1)

    def test_executor_exception_fallback_to_pipeline(self) -> None:
        """Verify that unexpected Executor exceptions fall back to IntentRouter/LLM pipeline."""
        class CrashingExecutor:
            def execute_plan(self, plan: Plan) -> ExecutionResult:
                raise RuntimeError("Catastrophic hardware fault")

        crashing_executor = CrashingExecutor()
        mgr = self._create_manager(executor_inst=crashing_executor)  # type: ignore

        query = "open chrome and search ChatGPT"
        resp = mgr.generate(query)

        # Should NOT crash; should fall through to IntentRouter and then LLM
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(resp.content, "LLM response")


class MockPlanningProvider:
    """Mock LLM planning provider for unit testing."""

    def __init__(self, response_text: str = '{"tasks": []}') -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt: Optional[Any] = None
        self.model = "mock-planning-provider"

    def generate(self, payload: Any, config: Any = None) -> AIResponse:
        self.call_count += 1
        self.last_prompt = payload
        return AIResponse(content=self.response_text, model="mock-planning-provider")

    async def generate_async(self, payload: Any, config: Any = None) -> AIResponse:
        return self.generate(payload, config)


class TestHybridLLMPlanner(unittest.TestCase):
    """Test suite for Hybrid LLM Planner (Phase 13)."""

    def setUp(self) -> None:
        self.mock_provider = MockPlanningProvider()
        self.planner = Planner(
            provider_instance=self.mock_provider,
            strategy=PlanningStrategy.HYBRID,
        )

    def test_rule_planner_direct(self) -> None:
        """Verify deterministic rule planner operates without calling the LLM provider."""
        # Query with exact rule match
        plan = self.planner.create_plan("open chrome", strategy=PlanningStrategy.RULE_BASED)
        self.assertEqual(plan.strategy, PlanningStrategy.RULE_BASED)
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].action, "open_app")
        self.assertEqual(plan.tasks[0].target, "chrome")
        self.assertEqual(self.mock_provider.call_count, 0)

        # Unknown query under RULE_BASED strategy yields empty plan without calling LLM
        unknown_plan = self.planner.create_plan(
            "explain why quantum computing is revolutionary",
            strategy=PlanningStrategy.RULE_BASED,
        )
        self.assertTrue(unknown_plan.is_empty())
        self.assertEqual(unknown_plan.strategy, PlanningStrategy.RULE_BASED)
        self.assertEqual(self.mock_provider.call_count, 0)

    def test_llm_fallback(self) -> None:
        """Verify planner falls back to injected LLM provider when rules yield no tasks."""
        llm_output = json.dumps({
            "tasks": [
                {
                    "id": "task_1",
                    "action": "summarize_file",
                    "target": "earnings.pdf",
                    "parameters": {},
                    "dependencies": [],
                },
                {
                    "id": "task_2",
                    "action": "web_search",
                    "target": "competitor analysis",
                    "parameters": {},
                    "dependencies": ["task_1"],
                },
            ]
        })
        self.mock_provider.response_text = llm_output

        query = "review earnings report and perform competitor analysis"
        plan = self.planner.create_plan(query)

        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertEqual(plan.strategy, PlanningStrategy.LLM)
        self.assertEqual(len(plan.tasks), 2)
        self.assertEqual(plan.tasks[0].action, "summarize_file")
        self.assertEqual(plan.tasks[0].target, "earnings.pdf")
        self.assertEqual(plan.tasks[0].id, "task_1")
        self.assertEqual(plan.tasks[1].action, "web_search")
        self.assertEqual(plan.tasks[1].dependencies, ["task_1"])

    def test_invalid_json(self) -> None:
        """Verify malformed JSON from LLM is safely rejected and yields an empty plan."""
        self.mock_provider.response_text = "Here is your plan: [open_app -> chrome] {invalid json syntax}"

        plan = self.planner.create_plan("do something non-deterministic")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())
        self.assertEqual(plan.strategy, PlanningStrategy.LLM)

    def test_malformed_plan_missing_tasks(self) -> None:
        """Verify plan without 'tasks' key is rejected."""
        self.mock_provider.response_text = json.dumps({"result": "done", "status": "ok"})

        plan = self.planner.create_plan("do complex action")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())

    def test_malformed_plan_tasks_not_list(self) -> None:
        """Verify plan where 'tasks' is not a list is rejected."""
        self.mock_provider.response_text = json.dumps({"tasks": "open_app chrome"})

        plan = self.planner.create_plan("do complex action")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())

    def test_malformed_plan_missing_action(self) -> None:
        """Verify task missing the required 'action' field is rejected."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [{"target": "chrome"}]
        })

        plan = self.planner.create_plan("do complex action")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())

    def test_malformed_plan_duplicate_task_ids(self) -> None:
        """Verify plan with duplicate task IDs is rejected."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "open_app", "target": "chrome"},
                {"id": "t1", "action": "web_search", "target": "weather"},
            ]
        })

        plan = self.planner.create_plan("do complex action")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())

    def test_malformed_plan_invalid_dependencies(self) -> None:
        """Verify plan with invalid or forward dependencies is rejected."""
        # Non-existent dependency
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "open_app", "target": "chrome", "dependencies": ["nonexistent_task"]},
            ]
        })
        plan1 = self.planner.create_plan("do complex action 1")
        self.assertTrue(plan1.is_empty())

        # Forward dependency (t1 depends on t2 which comes after t1)
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "open_app", "target": "chrome", "dependencies": ["t2"]},
                {"id": "t2", "action": "web_search", "target": "weather", "dependencies": []},
            ]
        })
        plan2 = self.planner.create_plan("do complex action 2")
        self.assertTrue(plan2.is_empty())

    def test_unknown_actions(self) -> None:
        """Verify plan containing unknown or unsafe actions is rejected."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "drop_production_db", "target": "main"},
            ]
        })

        plan = self.planner.create_plan("cleanup the server database")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertTrue(plan.is_empty())

    def test_hybrid_success(self) -> None:
        """Verify full hybrid flow: simple rule query uses rules; complex query uses LLM."""
        # 1. Known deterministic command uses rules (no LLM call)
        rule_plan = self.planner.create_plan("open chrome and search ChatGPT")
        self.assertEqual(rule_plan.strategy, PlanningStrategy.RULE_BASED)
        self.assertEqual(len(rule_plan.tasks), 2)
        self.assertEqual(self.mock_provider.call_count, 0)

        # 2. Complex command falls back to LLM
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "calculate", "target": "100 * 42"},
            ]
        })
        llm_plan = self.planner.create_plan("forecast revenue growth based on market indicators")
        self.assertEqual(llm_plan.strategy, PlanningStrategy.LLM)
        self.assertEqual(len(llm_plan.tasks), 1)
        self.assertEqual(self.mock_provider.call_count, 1)

    def test_markdown_code_fences_stripped(self) -> None:
        """Verify JSON enclosed in markdown code fences is properly extracted and parsed."""
        fenced_json = (
            "```json\n"
            "{\n"
            '  "tasks": [\n'
            '    {"id": "task_1", "action": "open_app", "target": "terminal", "parameters": {}, "dependencies": []}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        self.mock_provider.response_text = fenced_json

        plan = self.planner.create_plan("prepare workstation environment for morning standup")
        self.assertEqual(self.mock_provider.call_count, 1)
        self.assertFalse(plan.is_empty())
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].action, "open_app")
        self.assertEqual(plan.tasks[0].target, "terminal")

    def test_planning_strategy_metadata(self) -> None:
        """Verify planning strategy metadata is preserved in Plan models and AIManager responses."""
        # 1. Plan model serialization includes strategy
        rule_plan = Plan(query="test", tasks=[], strategy=PlanningStrategy.RULE_BASED)
        d_rule = rule_plan.to_dict()
        self.assertEqual(d_rule["strategy"], "RULE_BASED")

        llm_plan = Plan(query="test", tasks=[], strategy=PlanningStrategy.LLM)
        d_llm = llm_plan.to_dict()
        self.assertEqual(d_llm["strategy"], "LLM")

        hybrid_plan = Plan(query="test", tasks=[], strategy=PlanningStrategy.HYBRID)
        d_hybrid = hybrid_plan.to_dict()
        self.assertEqual(d_hybrid["strategy"], "HYBRID")

        # 2. AIManager tracks last_plan strategy and exposes planning_strategy in metadata
        container = ServiceContainer()
        event_bus = EventBus()
        memory_mgr = MemoryManager()
        executor_inst = Executor(container_instance=container, auto_register_in_container=False)
        executor_inst.register_handler("open_app", lambda t: f"Opened {t.target}")
        executor_inst.register_handler("web_search", lambda t: f"Searched {t.target}")

        mgr = AIManager(
            container_instance=container,
            event_bus_instance=event_bus,
            memory_manager_instance=memory_mgr,
            planner_instance=self.planner,
            executor_instance=executor_inst,
            provider_instance=self.mock_provider,
            auto_register_in_container=False,
        )

        # Rule-based plan execution via AIManager
        resp_rule = mgr.generate("open chrome and search ChatGPT")
        self.assertIsNotNone(mgr.last_plan)
        self.assertEqual(mgr.last_plan.strategy, PlanningStrategy.RULE_BASED)
        self.assertIn("planning_strategy", resp_rule.metadata)
        self.assertEqual(resp_rule.metadata["planning_strategy"], "RULE_BASED")

        # LLM plan execution via AIManager
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "open_app", "target": "chrome"},
                {"id": "t2", "action": "web_search", "target": "quantum computing", "dependencies": ["t1"]},
            ]
        })
        resp_llm = mgr.generate("explore cutting edge quantum advancements")
        self.assertIsNotNone(mgr.last_plan)
        self.assertEqual(mgr.last_plan.strategy, PlanningStrategy.LLM)
        self.assertIn("planning_strategy", resp_llm.metadata)
        self.assertEqual(resp_llm.metadata["planning_strategy"], "LLM")

    def test_async_plan_creation(self) -> None:
        """Verify asynchronous plan creation with Planner."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "web_search", "target": "python async"},
            ]
        })
        plan = asyncio.run(self.planner.create_plan_async("retrieve benchmarks for modern web frameworks"))
        self.assertEqual(plan.strategy, PlanningStrategy.LLM)
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].action, "web_search")



class TestExecutorDAG(unittest.TestCase):
    """Test suite for Phase 14: Executor Dependency Scheduling (DAG Execution)."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def test_linear_dependency_chain(self) -> None:
        """Verify linear dependency chain: A -> B -> C."""
        execution_order = []

        def make_handler(name: str):
            def handler(t: Task) -> str:
                execution_order.append(name)
                return f"Executed {name}"
            return handler

        self.executor.register_handler("action_a", make_handler("A"))
        self.executor.register_handler("action_b", make_handler("B"))
        self.executor.register_handler("action_c", make_handler("C"))

        task_a = Task(id="task_a", action="action_a", target="target_a", dependencies=[])
        task_b = Task(id="task_b", action="action_b", target="target_b", dependencies=["task_a"])
        task_c = Task(id="task_c", action="action_c", target="target_c", dependencies=["task_b"])

        plan = Plan(query="linear chain", tasks=[task_a, task_b, task_c])
        res = self.executor.execute_plan(plan)

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 3)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(len(res.skipped_tasks), 0)
        self.assertEqual(execution_order, ["A", "B", "C"])
        self.assertEqual(res.execution_order, ["task_a", "task_b", "task_c"])

    def test_linear_dependency_chain_async(self) -> None:
        """Verify linear dependency chain with async execution."""
        execution_order = []

        def make_async_handler(name: str):
            async def handler(t: Task) -> str:
                await asyncio.sleep(0.001)
                execution_order.append(name)
                return f"Executed async {name}"
            return handler

        self.executor.register_handler("action_a", make_async_handler("A"))
        self.executor.register_handler("action_b", make_async_handler("B"))
        self.executor.register_handler("action_c", make_async_handler("C"))

        task_a = Task(id="task_a", action="action_a", target="target_a", dependencies=[])
        task_b = Task(id="task_b", action="action_b", target="target_b", dependencies=["task_a"])
        task_c = Task(id="task_c", action="action_c", target="target_c", dependencies=["task_b"])

        plan = Plan(query="async linear chain", tasks=[task_a, task_b, task_c])
        res = asyncio.run(self.executor.execute_plan_async(plan))

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 3)
        self.assertEqual(execution_order, ["A", "B", "C"])
        self.assertEqual(res.execution_order, ["task_a", "task_b", "task_c"])

    def test_diamond_graph(self) -> None:
        """Verify diamond graph: A -> B, C -> D."""
        order = []

        def make_handler(name: str):
            def handler(t: Task) -> str:
                order.append(name)
                return f"Done {name}"
            return handler

        self.executor.register_handler("action_a", make_handler("A"))
        self.executor.register_handler("action_b", make_handler("B"))
        self.executor.register_handler("action_c", make_handler("C"))
        self.executor.register_handler("action_d", make_handler("D"))

        task_a = Task(id="task_a", action="action_a", target="A", dependencies=[])
        task_b = Task(id="task_b", action="action_b", target="B", dependencies=["task_a"])
        task_c = Task(id="task_c", action="action_c", target="C", dependencies=["task_a"])
        task_d = Task(id="task_d", action="action_d", target="D", dependencies=["task_b", "task_c"])

        plan = Plan(query="diamond graph", tasks=[task_a, task_b, task_c, task_d])
        res = self.executor.execute_plan(plan)

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 4)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(len(res.skipped_tasks), 0)

        # A must be first, D must be last, B and C in the middle
        self.assertEqual(order[0], "A")
        self.assertEqual(order[3], "D")
        self.assertEqual(set(order[1:3]), {"B", "C"})

        self.assertEqual(res.execution_order[0], "task_a")
        self.assertEqual(res.execution_order[3], "task_d")
        self.assertEqual(set(res.execution_order[1:3]), {"task_b", "task_c"})

    def test_diamond_graph_async(self) -> None:
        """Verify diamond graph with execute_plan_async."""
        order = []

        def make_async_handler(name: str):
            async def handler(t: Task) -> str:
                await asyncio.sleep(0.005)
                order.append(name)
                return f"Done {name}"
            return handler

        self.executor.register_handler("action_a", make_async_handler("A"))
        self.executor.register_handler("action_b", make_async_handler("B"))
        self.executor.register_handler("action_c", make_async_handler("C"))
        self.executor.register_handler("action_d", make_async_handler("D"))

        task_a = Task(id="task_a", action="action_a", target="A", dependencies=[])
        task_b = Task(id="task_b", action="action_b", target="B", dependencies=["task_a"])
        task_c = Task(id="task_c", action="action_c", target="C", dependencies=["task_a"])
        task_d = Task(id="task_d", action="action_d", target="D", dependencies=["task_b", "task_c"])

        plan = Plan(query="diamond graph async", tasks=[task_a, task_b, task_c, task_d])
        res = asyncio.run(self.executor.execute_plan_async(plan))

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 4)
        self.assertEqual(order[0], "A")
        self.assertEqual(order[3], "D")
        self.assertEqual(set(order[1:3]), {"B", "C"})

    def test_independent_parallel_tasks(self) -> None:
        """Verify independent parallel tasks execute concurrently."""
        def slow_handler(t: Task) -> str:
            time.sleep(0.04)
            return f"Processed {t.id}"

        self.executor.register_handler("slow_action", slow_handler)

        t1 = Task(id="p1", action="slow_action", target="1", dependencies=[])
        t2 = Task(id="p2", action="slow_action", target="2", dependencies=[])
        t3 = Task(id="p3", action="slow_action", target="3", dependencies=[])

        plan = Plan(query="parallel tasks", tasks=[t1, t2, t3])

        start_time = time.time()
        res = self.executor.execute_plan(plan)
        elapsed = time.time() - start_time

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 3)
        # If executed sequentially it would take >= 0.12s. With 3 concurrent threads it takes < 0.10s.
        self.assertLess(elapsed, 0.11)

    def test_independent_parallel_tasks_async(self) -> None:
        """Verify independent tasks execute concurrently in async mode."""
        async def slow_async_handler(t: Task) -> str:
            await asyncio.sleep(0.04)
            return f"Processed async {t.id}"

        self.executor.register_handler("slow_async", slow_async_handler)

        t1 = Task(id="p1", action="slow_async", target="1", dependencies=[])
        t2 = Task(id="p2", action="slow_async", target="2", dependencies=[])
        t3 = Task(id="p3", action="slow_async", target="3", dependencies=[])

        plan = Plan(query="parallel async tasks", tasks=[t1, t2, t3])

        start_time = time.time()
        res = asyncio.run(self.executor.execute_plan_async(plan))
        elapsed = time.time() - start_time

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 3)
        self.assertLess(elapsed, 0.10)

    def test_cycle_detection(self) -> None:
        """Verify Kahn's algorithm detects circular dependency and halts before execution."""
        called_tasks = []

        def dummy_handler(t: Task) -> str:
            called_tasks.append(t.id)
            return "Done"

        self.executor.register_handler("dummy", dummy_handler)

        # Cycle: A -> B -> A
        task_a = Task(id="task_a", action="dummy", target="A", dependencies=["task_b"])
        task_b = Task(id="task_b", action="dummy", target="B", dependencies=["task_a"])

        plan = Plan(query="cycle plan", tasks=[task_a, task_b])
        res = self.executor.execute_plan(plan)

        self.assertFalse(res.success)
        self.assertEqual(len(res.completed_tasks), 0)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(len(called_tasks), 0)
        self.assertIn("circular", res.output.lower())

        # Async mode should also reject without execution
        res_async = asyncio.run(self.executor.execute_plan_async(plan))
        self.assertFalse(res_async.success)
        self.assertEqual(len(called_tasks), 0)
        self.assertIn("circular", res_async.output.lower())

    def test_missing_dependency(self) -> None:
        """Verify missing dependency ID halts execution with failure."""
        called = []
        self.executor.register_handler("dummy", lambda t: called.append(t.id) or "ok")

        task_a = Task(id="task_a", action="dummy", target="A", dependencies=["non_existent_id"])
        plan = Plan(query="missing dep plan", tasks=[task_a])

        res = self.executor.execute_plan(plan)
        self.assertFalse(res.success)
        self.assertEqual(len(called), 0)
        self.assertIn("missing", res.output.lower())

    def test_duplicate_task_ids(self) -> None:
        """Verify duplicate task IDs are rejected during graph validation."""
        called = []
        self.executor.register_handler("dummy", lambda t: called.append(t.id) or "ok")

        t1 = Task(id="duplicate_id", action="dummy", target="A", dependencies=[])
        t2 = Task(id="duplicate_id", action="dummy", target="B", dependencies=[])
        plan = Plan(query="dup plan", tasks=[t1, t2])

        res = self.executor.execute_plan(plan)
        self.assertFalse(res.success)
        self.assertEqual(len(called), 0)
        self.assertIn("duplicate", res.output.lower())

    def test_self_dependency(self) -> None:
        """Verify self-dependency is rejected during graph validation."""
        called = []
        self.executor.register_handler("dummy", lambda t: called.append(t.id) or "ok")

        t1 = Task(id="self_dep", action="dummy", target="A", dependencies=["self_dep"])
        plan = Plan(query="self dep plan", tasks=[t1])

        res = self.executor.execute_plan(plan)
        self.assertFalse(res.success)
        self.assertEqual(len(called), 0)
        self.assertIn("self-dependency", res.output.lower())

    def test_failure_propagation(self) -> None:
        """Verify failure in task A cascades SKIPPED to descendants B and C."""
        executed = []

        def failing_handler(t: Task) -> str:
            executed.append(t.id)
            raise RuntimeError("Task A encountered fatal error")

        def regular_handler(t: Task) -> str:
            executed.append(t.id)
            return "OK"

        self.executor.register_handler("failing", failing_handler)
        self.executor.register_handler("regular", regular_handler)

        task_a = Task(id="task_a", action="failing", target="A", dependencies=[])
        task_b = Task(id="task_b", action="regular", target="B", dependencies=["task_a"])
        task_c = Task(id="task_c", action="regular", target="C", dependencies=["task_b"])

        plan = Plan(query="failure propagation plan", tasks=[task_a, task_b, task_c])
        res = self.executor.execute_plan(plan)

        self.assertFalse(res.success)
        self.assertEqual(executed, ["task_a"])  # B and C should NEVER be executed
        self.assertEqual(len(res.completed_tasks), 0)
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(res.failed_tasks[0].id, "task_a")
        self.assertEqual(res.failed_tasks[0].status, TaskStatus.FAILED)

        self.assertEqual(len(res.skipped_tasks), 2)
        skipped_ids = [t.id for t in res.skipped_tasks]
        self.assertIn("task_b", skipped_ids)
        self.assertIn("task_c", skipped_ids)
        self.assertEqual(task_b.status, TaskStatus.SKIPPED)
        self.assertEqual(task_c.status, TaskStatus.SKIPPED)

        self.assertIn("task_b", res.dependency_failures)
        self.assertIn("task_c", res.dependency_failures)
        self.assertIn("task_a", res.dependency_failures["task_b"])
        self.assertIn("task_a", res.dependency_failures["task_c"])

    def test_failure_propagation_async(self) -> None:
        """Verify failure propagation works identically in execute_plan_async."""
        executed = []

        async def failing_async(t: Task) -> str:
            executed.append(t.id)
            raise RuntimeError("Async Task A exploded")

        async def regular_async(t: Task) -> str:
            executed.append(t.id)
            return "OK"

        self.executor.register_handler("failing_async", failing_async)
        self.executor.register_handler("regular_async", regular_async)

        task_a = Task(id="task_a", action="failing_async", target="A", dependencies=[])
        task_b = Task(id="task_b", action="regular_async", target="B", dependencies=["task_a"])
        task_c = Task(id="task_c", action="regular_async", target="C", dependencies=["task_b"])

        plan = Plan(query="failure async plan", tasks=[task_a, task_b, task_c])
        res = asyncio.run(self.executor.execute_plan_async(plan))

        self.assertFalse(res.success)
        self.assertEqual(executed, ["task_a"])
        self.assertEqual(len(res.completed_tasks), 0)
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(len(res.skipped_tasks), 2)
        self.assertEqual(task_b.status, TaskStatus.SKIPPED)
        self.assertEqual(task_c.status, TaskStatus.SKIPPED)

    def test_independent_branch_still_executes(self) -> None:
        """Verify failing branch does not prevent independent branches from executing."""
        executed = []

        def failing_handler(t: Task) -> str:
            executed.append(t.id)
            raise RuntimeError("Failure in branch 1")

        def success_handler(t: Task) -> str:
            executed.append(t.id)
            return f"Success {t.id}"

        self.executor.register_handler("failing_act", failing_handler)
        self.executor.register_handler("success_act", success_handler)

        # Branch 1: task_a (fails) -> task_b (skipped)
        task_a = Task(id="task_a", action="failing_act", target="A", dependencies=[])
        task_b = Task(id="task_b", action="success_act", target="B", dependencies=["task_a"])

        # Branch 2: task_d (independent, succeeds)
        task_d = Task(id="task_d", action="success_act", target="D", dependencies=[])

        plan = Plan(query="branch test", tasks=[task_a, task_b, task_d])
        res = self.executor.execute_plan(plan)

        self.assertFalse(res.success)
        # task_a and task_d both executed
        self.assertIn("task_a", executed)
        self.assertIn("task_d", executed)
        self.assertNotIn("task_b", executed)

        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(res.completed_tasks[0].id, "task_d")
        self.assertEqual(len(res.failed_tasks), 1)
        self.assertEqual(res.failed_tasks[0].id, "task_a")
        self.assertEqual(len(res.skipped_tasks), 1)
        self.assertEqual(res.skipped_tasks[0].id, "task_b")

    def test_execution_order_correctness(self) -> None:
        """Verify topological execution order correctness across multi-tiered DAG."""
        self.executor.register_handler("act", lambda t: f"Done {t.id}")

        # Graph:
        # A1 -> B1 -> C1
        # A2 -> B2
        # C1 + B2 -> D
        t_a1 = Task(id="a1", action="act", target="1", dependencies=[])
        t_a2 = Task(id="a2", action="act", target="2", dependencies=[])
        t_b1 = Task(id="b1", action="act", target="3", dependencies=["a1"])
        t_b2 = Task(id="b2", action="act", target="4", dependencies=["a2"])
        t_c1 = Task(id="c1", action="act", target="5", dependencies=["b1"])
        t_d = Task(id="d", action="act", target="6", dependencies=["c1", "b2"])

        plan = Plan(query="complex dag", tasks=[t_a1, t_a2, t_b1, t_b2, t_c1, t_d])
        res = self.executor.execute_plan(plan)

        self.assertTrue(res.success)
        order = res.execution_order

        # Assert topological invariants
        self.assertLess(order.index("a1"), order.index("b1"))
        self.assertLess(order.index("b1"), order.index("c1"))
        self.assertLess(order.index("a2"), order.index("b2"))
        self.assertLess(order.index("c1"), order.index("d"))
        self.assertLess(order.index("b2"), order.index("d"))

    def test_backward_compatibility_with_sequential_plans(self) -> None:
        """Verify sequential plans without explicit dependencies execute sequentially and preserve API."""
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: f"Searched {t.target}")

        t1 = Task(action="open_app", target="chrome")
        t2 = Task(action="web_search", target="Python")
        plan = Plan(query="sequential plan", tasks=[t1, t2])

        res = self.executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 2)
        self.assertEqual(len(res.failed_tasks), 0)
        self.assertEqual(len(res.skipped_tasks), 0)
        self.assertEqual(res.completed_tasks[0].target, "chrome")
        self.assertEqual(res.completed_tasks[1].target, "Python")

        d = res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(len(d["completed_tasks"]), 2)
        self.assertEqual(len(d["failed_tasks"]), 0)
        self.assertEqual(len(d["skipped_tasks"]), 0)
        self.assertEqual(len(d["execution_order"]), 2)
        self.assertEqual(d["dependency_failures"], {})



class TestPhase15PlanningExtractionAndReplanningStubs(unittest.TestCase):
    """Test suite for Phase 15 Step 1: Shared Planning Pipeline and Replanning Stubs."""

    def setUp(self) -> None:
        self.mock_provider = MockPlanningProvider()
        self.planner = Planner(
            provider_instance=self.mock_provider,
            strategy=PlanningStrategy.HYBRID,
        )

    def test_shared_generate_plan_sync(self) -> None:
        """Verify _generate_plan_sync correctly invokes provider and parses valid JSON."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t1", "action": "open_app", "target": "chrome"},
                {"id": "t2", "action": "web_search", "target": "news", "dependencies": ["t1"]},
            ]
        })
        tasks = self.planner._generate_plan_sync("sample prompt")
        self.assertIsNotNone(tasks)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].action, "open_app")
        self.assertEqual(tasks[1].dependencies, ["t1"])

        # Returns None on invalid JSON
        self.mock_provider.response_text = "invalid json"
        tasks_invalid = self.planner._generate_plan_sync("sample prompt")
        self.assertIsNone(tasks_invalid)

    def test_shared_generate_plan_async(self) -> None:
        """Verify _generate_plan_async correctly invokes provider and parses valid JSON asynchronously."""
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t_async", "action": "calculate", "target": "10*10"},
            ]
        })
        tasks = asyncio.run(self.planner._generate_plan_async("sample prompt async"))
        self.assertIsNotNone(tasks)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].action, "calculate")

    def test_build_recovery_prompt(self) -> None:
        """Verify _build_recovery_prompt formats completed, failed, and skipped tasks into prompt text."""
        t_completed = Task(id="t_done", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        t_failed = Task(id="t_err", action="web_search", target="invalid_url", status=TaskStatus.FAILED)
        t_skipped = Task(id="t_skip", action="calculate", target="2+2", status=TaskStatus.SKIPPED)

        exec_res = ExecutionResult(
            success=False,
            completed_tasks=[t_completed],
            failed_tasks=[t_failed],
            skipped_tasks=[t_skipped],
        )

        prompt = self.planner._build_recovery_prompt(
            user_query="open chrome and search news and calculate",
            execution_result=exec_res,
            original_plan=[t_completed, t_failed, t_skipped],
        )

        self.assertIn("t_done", prompt)
        self.assertIn("open_app", prompt)
        self.assertIn("t_err", prompt)
        self.assertIn("t_skip", prompt)
        self.assertIn("DO NOT REPEAT", prompt)

    def test_validate_recovery_tasks_success(self) -> None:
        """Verify _validate_recovery_tasks approves valid uncompleted tasks with valid dependencies."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        # Recovery task depends on completed c1 and newly introduced r1
        r1 = Task(id="r1", action="web_search", target="python docs", dependencies=["c1"])
        r2 = Task(id="r2", action="calculate", target="5+5", dependencies=["r1"])

        is_valid, errors = self.planner._validate_recovery_tasks([r1, r2], exec_res)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_recovery_tasks_reject_completed_id(self) -> None:
        """Verify _validate_recovery_tasks rejects task ID that already completed."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        r_bad = Task(id="c1", action="web_search", target="different")
        is_valid, errors = self.planner._validate_recovery_tasks([r_bad], exec_res)
        self.assertFalse(is_valid)
        self.assertTrue(any("already completed in prior execution" in e for e in errors))

    def test_validate_recovery_tasks_reject_completed_action_target(self) -> None:
        """Verify _validate_recovery_tasks rejects duplicate action+target that already completed."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        r_dup = Task(id="new_id", action="open_app", target="chrome")
        is_valid, errors = self.planner._validate_recovery_tasks([r_dup], exec_res)
        self.assertFalse(is_valid)
        self.assertTrue(any("repeats already completed action" in e for e in errors))

    def test_validate_recovery_tasks_unresolved_dependency(self) -> None:
        """Verify _validate_recovery_tasks rejects dependencies not in recovery tasks or completed tasks."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        r_bad_dep = Task(id="r1", action="web_search", target="query", dependencies=["unknown_id"])
        is_valid, errors = self.planner._validate_recovery_tasks([r_bad_dep], exec_res)
        self.assertFalse(is_valid)
        self.assertTrue(any("unsatisfied dependency 'unknown_id'" in e for e in errors))

    def test_replan_success(self) -> None:
        """Verify replan() generates and validates a valid recovery plan."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        t_failed = Task(id="f1", action="web_search", target="weather", status=TaskStatus.FAILED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done], failed_tasks=[t_failed])

        # Provider provides recovery plan with dependency referencing completed c1
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "web_search", "target": "weather in Tokyo", "dependencies": ["c1"]},
            ]
        })

        plan = self.planner.replan("check weather in tokyo", exec_res, [t_done, t_failed])
        self.assertFalse(plan.is_empty())
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].id, "r1")
        self.assertEqual(plan.tasks[0].dependencies, ["c1"])
        self.assertTrue(plan.metadata.get("is_recovery"))

    def test_replan_async_success(self) -> None:
        """Verify replan_async() works asynchronously."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r_async", "action": "calculate", "target": "100/4"},
            ]
        })

        plan = asyncio.run(self.planner.replan_async("calculate 100/4", exec_res))
        self.assertFalse(plan.is_empty())
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].id, "r_async")
        self.assertTrue(plan.metadata.get("is_recovery"))

    def test_replan_validation_failure(self) -> None:
        """Verify replan() returns empty plan when recovery tasks violate recovery validation."""
        t_done = Task(id="c1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t_done])

        # LLM attempts to re-execute already completed action
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "open_app", "target": "chrome"},
            ]
        })

        plan = self.planner.replan("open chrome", exec_res)
        self.assertTrue(plan.is_empty())
        self.assertIn("validation_errors", plan.metadata)



class TestPhase15RecoveryOrchestration(unittest.TestCase):
    """Test suite for Phase 15 Step 2: ExecutionResult Merging & AIManager Recovery Orchestration."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.mock_provider = MockPlanningProvider()
        self.planner = Planner(
            provider_instance=self.mock_provider,
            strategy=PlanningStrategy.HYBRID,
        )
        self.executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def test_execution_result_merge_immutable(self) -> None:
        """Verify ExecutionResult.merge() immutably merges two execution results."""
        t1 = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", target="bad", status=TaskStatus.FAILED)
        t3 = Task(id="t3", action="calculate", target="5+5", status=TaskStatus.COMPLETED)

        res1 = ExecutionResult(
            success=False,
            completed_tasks=[t1],
            failed_tasks=[t2],
            skipped_tasks=[],
            execution_order=["t1", "t2"],
            dependency_failures={"t2": ["t1"]},
            execution_duration=0.5,
            output="Initial Failure",
        )

        res2 = ExecutionResult(
            success=True,
            completed_tasks=[t3],
            failed_tasks=[],
            skipped_tasks=[],
            execution_order=["t3"],
            dependency_failures={},
            execution_duration=0.3,
            output="Recovery Succeeded",
        )

        merged = res1.merge(res2)

        # Assert merged fields
        self.assertTrue(merged.success)
        self.assertEqual(len(merged.completed_tasks), 2)
        self.assertEqual(merged.completed_tasks[0].id, "t1")
        self.assertEqual(merged.completed_tasks[1].id, "t3")
        self.assertEqual(len(merged.failed_tasks), 0)
        self.assertEqual(len(merged.skipped_tasks), 0)
        self.assertEqual(merged.execution_order, ["t1", "t2", "t3"])
        self.assertEqual(merged.dependency_failures, {"t2": ["t1"]})
        self.assertAlmostEqual(merged.execution_duration, 0.8)
        self.assertEqual(merged.output, "Recovery Succeeded")

        # Assert immutability of source objects
        self.assertFalse(res1.success)
        self.assertEqual(len(res1.completed_tasks), 1)
        self.assertEqual(len(res2.completed_tasks), 1)

    def test_execution_result_helper_properties(self) -> None:
        """Verify completed_task_ids, failed_task_ids, and skipped_task_ids properties."""
        t1 = Task(id="t1", action="open_app")
        t2 = Task(id="t2", action="web_search")
        t3 = Task(id="t3", action="calculate")

        res = ExecutionResult(
            success=False,
            completed_tasks=[t1],
            failed_tasks=[t2],
            skipped_tasks=[t3],
        )

        self.assertEqual(res.completed_task_ids, {"t1"})
        self.assertEqual(res.failed_task_ids, {"t2"})
        self.assertEqual(res.skipped_task_ids, {"t3"})

    def test_should_replan_conditions(self) -> None:
        """Verify AIManager._should_replan evaluates all preconditions accurately."""
        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=True,
            max_replans=1,
            auto_register_in_container=False,
        )

        t_fail = Task(id="f1", action="calc")
        res_fail = ExecutionResult(success=False, failed_tasks=[t_fail])
        res_success = ExecutionResult(success=True)
        res_no_failed_tasks = ExecutionResult(success=False, failed_tasks=[], skipped_tasks=[])

        # Normal condition: should replan
        self.assertTrue(mgr._should_replan(res_fail, attempt=0))

        # No replan on success
        self.assertFalse(mgr._should_replan(res_success, attempt=0))

        # No replan when attempt >= max_replans
        self.assertFalse(mgr._should_replan(res_fail, attempt=1))

        # No replan if no failed/skipped tasks remain
        self.assertFalse(mgr._should_replan(res_no_failed_tasks, attempt=0))

        # No replan when disabled
        mgr_disabled = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=False,
            max_replans=1,
            auto_register_in_container=False,
        )
        self.assertFalse(mgr_disabled._should_replan(res_fail, attempt=0))

    def test_build_replan_response_metadata(self) -> None:
        """Verify _build_replan_response_metadata constructs complete response metadata."""
        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_register_in_container=False,
        )

        plan = Plan(query="test", tasks=[Task(id="1", action="open_app")])
        rec_plan = Plan(query="test", tasks=[Task(id="2", action="web_search")])
        res = ExecutionResult(success=True, completed_tasks=[Task(id="1", action="open_app")])

        meta = mgr._build_replan_response_metadata(
            original_plan=plan,
            recovery_plan=rec_plan,
            execution_result=res,
            execution_results=[res],
            replanned=True,
            replan_attempts=1,
            execution_duration=1.25,
        )

        self.assertIn("planning_strategy", meta)
        self.assertTrue(meta["replanned"])
        self.assertEqual(meta["replan_attempts"], 1)
        self.assertIsNotNone(meta["original_plan"])
        self.assertIsNotNone(meta["recovery_plan"])
        self.assertEqual(len(meta["execution_results"]), 1)
        self.assertEqual(meta["completed_count"], 1)
        self.assertEqual(meta["failed_count"], 0)
        self.assertEqual(meta["execution_duration"], 1.25)

    def test_successful_recovery_execution_sync(self) -> None:
        """Verify end-to-end sync recovery execution in AIManager.generate()."""
        # Step 1: Handler setup
        # open_app succeeds; web_search fails initially; calculate succeeds on recovery
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: (_ for _ in ()).throw(RuntimeError("Network down")))
        self.executor.register_handler("calculate", lambda t: f"Result is 10")

        # Setup recovery response from mock provider
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "calculate", "target": "5+5"},
            ]
        })

        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=True,
            max_replans=1,
            auto_register_in_container=False,
        )

        resp = mgr.generate("open chrome and search ChatGPT")

        self.assertTrue(resp.metadata.get("replanned"))
        self.assertEqual(resp.metadata.get("replan_attempts"), 1)
        self.assertIsNotNone(resp.metadata.get("recovery_plan"))
        self.assertEqual(len(resp.metadata.get("execution_results", [])), 2)
        self.assertEqual(resp.metadata.get("completed_count"), 2)
        self.assertIn("Calculate 5+5", resp.content)
        self.assertIn("2 completed", resp.content)

    def test_recovery_execution_failure_sync(self) -> None:
        """Verify end-to-end sync recovery failure in AIManager.generate()."""
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: (_ for _ in ()).throw(RuntimeError("Network down")))

        # Recovery plan also fails
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "web_search", "target": "backup server"},
            ]
        })

        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=True,
            max_replans=1,
            auto_register_in_container=False,
        )

        resp = mgr.generate("open chrome and search ChatGPT")

        self.assertTrue(resp.metadata.get("replanned"))
        self.assertEqual(resp.metadata.get("replan_attempts"), 1)
        self.assertEqual(resp.metadata.get("completed_count"), 1)
        self.assertEqual(resp.metadata.get("failed_count"), 1)
        self.assertIn("Failed:", resp.content)

    def test_async_recovery_flow(self) -> None:
        """Verify end-to-end async recovery execution in AIManager.generate_async()."""
        async def async_open(t: Task) -> str:
            await asyncio.sleep(0.001)
            return f"Opened {t.target}"

        async def async_fail(t: Task) -> str:
            await asyncio.sleep(0.001)
            raise RuntimeError("Async search error")

        async def async_calc(t: Task) -> str:
            await asyncio.sleep(0.001)
            return "10"

        self.executor.register_handler("open_app", async_open)
        self.executor.register_handler("web_search", async_fail)
        self.executor.register_handler("calculate", async_calc)

        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r_async", "action": "calculate", "target": "5+5"},
            ]
        })

        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=True,
            max_replans=1,
            auto_register_in_container=False,
        )

        resp = asyncio.run(mgr.generate_async("open chrome and search ChatGPT"))

        self.assertTrue(resp.metadata.get("replanned"))
        self.assertEqual(resp.metadata.get("replan_attempts"), 1)
        self.assertEqual(resp.metadata.get("completed_count"), 2)
        self.assertEqual(resp.metadata.get("failed_count"), 0)

    def test_no_replan_when_auto_replan_disabled(self) -> None:
        """Verify no recovery replan occurs when auto_replan is disabled."""
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: (_ for _ in ()).throw(RuntimeError("Network down")))

        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=False,
            max_replans=1,
            auto_register_in_container=False,
        )

        resp = mgr.generate("open chrome and search ChatGPT")

        self.assertFalse(resp.metadata.get("replanned"))
        self.assertEqual(resp.metadata.get("replan_attempts"), 0)
        self.assertIsNone(resp.metadata.get("recovery_plan"))
        self.assertEqual(len(resp.metadata.get("execution_results", [])), 1)
        self.assertEqual(resp.metadata.get("failed_count"), 1)


class TestPhase15Step3RecoveryHardening(unittest.TestCase):
    """Unit tests for Phase 15 Step 3: Recovery hardening, edge cases, and production readiness."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.mock_provider = MockPlanningProvider()
        self.planner = Planner(provider_instance=self.mock_provider)
        self.executor = Executor(container_instance=self.container)

    def test_merge_immutability_and_deduplication(self) -> None:
        """Verify ExecutionResult.merge preserves immutability and deduplicates completed tasks."""
        t1 = Task(id="t1", action="open_app", target="notepad")
        t2 = Task(id="t2", action="calculate", target="2+2")
        t_dup = Task(id="t1", action="open_app", target="notepad")  # duplicate ID

        res1 = ExecutionResult(
            success=False,
            completed_tasks=[t1],
            failed_tasks=[t2],
            execution_order=["t1", "t2"],
            dependency_failures={"t3": ["t2"]},
            execution_duration=1.5,
            output="Initial output",
        )

        t3 = Task(id="t3", action="web_search", target="python")
        res2 = ExecutionResult(
            success=True,
            completed_tasks=[t_dup, t3],
            failed_tasks=[],
            skipped_tasks=[],
            execution_order=["t1", "t3"],
            dependency_failures={"t3": ["t4"]},
            execution_duration=2.0,
            output="Recovery output",
        )

        merged = res1.merge(res2)

        # Immutability: original objects remain untouched
        self.assertEqual(len(res1.completed_tasks), 1)
        self.assertEqual(len(res2.completed_tasks), 2)
        self.assertEqual(res1.execution_duration, 1.5)
        self.assertEqual(res2.execution_duration, 2.0)

        # Deduplication of completed tasks by ID: t1 appears only once
        self.assertEqual(len(merged.completed_tasks), 2)
        self.assertEqual([t.id for t in merged.completed_tasks], ["t1", "t3"])

        # Deduplication of execution_order: t1, t2, t3
        self.assertEqual(merged.execution_order, ["t1", "t2", "t3"])

        # Dependency failures unioned
        self.assertEqual(merged.dependency_failures["t3"], ["t2", "t4"])

        # Accumulated duration
        self.assertAlmostEqual(merged.execution_duration, 3.5)

        # Final authoritative outcome from res2
        self.assertTrue(merged.success)
        self.assertEqual(len(merged.failed_tasks), 0)
        self.assertEqual(merged.output, "Recovery output")

    def test_merge_with_empty_results(self) -> None:
        """Verify ExecutionResult.merge handles empty results cleanly."""
        empty1 = ExecutionResult(success=True)
        empty2 = ExecutionResult(success=False)

        merged_empty = empty1.merge(empty2)
        self.assertFalse(merged_empty.success)
        self.assertEqual(merged_empty.completed_tasks, [])
        self.assertEqual(merged_empty.execution_order, [])
        self.assertEqual(merged_empty.execution_duration, 0.0)

        t1 = Task(id="1", action="open_app")
        non_empty = ExecutionResult(success=True, completed_tasks=[t1], output="Done")
        merged_with_non_empty = empty1.merge(non_empty)
        self.assertTrue(merged_with_non_empty.success)
        self.assertEqual(len(merged_with_non_empty.completed_tasks), 1)
        self.assertEqual(merged_with_non_empty.output, "Done")

    def test_recovery_returning_none_or_empty_plan(self) -> None:
        """Verify replan returns empty plan when provider returns empty or None."""
        res = ExecutionResult(
            success=False,
            failed_tasks=[Task(id="f1", action="web_search", target="test")],
        )

        # 1. Provider returns None
        self.mock_provider.response_text = ""
        rec_none = self.planner.replan("search test", res)
        self.assertTrue(rec_none.is_empty())
        self.assertTrue(rec_none.metadata.get("is_recovery"))

        # 2. Provider returns empty tasks JSON
        self.mock_provider.response_text = json.dumps({"tasks": []})
        rec_empty = self.planner.replan("search test", res)
        self.assertTrue(rec_empty.is_empty())
        self.assertTrue(rec_empty.metadata.get("is_recovery"))

    def test_invalid_recovery_dag_cycle_rejected(self) -> None:
        """Verify replan detects and rejects DAG validation failures in the combined recovery graph."""
        res = ExecutionResult(
            success=False,
            completed_tasks=[Task(id="c1", action="open_app", target="chrome")],
            failed_tasks=[Task(id="f1", action="web_search", target="test")],
        )

        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "calculate", "target": "5+5", "dependencies": ["c1"]},
            ]
        })

        from unittest.mock import patch
        with patch("app.ai.planner.executor.executor.validate_dag", return_value=(False, "Cycle detected in recovery graph")):
            rec_plan = self.planner.replan("calculate 5+5", res)
            self.assertTrue(rec_plan.is_empty())
            self.assertIn("validation_errors", rec_plan.metadata)
            self.assertIn("Cycle detected in recovery graph", rec_plan.metadata["validation_errors"])

    def test_recovery_validation_completed_task_rejection(self) -> None:
        """Verify _validate_recovery_tasks rejects tasks already completed or repeating completed actions."""
        completed_task = Task(id="c1", action="open_app", target="chrome")
        res = ExecutionResult(success=False, completed_tasks=[completed_task])

        # 1. Task reuses completed ID
        tasks_dup_id = [Task(id="c1", action="calculate", target="5+5")]
        valid, errors = self.planner._validate_recovery_tasks(tasks_dup_id, res)
        self.assertFalse(valid)
        self.assertTrue(any("already completed in prior execution" in e for e in errors))

        # 2. Task repeats completed action and target
        tasks_dup_action = [Task(id="r1", action="open_app", target="chrome")]
        valid2, errors2 = self.planner._validate_recovery_tasks(tasks_dup_action, res)
        self.assertFalse(valid2)
        self.assertTrue(any("repeats already completed action" in e for e in errors2))

    def test_recovery_execution_no_progress_halts(self) -> None:
        """Verify AIManager halts recovery immediately if recovery produces zero progress."""
        self.executor.register_handler("open_app", lambda t: f"Opened {t.target}")
        self.executor.register_handler("web_search", lambda t: (_ for _ in ()).throw(RuntimeError("Network down")))

        # Recovery plan also fails without completing any task
        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "r1", "action": "web_search", "target": "backup server"},
            ]
        })

        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_replan=True,
            max_replans=3,  # allows up to 3 attempts, but should halt after 1 due to 0 progress
            auto_register_in_container=False,
        )

        resp = mgr.generate("open chrome and search ChatGPT")
        self.assertTrue(resp.metadata.get("replanned"))
        # Halts after 1 attempt because 0 progress was made
        self.assertEqual(resp.metadata.get("replan_attempts"), 1)
        self.assertEqual(resp.metadata.get("failed_count"), 1)

    def test_metadata_backwards_compatibility_and_types(self) -> None:
        """Verify all expected metadata fields are present and backwards-compatible."""
        mgr = AIManager(
            container_instance=self.container,
            planner_instance=self.planner,
            executor_instance=self.executor,
            provider_instance=self.mock_provider,
            auto_register_in_container=False,
        )

        plan = Plan(query="sample query", tasks=[Task(id="1", action="open_app")])
        res = ExecutionResult(success=True, completed_tasks=[Task(id="1", action="open_app")])

        meta = mgr._build_replan_response_metadata(
            original_plan=plan,
            recovery_plan=None,
            execution_result=res,
            execution_results=[res],
            replanned=False,
            replan_attempts=0,
            execution_duration=0.5,
        )

        expected_keys = {
            "plan",
            "planning_strategy",
            "replanned",
            "replan_attempts",
            "original_plan",
            "recovery_plan",
            "execution_results",
            "execution_result",
            "completed_count",
            "failed_count",
            "execution_duration",
        }
        self.assertTrue(expected_keys.issubset(meta.keys()))
        self.assertEqual(meta["plan"], meta["original_plan"])
        self.assertIsNone(meta["recovery_plan"])
        self.assertFalse(meta["replanned"])
        self.assertEqual(meta["replan_attempts"], 0)
        self.assertEqual(meta["completed_count"], 1)
        self.assertEqual(meta["failed_count"], 0)


class TestPhase16AdaptivePlanningAndExecutionMemory(unittest.TestCase):
    """Comprehensive test suite for Phase 16: Adaptive Planning and Execution Memory."""

    def setUp(self) -> None:
        self.mock_provider = MockPlanningProvider()
        self.planner = Planner(
            provider_instance=self.mock_provider,
            strategy=PlanningStrategy.HYBRID,
        )

    # 1. ExecutionMemory & TaskExecutionRecord & ExecutionMetrics
    def test_execution_memory_creation_and_computed_properties(self) -> None:
        """Verify ExecutionMemory initialization from ExecutionResult and computed properties."""
        t1 = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", target="query", status=TaskStatus.FAILED)
        t3 = Task(id="t3", action="calculate", target="1+1", status=TaskStatus.SKIPPED)

        res = ExecutionResult(
            success=False,
            completed_tasks=[t1],
            failed_tasks=[t2],
            skipped_tasks=[t3],
            output="output_t1",
            execution_order=["t1", "t2"],
            dependency_failures={"t3": ["t2"]},
            execution_duration=1.5,
        )

        mem = ExecutionMemory.from_execution_result(res, wave=1)

        self.assertEqual(len(mem.records), 3)
        self.assertEqual(mem.completed_task_ids, {"t1"})
        self.assertEqual(mem.failed_task_ids, {"t2"})
        self.assertEqual(mem.skipped_task_ids, {"t3"})
        self.assertEqual(mem.execution_order, ["t1", "t2", "t3"])
        self.assertEqual(mem.dependency_failures, {"t3": ["t2"]})
        self.assertEqual(mem.task_outputs, {"t1": "output_t1"})
        self.assertEqual(mem.retry_history, {"t1": 1, "t2": 1, "t3": 1})
        self.assertEqual(mem.metrics.execution_waves, 1)

    def test_execution_memory_record_execution_and_accumulation(self) -> None:
        """Verify recording subsequent execution waves updates attempts and history."""
        t1 = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.FAILED)
        res1 = ExecutionResult(success=False, failed_tasks=[t1], output="timeout error", execution_order=["t1"])

        mem = ExecutionMemory.from_execution_result(res1, wave=1)
        self.assertEqual(mem.retry_history["t1"], 1)

        # Retry in wave 2 succeeds
        t1_retry = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        res2 = ExecutionResult(success=True, completed_tasks=[t1_retry], output="ok", execution_order=["t1"])
        mem = mem.record_execution(res2, wave=2)

        self.assertEqual(len(mem.records), 2)
        self.assertEqual(mem.retry_history["t1"], 2)
        self.assertEqual(mem.completed_task_ids, {"t1"})
        self.assertEqual(mem.failed_task_ids, set())
        self.assertEqual(mem.metrics.execution_waves, 2)

    def test_execution_memory_immutable_merge(self) -> None:
        """Verify merge() creates a new ExecutionMemory without mutating either source."""
        t1 = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", target="url", status=TaskStatus.FAILED)

        res1 = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="err")
        mem1 = ExecutionMemory.from_execution_result(res1, wave=1)

        t2_fixed = Task(id="t2", action="web_search", target="url", status=TaskStatus.COMPLETED)
        res2 = ExecutionResult(success=True, completed_tasks=[t2_fixed], output="fixed")
        mem2 = ExecutionMemory.from_execution_result(res2, wave=2)

        merged = mem1.merge(mem2)

        self.assertIsNot(merged, mem1)
        self.assertIsNot(merged, mem2)
        # Original remains unchanged
        self.assertEqual(len(mem1.records), 2)
        self.assertEqual(len(mem2.records), 1)
        # Merged contains both records
        self.assertEqual(len(merged.records), 3)
        self.assertEqual(merged.completed_task_ids, {"t1", "t2"})
        self.assertEqual(merged.failed_task_ids, set())

    def test_execution_memory_serialization(self) -> None:
        """Verify to_dict and get_summary serialization structure."""
        t1 = Task(id="t1", action="calculate", target="2+2", status=TaskStatus.COMPLETED)
        res = ExecutionResult(success=True, completed_tasks=[t1], output="4")
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        d = mem.to_dict()
        self.assertIn("records", d)
        self.assertIn("dependency_failures", d)
        self.assertIn("metrics", d)
        self.assertEqual(len(d["records"]), 1)
        self.assertEqual(d["records"][0]["task_id"], "t1")
        self.assertEqual(d["records"][0]["status"], "COMPLETED")

        summary = mem.get_summary()
        self.assertEqual(summary["total_records"], 1)
        self.assertEqual(summary["completed_count"], 1)

    def test_execution_metrics_average_duration(self) -> None:
        """Verify ExecutionMetrics computes average_recovery_duration correctly."""
        metrics = ExecutionMetrics(
            planning_count=2,
            replan_count=2,
            successful_recoveries=1,
            failed_recoveries=1,
            total_recovery_duration=3.0,
            execution_waves=2,
        )
        self.assertAlmostEqual(metrics.average_recovery_duration, 1.5)

        zero_metrics = ExecutionMetrics()
        self.assertEqual(zero_metrics.average_recovery_duration, 0.0)

    # 2. FailureClassifier
    def test_failure_classifier_categories(self) -> None:
        """Verify FailureClassifier correctly categorizes distinct failure types and falls back to UNKNOWN."""
        t = Task(id="t", action="open_app")

        # Dependency failure
        cat_dep = FailureClassifier.classify(t, "dependency failed", dependency_failures={"t": ["parent"]})
        self.assertEqual(cat_dep, FailureCategory.DEPENDENCY_FAILURE)

        # Timeout
        cat_to = FailureClassifier.classify(t, "Task exceeded timeout deadline")
        self.assertEqual(cat_to, FailureCategory.TIMEOUT)

        # Validation failure
        cat_val = FailureClassifier.classify(t, "Schema validation error: missing target field")
        self.assertEqual(cat_val, FailureCategory.VALIDATION_FAILURE)

        # Provider error
        cat_prov = FailureClassifier.classify(t, "LLM provider error: 429 rate limit reached")
        self.assertEqual(cat_prov, FailureCategory.PROVIDER_ERROR)

        # Tool failure
        cat_tool = FailureClassifier.classify(t, "Tool execution error: app not installed")
        self.assertEqual(cat_tool, FailureCategory.TOOL_FAILURE)

        # Generic execution error
        cat_exec = FailureClassifier.classify(t, "RuntimeError: unhandled system error")
        self.assertEqual(cat_exec, FailureCategory.EXECUTION_ERROR)

        # Unknown fallback
        cat_unk = FailureClassifier.classify(t, "")
        self.assertEqual(cat_unk, FailureCategory.UNKNOWN)

    # 3. RecoveryDecision & RecoveryHeuristics
    def test_recovery_heuristics_decisions(self) -> None:
        """Verify evaluate_recovery_viability under all standard conditions."""
        # 1. Viable scenario
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", status=TaskStatus.FAILED)
        plan = Plan(query="test", tasks=[t1, t2])
        res = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="Network glitch")
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        decision = evaluate_recovery_viability("test", plan, mem)
        self.assertTrue(decision.viable)
        self.assertIn("viable", decision.reason.lower())

        # 2. All completed scenario
        t2_done = Task(id="t2", action="web_search", status=TaskStatus.COMPLETED)
        res_done = ExecutionResult(success=True, completed_tasks=[t1, t2_done], output="done")
        mem_done = ExecutionMemory.from_execution_result(res_done, wave=2)
        decision_done = evaluate_recovery_viability("test", plan, mem_done)
        self.assertFalse(decision_done.viable)
        self.assertIn("completed successfully", decision_done.reason.lower())

        # 3. Max retries exceeded
        res_failed = ExecutionResult(success=False, failed_tasks=[t2], output="retry fail")
        mem_retried = ExecutionMemory.from_execution_result(res_failed, wave=1)
        mem_retried = mem_retried.record_execution(res_failed, wave=2)
        mem_retried = mem_retried.record_execution(res_failed, wave=3)
        decision_retried = evaluate_recovery_viability("test", plan, mem_retried)
        self.assertFalse(decision_retried.viable)
        self.assertIn("retry limit", decision_retried.reason)

        # 4. Permanent validation failure
        t_val = Task(id="t_val", action="open_app", status=TaskStatus.FAILED)
        plan_val = Plan(query="val", tasks=[t_val])
        res_val = ExecutionResult(success=False, failed_tasks=[t_val], output="Validation error: invalid schema")
        mem_val = ExecutionMemory.from_execution_result(res_val, wave=1)
        decision_val = evaluate_recovery_viability("val", plan_val, mem_val)
        self.assertFalse(decision_val.viable)
        self.assertIn("validation", decision_val.reason.lower())

    # 4. MemorySummaryBuilder
    def test_memory_summary_builder_outputs(self) -> None:
        """Verify MemorySummaryBuilder produces structured planner and metadata summaries."""
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", status=TaskStatus.FAILED)
        res = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="Tool failed to run")
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        # Planner summary
        p_summary = MemorySummaryBuilder.build_planner_summary(mem)
        self.assertIn("Completed Tasks", p_summary)
        self.assertIn("t1", p_summary)
        self.assertIn("Failed Tasks", p_summary)
        self.assertIn("tool_failure", p_summary)

        # Metadata summary
        m_summary = MemorySummaryBuilder.build_metadata_summary(mem)
        self.assertEqual(m_summary["completed_count"], 1)
        self.assertEqual(m_summary["failed_count"], 1)
        self.assertEqual(m_summary["total_records"], 2)

        # Human readable summary
        h_summary = MemorySummaryBuilder.build_human_readable_summary(mem)
        self.assertIn("across 1 wave(s)", h_summary)
        self.assertIn("1 completed", h_summary)

    # 5. Planner Integration
    def test_planner_replan_execution_result_backwards_compatibility(self) -> None:
        """Verify Planner.replan() continues accepting ExecutionResult seamlessly."""
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", status=TaskStatus.FAILED)
        exec_res = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="timeout error")

        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t2_retry", "action": "web_search", "target": "news", "dependencies": []}
            ]
        })

        rec_plan = self.planner.replan("search news", exec_res)
        self.assertIsNotNone(rec_plan)
        self.assertEqual(len(rec_plan.tasks), 1)
        self.assertEqual(rec_plan.tasks[0].id, "t2_retry")

    def test_planner_replan_with_execution_memory_and_heuristic_short_circuit(self) -> None:
        """Verify Planner.replan accepts ExecutionMemory and aborts if heuristics evaluate non-viable."""
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        plan = Plan(query="open app", tasks=[t1])
        res = ExecutionResult(success=True, completed_tasks=[t1])
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        # Provider should NOT be called since all tasks completed
        rec_plan = self.planner.replan("open app", mem, original_plan=plan)
        self.assertIsNotNone(rec_plan)
        self.assertTrue(rec_plan.is_empty())
        self.assertEqual(len(rec_plan.tasks), 0)
        self.assertFalse(rec_plan.metadata["recovery_decision"]["viable"])
        self.assertIsNone(self.mock_provider.last_prompt)

    def test_planner_replan_async_with_execution_memory(self) -> None:
        """Verify Planner.replan_async works with ExecutionMemory."""
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="calculate", status=TaskStatus.FAILED)
        plan = Plan(query="calc", tasks=[t1, t2])
        res = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="calc failure")
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        self.mock_provider.response_text = json.dumps({
            "tasks": [
                {"id": "t2_new", "action": "calculate", "target": "2*2"}
            ]
        })

        rec_plan = asyncio.run(self.planner.replan_async("calc", mem, original_plan=plan))
        self.assertIsNotNone(rec_plan)
        self.assertEqual(len(rec_plan.tasks), 1)
        self.assertEqual(rec_plan.tasks[0].id, "t2_new")

    # 6. AIManager Integration & Metadata
    def test_aimanager_execution_memory_and_metrics_accumulation(self) -> None:
        """Verify AIManager tracks ExecutionMemory across waves and adds execution_summary & execution_metrics to metadata."""
        container = ServiceContainer()
        mock_provider = MockPlanningProvider()
        planner_inst = Planner(provider_instance=mock_provider, strategy=PlanningStrategy.LLM)
        executor_inst = Executor(container_instance=container, auto_register_in_container=False)

        search_attempts = 0
        def search_handler(t: Task) -> str:
            nonlocal search_attempts
            search_attempts += 1
            if search_attempts == 1:
                raise RuntimeError("network timeout")
            return "weather is sunny"

        executor_inst.register_handler("open_app", lambda t: "opened chrome")
        executor_inst.register_handler("web_search", search_handler)

        mgr = AIManager(
            container_instance=container,
            planner_instance=planner_inst,
            executor_instance=executor_inst,
            provider_instance=mock_provider,
            auto_replan=True,
            max_replans=1,
            auto_register_in_container=False,
        )

        # Set up provider to return initial plan then recovery plan
        call_count = 0
        def mock_generate(prompt: str, **kwargs: Any) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                content = json.dumps({
                    "tasks": [
                        {"id": "t1", "action": "open_app", "target": "chrome"},
                        {"id": "t2", "action": "web_search", "target": "weather", "dependencies": ["t1"]},
                    ]
                })
            else:
                content = json.dumps({
                    "tasks": [
                        {"id": "t2_rec", "action": "web_search", "target": "weather", "dependencies": []}
                    ]
                })
            return AIResponse(content=content, model="mock-planning-provider")

        mock_provider.generate = mock_generate

        response: AIResponse = mgr.generate("open chrome and check weather")
        self.assertIsNotNone(response)
        self.assertTrue(response.metadata.get("replanned"))
        self.assertEqual(response.metadata.get("replan_attempts"), 1)

        # Check Phase 16 specific metadata
        self.assertIn("execution_summary", response.metadata)
        self.assertIn("execution_metrics", response.metadata)
        # Verify raw full memory is NOT dumped
        self.assertNotIn("execution_memory", response.metadata)

        metrics = response.metadata["execution_metrics"]
        self.assertEqual(metrics["planning_count"], 1)
        self.assertEqual(metrics["replan_count"], 1)
        self.assertEqual(metrics["successful_recoveries"], 1)
        self.assertEqual(metrics["execution_waves"], 2)


if __name__ == "__main__":
    unittest.main()



