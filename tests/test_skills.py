"""Unit tests for the J.A.R.V.I.S Skill Framework Subsystem."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from app.core.config import Settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.skills.base import (
    BaseSkill,
    InvalidSkillError,
    SkillAlreadyRegisteredError,
    SkillDiscoveryError,
    SkillError,
    SkillExecutionError,
    SkillInitializationError,
    SkillNotFoundError,
)
from app.skills.manager import (
    SkillManager,
    skill_manager,
)


class MockEchoSkill(BaseSkill):
    """Mock skill for testing basic handling and execution."""

    name = "mock_echo"
    description = "Echoes inputs back"
    version = "1.2.0"
    priority = 100
    tags = ["test", "echo"]
    permissions = {"console"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initialized = False
        self.shutdown_called = False

    def initialize(self) -> None:
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def can_handle(self, command: Any) -> bool:
        if isinstance(command, str) and command.startswith("echo:"):
            return True
        if isinstance(command, dict) and command.get("action") == "echo":
            return True
        return False

    def execute(self, command: Any) -> Any:
        if isinstance(command, str):
            return command.replace("echo:", "", 1).strip()
        if isinstance(command, dict):
            return command.get("payload")
        return command


class MockHighPrioritySkill(BaseSkill):
    """Mock skill with high priority."""

    name = "mock_high"
    description = "High priority skill"
    priority = 200
    tags = ["priority", "test"]

    def can_handle(self, command: Any) -> bool:
        return command == "compete"

    def execute(self, command: Any) -> str:
        return "high_priority_won"


class MockLowPrioritySkill(BaseSkill):
    """Mock skill with low priority."""

    name = "mock_low"
    description = "Low priority skill"
    priority = 50
    tags = ["priority", "test"]

    def can_handle(self, command: Any) -> bool:
        return command == "compete"

    def execute(self, command: Any) -> str:
        return "low_priority_won"


class MockFailingSkill(BaseSkill):
    """Mock skill that deliberately fails during execution."""

    name = "mock_failing"
    description = "Fails on execute"

    def can_handle(self, command: Any) -> bool:
        return command == "fail"

    def execute(self, command: Any) -> Any:
        raise ValueError("Simulated skill execution error.")


class MockInitFailingSkill(BaseSkill):
    """Mock skill that fails during initialization."""

    name = "mock_init_fail"
    description = "Fails on initialize"

    def initialize(self) -> None:
        raise RuntimeError("Crash during skill initialization.")

    def can_handle(self, command: Any) -> bool:
        return False

    def execute(self, command: Any) -> Any:
        return None


class MockAsyncSkill(BaseSkill):
    """Mock skill implementing an asynchronous coroutine execute method."""

    name = "mock_async"
    description = "Async coroutine skill"

    def can_handle(self, command: Any) -> bool:
        return command == "run_async"

    async def execute(self, command: Any) -> str:
        await asyncio.sleep(0.01)
        return "async_success"


class TestBaseSkill(unittest.TestCase):
    """Test suite for BaseSkill contract, attributes, and metadata."""

    def test_subclass_cannot_be_instantiated_without_abstract_methods(self) -> None:
        """Verify that BaseSkill cannot be instantiated without can_handle and execute."""
        class IncompleteSkill(BaseSkill):
            name = "incomplete"

        with self.assertRaises(TypeError):
            IncompleteSkill()  # type: ignore[abstract]

    def test_skill_attributes_and_defaults(self) -> None:
        """Verify default and overridden attributes on BaseSkill."""
        skill = MockEchoSkill()
        self.assertEqual(skill.name, "mock_echo")
        self.assertEqual(skill.description, "Echoes inputs back")
        self.assertEqual(skill.version, "1.2.0")
        self.assertTrue(skill.enabled)
        self.assertEqual(skill.priority, 100)
        self.assertEqual(skill.tags, ["test", "echo"])
        self.assertEqual(skill.permissions, {"console"})

    def test_skill_constructor_overrides(self) -> None:
        """Verify constructor keyword arguments properly override class attributes."""
        skill = MockEchoSkill(
            name="custom_echo",
            description="Custom description",
            version="2.0.0",
            enabled=False,
            priority=150,
            tags=["custom"],
            permissions={"network", "disk"},
        )
        self.assertEqual(skill.name, "custom_echo")
        self.assertEqual(skill.description, "Custom description")
        self.assertEqual(skill.version, "2.0.0")
        self.assertFalse(skill.enabled)
        self.assertEqual(skill.priority, 150)
        self.assertEqual(skill.tags, ["custom"])
        self.assertEqual(skill.permissions, {"network", "disk"})

    def test_invalid_skill_name_raises(self) -> None:
        """Verify empty or non-string name raises InvalidSkillError."""
        with self.assertRaises(InvalidSkillError):
            MockEchoSkill(name="")
        with self.assertRaises(InvalidSkillError):
            MockEchoSkill(name="   ")

    def test_skill_metadata_export(self) -> None:
        """Verify metadata() exports correct dictionary structure."""
        skill = MockEchoSkill()
        meta = skill.metadata()
        self.assertEqual(
            meta,
            {
                "name": "mock_echo",
                "description": "Echoes inputs back",
                "version": "1.2.0",
                "enabled": True,
                "priority": 100,
                "tags": ["test", "echo"],
                "permissions": ["console"],
            },
        )
        # to_dict() alias
        self.assertEqual(skill.to_dict(), meta)

    def test_skill_dependency_injection_accessors(self) -> None:
        """Verify skill properties resolve injected or default container, config, logger, bus."""
        custom_bus = EventBus()
        custom_container = ServiceContainer()
        skill = MockEchoSkill()

        # Before binding, properties resolve defaults
        self.assertIsNotNone(skill.logger)
        self.assertIsNotNone(skill.config)
        self.assertIsNotNone(skill.container)
        self.assertIsNotNone(skill.event_bus)

        # After binding, properties resolve injected instances
        skill.bind(event_bus=custom_bus, container=custom_container)
        self.assertIs(skill.event_bus, custom_bus)
        self.assertIs(skill.container, custom_container)

    def test_skill_enable_disable_methods(self) -> None:
        """Verify enable() and disable() methods on BaseSkill instance."""
        skill = MockEchoSkill()
        self.assertTrue(skill.enabled)
        skill.disable()
        self.assertFalse(skill.enabled)
        skill.enable()
        self.assertTrue(skill.enabled)

    def test_skill_execute_async(self) -> None:
        """Verify BaseSkill execute_async on synchronous and asynchronous skills."""
        sync_skill = MockEchoSkill()
        async_skill = MockAsyncSkill()

        async def _test() -> None:
            res_sync = await sync_skill.execute_async("echo: hello async")
            self.assertEqual(res_sync, "hello async")

            res_async = await async_skill.execute_async("run_async")
            self.assertEqual(res_async, "async_success")

        asyncio.run(_test())


class TestSkillManager(unittest.TestCase):
    """Comprehensive test suite for SkillManager lifecycle, routing, and events."""

    def setUp(self) -> None:
        """Create isolated container, event bus, and manager for each test."""
        self.test_container = ServiceContainer()
        self.test_bus = EventBus()
        self.manager = SkillManager(
            container_instance=self.test_container,
            event_bus_instance=self.test_bus,
            auto_register_in_container=True,
        )

    def tearDown(self) -> None:
        """Clean up manager and global singletons."""
        self.manager.clear()
        skill_manager.clear()

    def test_service_container_self_registration(self) -> None:
        """Verify SkillManager registers itself as a singleton in the container."""
        self.assertTrue(self.test_container.exists("skill_manager"))
        resolved = self.test_container.resolve("skill_manager")
        self.assertIs(resolved, self.manager)

    def test_register_and_get_skill(self) -> None:
        """Verify registering a skill calls initialize() and stores it."""
        skill = MockEchoSkill()
        self.assertFalse(skill.initialized)

        self.manager.register(skill)
        self.assertTrue(skill.initialized)
        self.assertEqual(len(self.manager), 1)
        self.assertIn("mock_echo", self.manager)
        self.assertIs(self.manager.get("mock_echo"), skill)
        self.assertIs(self.manager["mock_echo"], skill)

    def test_register_invalid_skill_raises(self) -> None:
        """Verify non-BaseSkill instances raise InvalidSkillError."""
        with self.assertRaises(InvalidSkillError):
            self.manager.register("not_a_skill")  # type: ignore[arg-type]

    def test_duplicate_registration_handling(self) -> None:
        """Verify duplicate skill name raises SkillAlreadyRegisteredError unless allow_override=True."""
        skill1 = MockEchoSkill(name="dup_skill", version="1.0.0")
        skill2 = MockEchoSkill(name="dup_skill", version="2.0.0")

        self.manager.register(skill1)
        with self.assertRaises(SkillAlreadyRegisteredError):
            self.manager.register(skill2)

        # Allow override
        self.manager.register(skill2, allow_override=True)
        retrieved = self.manager.get("dup_skill")
        self.assertIs(retrieved, skill2)
        self.assertEqual(retrieved.version, "2.0.0")

    def test_failed_initialization_raises_and_publishes_event(self) -> None:
        """Verify SkillInitializationError when initialize() fails and event is published."""
        failed_events: list[Event] = []
        self.test_bus.subscribe("skill.failed", lambda e: failed_events.append(e))

        skill = MockInitFailingSkill()
        with self.assertRaises(SkillInitializationError):
            self.manager.register(skill)

        self.assertEqual(len(self.manager), 0)
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["skill_name"], "mock_init_fail")
        self.assertEqual(failed_events[0].payload["stage"], "initialize")

    def test_unregister_invokes_shutdown(self) -> None:
        """Verify unregistering a skill invokes shutdown() and removes it."""
        skill = MockEchoSkill()
        self.manager.register(skill)

        removed = self.manager.unregister("mock_echo")
        self.assertTrue(removed)
        self.assertTrue(skill.shutdown_called)
        self.assertNotIn("mock_echo", self.manager)
        self.assertIsNone(self.manager.get("mock_echo"))

        # Unregister non-existent skill returns False
        self.assertFalse(self.manager.unregister("non_existent"))

    def test_unregister_publishes_event(self) -> None:
        """Verify unregistering a skill publishes skill.unregistered event."""
        unreg_events: list[Event] = []
        self.test_bus.subscribe("skill.unregistered", lambda e: unreg_events.append(e))

        skill = MockEchoSkill()
        self.manager.register(skill)
        self.manager.unregister("mock_echo")

        self.assertEqual(len(unreg_events), 1)
        self.assertEqual(unreg_events[0].payload["skill_name"], "mock_echo")

    def test_list_skills_ordering_and_filters(self) -> None:
        """Verify list_skills orders by priority and supports filtering."""
        low = MockLowPrioritySkill()  # priority=50
        high = MockHighPrioritySkill()  # priority=200
        echo = MockEchoSkill(name="echo_disabled", enabled=False)

        self.manager.register(low)
        self.manager.register(high)
        self.manager.register(echo)

        # All skills ordered by priority (-priority, name)
        all_skills = self.manager.list_skills()
        self.assertEqual([s.name for s in all_skills], ["mock_high", "echo_disabled", "mock_low"])

        # Enabled only
        enabled_skills = self.manager.list_skills(enabled_only=True)
        self.assertEqual([s.name for s in enabled_skills], ["mock_high", "mock_low"])

        # Tag filter
        echo_tagged = self.manager.list_skills(tag="echo")
        self.assertEqual([s.name for s in echo_tagged], ["echo_disabled"])

    def test_enable_and_disable_skill(self) -> None:
        """Verify toggling skill enabled status and event emissions."""
        enabled_events: list[Event] = []
        disabled_events: list[Event] = []
        self.test_bus.subscribe("skill.enabled", lambda e: enabled_events.append(e))
        self.test_bus.subscribe("skill.disabled", lambda e: disabled_events.append(e))

        skill = MockEchoSkill()
        self.manager.register(skill)

        self.assertTrue(self.manager.disable_skill("mock_echo"))
        self.assertFalse(skill.enabled)
        self.assertEqual(len(disabled_events), 1)
        self.assertEqual(disabled_events[0].payload["skill_name"], "mock_echo")

        self.assertTrue(self.manager.enable_skill("mock_echo"))
        self.assertTrue(skill.enabled)
        self.assertEqual(len(enabled_events), 1)
        self.assertEqual(enabled_events[0].payload["skill_name"], "mock_echo")

        self.assertFalse(self.manager.enable_skill("unknown_skill"))
        self.assertFalse(self.manager.disable_skill("unknown_skill"))

    def test_execute_routes_by_priority(self) -> None:
        """Verify execute picks the highest priority matching skill."""
        low = MockLowPrioritySkill()  # priority=50, handles 'compete'
        high = MockHighPrioritySkill()  # priority=200, handles 'compete'

        self.manager.register(low)
        self.manager.register(high)

        # Both can handle 'compete', but high priority (200) should win
        result = self.manager.execute("compete")
        self.assertEqual(result, "high_priority_won")

    def test_execute_synchronous_success_and_event(self) -> None:
        """Verify successful execution publishes skill.executed event."""
        executed_events: list[Event] = []
        self.test_bus.subscribe("skill.executed", lambda e: executed_events.append(e))

        skill = MockEchoSkill()
        self.manager.register(skill)

        result = self.manager.execute("echo: Hello J.A.R.V.I.S")
        self.assertEqual(result, "Hello J.A.R.V.I.S")

        self.assertEqual(len(executed_events), 1)
        evt = executed_events[0]
        self.assertEqual(evt.name, "skill.executed")
        self.assertEqual(evt.payload["skill_name"], "mock_echo")
        self.assertEqual(evt.payload["command"], "echo: Hello J.A.R.V.I.S")
        self.assertEqual(evt.payload["result"], "Hello J.A.R.V.I.S")
        self.assertGreater(evt.payload["duration"], 0.0)

    def test_execute_specific_skill_by_name(self) -> None:
        """Verify targeted execution with skill_name."""
        skill = MockEchoSkill()
        self.manager.register(skill)

        result = self.manager.execute("echo: targeted", skill_name="mock_echo")
        self.assertEqual(result, "targeted")

    def test_execute_unknown_command_raises_skill_not_found(self) -> None:
        """Verify SkillNotFoundError when no registered skill can handle the command."""
        skill = MockEchoSkill()
        self.manager.register(skill)

        with self.assertRaises(SkillNotFoundError):
            self.manager.execute("play jazz music")

    def test_execute_disabled_skill_raises_execution_error(self) -> None:
        """Verify SkillExecutionError when attempting to execute a disabled skill."""
        skill = MockEchoSkill()
        self.manager.register(skill)
        self.manager.disable_skill("mock_echo")

        with self.assertRaises(SkillExecutionError):
            self.manager.execute("echo: test", skill_name="mock_echo")

    def test_execute_failure_publishes_failed_event_and_raises(self) -> None:
        """Verify skill execution exception publishes skill.failed and raises SkillExecutionError."""
        failed_events: list[Event] = []
        self.test_bus.subscribe("skill.failed", lambda e: failed_events.append(e))

        skill = MockFailingSkill()
        self.manager.register(skill)

        with self.assertRaises(SkillExecutionError) as ctx:
            self.manager.execute("fail")

        self.assertIn("Simulated skill execution error.", str(ctx.exception))
        self.assertEqual(len(failed_events), 1)
        evt = failed_events[0]
        self.assertEqual(evt.name, "skill.failed")
        self.assertEqual(evt.payload["skill_name"], "mock_failing")
        self.assertEqual(evt.payload["exception_type"], "ValueError")

    def test_registration_publishes_event(self) -> None:
        """Verify registering a skill publishes skill.registered with metadata payload."""
        reg_events: list[Event] = []
        self.test_bus.subscribe("skill.registered", lambda e: reg_events.append(e))

        skill = MockEchoSkill()
        self.manager.register(skill)

        self.assertEqual(len(reg_events), 1)
        self.assertEqual(reg_events[0].payload["name"], "mock_echo")
        self.assertEqual(reg_events[0].payload["priority"], 100)
        self.assertEqual(reg_events[0].payload["tags"], ["test", "echo"])

    def test_async_execute(self) -> None:
        """Verify execute_async executes coroutine skills cleanly."""
        skill = MockAsyncSkill()
        self.manager.register(skill)

        async def _run() -> Any:
            return await self.manager.execute_async("run_async")

        result = asyncio.run(_run())
        self.assertEqual(result, "async_success")

    def test_thread_safety_concurrent_registration_and_execution(self) -> None:
        """Verify thread-safety during concurrent registrations and executions."""
        num_skills = 20
        skills = [
            MockEchoSkill(name=f"worker_{i}", priority=i * 5)
            for i in range(num_skills)
        ]

        def _register_worker(s: MockEchoSkill) -> None:
            self.manager.register(s)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_register_worker, s) for s in skills]
            for f in as_completed(futures):
                f.result()

        self.assertEqual(len(self.manager), num_skills)

        # Concurrently execute commands
        def _exec_worker(idx: int) -> Any:
            return self.manager.execute({"action": "echo", "payload": f"worker_{idx}"})

        with ThreadPoolExecutor(max_workers=8) as executor:
            exec_futures = [executor.submit(_exec_worker, i) for i in range(50)]
            results = [f.result() for f in as_completed(exec_futures)]

        self.assertEqual(len(results), 50)

    def test_clear_method(self) -> None:
        """Verify clear() shuts down and unregisters all skills."""
        s1 = MockEchoSkill(name="s1")
        s2 = MockEchoSkill(name="s2")
        self.manager.register(s1)
        self.manager.register(s2)

        self.assertEqual(len(self.manager), 2)
        self.manager.clear()
        self.assertEqual(len(self.manager), 0)
        self.assertTrue(s1.shutdown_called)
        self.assertTrue(s2.shutdown_called)

    def test_get_skills_by_tag_and_permission(self) -> None:
        """Verify querying skills by tag and required permission."""
        s1 = MockEchoSkill(name="s1", tags=["network", "api"], permissions={"net_access"})
        s2 = MockEchoSkill(name="s2", tags=["network", "db"], permissions={"db_access"})
        s3 = MockEchoSkill(name="s3", tags=["ui"], permissions={"net_access"}, enabled=False)

        self.manager.register(s1)
        self.manager.register(s2)
        self.manager.register(s3)

        by_tag = self.manager.get_skills_by_tag("NETWORK")
        self.assertEqual([s.name for s in by_tag], ["s1", "s2"])

        by_tag_enabled = self.manager.get_skills_by_tag("UI", enabled_only=True)
        self.assertEqual(by_tag_enabled, [])

        by_perm = self.manager.get_skills_by_permission("net_access")
        self.assertEqual([s.name for s in by_perm], ["s1", "s3"])

        by_perm_enabled = self.manager.get_skills_by_permission("net_access", enabled_only=True)
        self.assertEqual([s.name for s in by_perm_enabled], ["s1"])

    def test_find_skills(self) -> None:
        """Verify find_skills with multi-criteria filtering."""
        s1 = MockEchoSkill(name="web_search", description="Search Google online", tags=["web"], permissions={"net"})
        s2 = MockEchoSkill(name="local_find", description="Find files on disk", tags=["disk"], permissions={"fs"})

        self.manager.register(s1)
        self.manager.register(s2)

        # By query in description
        res = self.manager.find_skills(query="online")
        self.assertEqual([s.name for s in res], ["web_search"])

        # By query in name
        res = self.manager.find_skills(query="local")
        self.assertEqual([s.name for s in res], ["local_find"])

        # By tag and permission
        res = self.manager.find_skills(tag="web", permission="net")
        self.assertEqual([s.name for s in res], ["web_search"])

        res_none = self.manager.find_skills(tag="web", permission="fs")
        self.assertEqual(res_none, [])

    def test_manager_iter(self) -> None:
        """Verify iterating over SkillManager yields skills in priority order."""
        s1 = MockEchoSkill(name="s1", priority=10)
        s2 = MockEchoSkill(name="s2", priority=50)
        self.manager.register(s1)
        self.manager.register(s2)

        names = [s.name for s in self.manager]
        self.assertEqual(names, ["s2", "s1"])

    def test_discover_from_filesystem_directory(self) -> None:
        """Verify dynamic discovery of skills from a directory."""
        discovered_events: list[Event] = []
        self.test_bus.subscribe("skills.discovered", lambda e: discovered_events.append(e))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill1_code = '''
from app.skills.base import BaseSkill
from typing import Any

class DiscoveredSkillOne(BaseSkill):
    name = "discovered_one"
    description = "Dynamically discovered skill 1"
    priority = 120
    tags = ["discovered"]

    def can_handle(self, command: Any) -> bool:
        return command == "disc1"

    def execute(self, command: Any) -> str:
        return "result_one"
'''
            skill2_code = '''
from app.skills.base import BaseSkill
from typing import Any

class DiscoveredSkillTwo(BaseSkill):
    name = "discovered_two"
    priority = 80

    def can_handle(self, command: Any) -> bool:
        return command == "disc2"

    def execute(self, command: Any) -> str:
        return "result_two"

class NonSkillHelper:
    pass
'''
            (temp_path / "skill_one.py").write_text(skill1_code, encoding="utf-8")
            (temp_path / "skill_two.py").write_text(skill2_code, encoding="utf-8")
            (temp_path / "ignored.txt").write_text("not python", encoding="utf-8")

            discovered = self.manager.discover(temp_path)
            self.assertIn("discovered_one", discovered)
            self.assertIn("discovered_two", discovered)
            self.assertEqual(len(discovered), 2)

            self.assertTrue(self.manager.has_skill("discovered_one"))
            self.assertTrue(self.manager.has_skill("discovered_two"))

            result = self.manager.execute("disc1")
            self.assertEqual(result, "result_one")

            self.assertEqual(len(discovered_events), 1)
            self.assertEqual(discovered_events[0].name, "skills.discovered")
            self.assertEqual(discovered_events[0].payload["count"], 2)

    def test_discover_skills_alias_and_invalid_path(self) -> None:
        """Verify discover_skills alias and SkillDiscoveryError on missing module."""
        with self.assertRaises(SkillDiscoveryError):
            self.manager.discover_skills("non_existent_package_xyz_123")


if __name__ == "__main__":
    unittest.main()
