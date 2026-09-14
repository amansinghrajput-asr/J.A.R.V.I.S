"""Unit tests for Phase 27.6 Planner Executor integration with VisionSkills.

Verifies deterministic alias resolution, container resolution, parameter preservation,
and execution adapters for visual actions ('ask_screen' and 'verify_screen_state').
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.ai.planner.events import PlannerEventBus
from app.ai.planner.executor import Executor
from app.ai.planner.models import Plan, Task, TaskStatus
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.skills.manager import SkillManager
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult


class FakeVisionSkill(BaseSystemSkill):
    """Minimal fake VisionSkill satisfying BaseSystemSkill contract."""

    name: str = "vision"
    priority: int = 52
    permissions: set[str] = {"system:read", "vision:capture"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def can_handle(self, command) -> bool:
        return True

    def execute(self, command) -> SystemSkillResult:
        self.last_command = command
        op, target, params, _ = self.parse_command(command)
        if op == "ask_screen":
            return SystemSkillResult(
                success=True,
                operation="ask_screen",
                data={"answer": "The submit button is blue.", "question": params.get("question")},
            )
        if op == "verify_screen_state":
            return SystemSkillResult(
                success=True,
                operation="verify_screen_state",
                data={"verified": True, "status": "verified", "reason": "Download complete."},
            )
        return SystemSkillResult(success=True, operation=op, data={})


class TestPlannerExecutorVisionIntegration(unittest.TestCase):
    """Test Executor resolution and invocation of vision actions."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()
        self.skill_manager = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )
        self.fake_vision = FakeVisionSkill(container=self.container)
        self.skill_manager.register(self.fake_vision)

        self.executor = Executor(
            container_instance=self.container,
            skill_manager_instance=self.skill_manager,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

    def test_executor_alias_map_resolves_ask_screen(self) -> None:
        """Verify ask_screen resolves to 'vision' via alias_map."""
        task = Task(id="t1", action="ask_screen", parameters={"question": "What is visible?"})
        handler = self.executor._resolve_from_skill_manager("ask_screen", task=task)
        self.assertIsNotNone(handler)

    def test_executor_alias_map_resolves_verify_screen_state(self) -> None:
        """Verify verify_screen_state resolves to 'vision' via alias_map."""
        task = Task(id="t2", action="verify_screen_state", parameters={"condition": "Build passes"})
        handler = self.executor._resolve_from_skill_manager("verify_screen_state", task=task)
        self.assertIsNotNone(handler)

    def test_executor_resolves_existing_vision_aliases(self) -> None:
        """Verify previous vision aliases continue to resolve through alias_map."""
        for action in ("capture_screen", "read_screen_text", "explain_active_window", "diagnose_screen_error"):
            task = Task(id=f"t_{action}", action=action)
            handler = self.executor._resolve_from_skill_manager(action, task=task)
            self.assertIsNotNone(handler, f"Failed to resolve existing alias: {action}")

    def test_executor_container_resolution_candidate_keys(self) -> None:
        """Verify _resolve_from_container resolves vision actions when registered in container."""
        self.container.register_singleton("vision", self.fake_vision)
        executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )
        for action in ("ask_screen", "verify_screen_state", "capture_screen", "explain_active_window"):
            handler = executor._resolve_from_container(action)
            self.assertIsNotNone(handler, f"Failed container candidate resolution for: {action}")

    def test_executor_invokes_ask_screen_adapter_with_structured_payload(self) -> None:
        """Verify Executor executes ask_screen and passes parameters in BaseSystemSkill cmd."""
        task = Task(
            id="task-vqa-1",
            action="ask_screen",
            target="active_window",
            parameters={"question": "What color is the button?", "reuse_cache": True},
        )
        plan = Plan(
            id="plan-1",
            query="Check the button color",
            tasks=[task],
        )

        handler = self.executor.resolve_handler(task)
        self.assertIsNotNone(handler)
        result = handler(task)
        self.assertTrue(result.success)
        self.assertIsNotNone(self.fake_vision.last_command)
        self.assertEqual(self.fake_vision.last_command["action"], "ask_screen")
        self.assertEqual(self.fake_vision.last_command["parameters"]["question"], "What color is the button?")

    def test_executor_invokes_verify_screen_state_adapter(self) -> None:
        """Verify Executor executes verify_screen_state task."""
        task = Task(
            id="task-verify-1",
            action="verify_screen_state",
            parameters={"condition": "Download complete"},
        )
        plan = Plan(
            id="plan-1",
            query="Verify download",
            tasks=[task],
        )

        handler = self.executor.resolve_handler(task)
        self.assertIsNotNone(handler)
        result = handler(task)
        self.assertTrue(result.success)
        self.assertIsNotNone(self.fake_vision.last_command)
        self.assertEqual(self.fake_vision.last_command["action"], "verify_screen_state")
        self.assertEqual(self.fake_vision.last_command["parameters"]["condition"], "Download complete")


if __name__ == "__main__":
    unittest.main()
