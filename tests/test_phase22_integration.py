"""Phase 22.8 Integration Tests: Planner, Executor & Metacognition.

Validates:
A. Executor action resolution across all Phase 22 skill categories.
B. Structured parameter preservation for BaseSystemSkill adapters.
C. Legacy SystemSkill facade compatibility and conversational responses.
D. Metacognitive success telemetry (SkillEvolutionEngine and SemanticKnowledgeGraph).
E. Metacognitive failure telemetry.
F. Duplicate telemetry protection across direct event handling and trajectory completion.
G. Safety invariants (zero destructive execution).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.ai.planner.events import (
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillFailed,
)
from app.ai.planner.executor import Executor
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.controller import MetacognitiveController
from app.ai.planner.metacognition.evolution import (
    SkillEvolutionEngine,
    SkillStatus,
)
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.synthesizer import ToolSynthesizer
from app.ai.planner.models import Task, TaskStatus
from app.core.container import ServiceContainer
from app.skills.base import BaseSkill
from app.skills.manager import SkillManager
from app.skills.system.app_skills import AppSkills
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.browser_skills import BrowserSkills
from app.skills.system.file_skills import FileSkills
from app.skills.system.system_control_skills import SystemControlSkills
from app.skills.system.system_info_skills import SystemInfoSkills
from app.skills.system.window_skills import WindowSkills
from app.skills.system_skill import SystemSkill


class TestExecutorPhase22Resolution(unittest.TestCase):
    """Verify Executor._resolve_from_skill_manager resolves all Phase 22 actions."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.skill_manager = SkillManager(container_instance=self.container)

        # Register mock instances for each specialized Phase 22 skill category
        self.app_skills = MagicMock(spec=AppSkills)
        self.app_skills.name = "app"
        self.app_skills.execute = MagicMock(return_value=SystemSkillResult(operation="open_app", success=True))

        self.browser_skills = MagicMock(spec=BrowserSkills)
        self.browser_skills.name = "browser"
        self.browser_skills.execute = MagicMock(return_value=SystemSkillResult(operation="open_url", success=True))

        self.file_skills = MagicMock(spec=FileSkills)
        self.file_skills.name = "file"
        self.file_skills.execute = MagicMock(return_value=SystemSkillResult(operation="create_file", success=True))

        self.window_skills = MagicMock(spec=WindowSkills)
        self.window_skills.name = "window"
        self.window_skills.execute = MagicMock(return_value=SystemSkillResult(operation="list_windows", success=True))

        self.system_control_skills = MagicMock(spec=SystemControlSkills)
        self.system_control_skills.name = "system_control"
        self.system_control_skills.execute = MagicMock(return_value=SystemSkillResult(operation="get_volume", success=True))

        self.system_info_skills = MagicMock(spec=SystemInfoSkills)
        self.system_info_skills.name = "system_info"
        self.system_info_skills.execute = MagicMock(return_value=SystemSkillResult(operation="get_cpu_info", success=True))

        self.skill_manager.register(self.app_skills)
        self.skill_manager.register(self.browser_skills)
        self.skill_manager.register(self.file_skills)
        self.skill_manager.register(self.window_skills)
        self.skill_manager.register(self.system_control_skills)
        self.skill_manager.register(self.system_info_skills)

        self.executor = Executor(
            container_instance=self.container,
            skill_manager_instance=self.skill_manager,
            auto_register_in_container=False,
        )

    def test_app_skills_actions_resolution(self) -> None:
        for action in (
            "open_app",
            "launch_app",
            "close_app",
            "restart_app",
            "is_app_running",
            "list_running_apps",
        ):
            task = Task(action=action, target="notepad")
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.app_skills.execute.assert_called()

    def test_browser_skills_actions_resolution(self) -> None:
        for action in (
            "open_url",
            "browse_url",
            "open_link",
            "search_web",
            "web_search",
            "open_browser",
        ):
            task = Task(action=action, target="https://example.com")
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.browser_skills.execute.assert_called()

    def test_file_skills_actions_resolution(self) -> None:
        for action in (
            "create_folder",
            "create_file",
            "read_file",
            "list_directory",
            "rename_path",
            "move_path",
            "copy_path",
            "delete_path",
            "open_in_explorer",
        ):
            task = Task(action=action, target="C:\\test")
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.file_skills.execute.assert_called()

    def test_window_skills_actions_resolution(self) -> None:
        for action in (
            "list_windows",
            "get_active_window",
            "focus_window",
            "minimize_window",
            "maximize_window",
            "restore_window",
            "close_window",
        ):
            task = Task(action=action, target="notepad")
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.window_skills.execute.assert_called()

    def test_system_control_skills_actions_resolution(self) -> None:
        for action in (
            "get_volume",
            "set_volume",
            "volume_up",
            "volume_down",
            "mute_volume",
            "unmute_volume",
            "get_brightness",
            "set_brightness",
            "lock_workstation",
        ):
            task = Task(action=action, parameters={"level": 50})
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.system_control_skills.execute.assert_called()

    def test_system_info_skills_actions_resolution(self) -> None:
        for action in (
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
        ):
            task = Task(action=action)
            handler = self.executor.resolve_handler(task)
            self.assertIsNotNone(handler, f"Failed to resolve handler for {action}")
            handler(task)
            self.system_info_skills.execute.assert_called()


class TestExecutorStructuredParameterPreservation(unittest.TestCase):
    """Verify parameters survive Executor -> Skill adapter for BaseSystemSkill."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.skill_manager = SkillManager(container_instance=self.container)

        # Real BaseSystemSkill mock subclass
        class MockFileSkill(BaseSystemSkill):
            name = "file"
            def can_handle(self, command: Any) -> bool:
                return True
            def _execute_operation(self, op, target, params):
                return {"received_op": op, "received_target": target, "received_params": params}

        class MockSystemControlSkill(BaseSystemSkill):
            name = "system_control"
            def can_handle(self, command: Any) -> bool:
                return True
            def _execute_operation(self, op, target, params):
                return {"received_op": op, "received_target": target, "received_params": params}

        # Legacy non-BaseSystemSkill
        class MockLegacySkill(BaseSkill):
            name = "calc"
            def can_handle(self, command: Any) -> bool:
                return True
            def execute(self, cmd):
                return f"executed: {cmd}"

        self.mock_file = MockFileSkill()
        self.mock_sys_ctrl = MockSystemControlSkill()
        self.mock_legacy = MockLegacySkill()

        self.skill_manager.register(self.mock_file)
        self.skill_manager.register(self.mock_sys_ctrl)
        self.skill_manager.register(self.mock_legacy)

        self.executor = Executor(
            container_instance=self.container,
            skill_manager_instance=self.skill_manager,
            auto_register_in_container=False,
        )

    def test_create_file_preserves_content_parameter(self) -> None:
        task = Task(
            action="create_file",
            target="test.txt",
            parameters={"content": "abc"},
        )
        handler = self.executor.resolve_handler(task)
        self.assertIsNotNone(handler)

        result = handler(task)
        self.assertIsInstance(result, SystemSkillResult)
        self.assertTrue(result.success)
        self.assertEqual(result.data["received_op"], "create_file")
        self.assertEqual(result.data["received_target"], "test.txt")
        self.assertEqual(result.data["received_params"]["content"], "abc")
        self.assertEqual(result.data["received_params"]["task_id"], task.id)
        self.assertEqual(result.data["received_params"]["execution_id"], task.id)
        # Original task.parameters must NOT be mutated
        self.assertEqual(task.parameters, {"content": "abc"})

    def test_set_volume_preserves_level_parameter(self) -> None:
        task = Task(
            action="set_volume",
            target=None,
            parameters={"level": 50},
        )
        handler = self.executor.resolve_handler(task)
        self.assertIsNotNone(handler)

        result = handler(task)
        self.assertIsInstance(result, SystemSkillResult)
        self.assertTrue(result.success)
        self.assertEqual(result.data["received_op"], "set_volume")
        self.assertIsNone(result.data["received_target"])
        self.assertEqual(result.data["received_params"]["level"], 50)
        self.assertEqual(result.data["received_params"]["task_id"], task.id)
        self.assertEqual(result.data["received_params"]["execution_id"], task.id)
        # Original task.parameters must NOT be mutated
        self.assertEqual(task.parameters, {"level": 50})

    def test_legacy_skill_receives_formatted_string(self) -> None:
        task = Task(action="calculate", target="2 + 2")
        handler = self.executor.resolve_handler(task)
        self.assertIsNotNone(handler)

        result = handler(task)
        self.assertEqual(result, "executed: calculate 2 + 2")


class TestSystemSkillFacadeCompatibility(unittest.TestCase):
    """Verify legacy SystemSkill facade backward compatibility and conversational responses."""

    def setUp(self) -> None:
        self.container = ServiceContainer()

        self.mock_app_skills = MagicMock()
        self.mock_app_skills.name = "app"
        self.mock_app_skills.priority = 60
        self.mock_app_skills.execute.return_value = {"terminated_count": 1, "closed": True}

        self.mock_sys_info_skills = MagicMock()
        self.mock_sys_info_skills.name = "system_info"
        self.mock_sys_info_skills.priority = 60
        self.mock_sys_info_skills.execute.return_value = SystemSkillResult(
            operation="get_cpu_info",
            success=True,
            data={
                "percent": 18.5,
                "logical_cores": 8,
                "physical_cores": 4,
                "frequency_mhz": 3200.0,
            },
        )

        self.container.register_singleton("app_skills", self.mock_app_skills)
        self.container.register_singleton("system_info_skills", self.mock_sys_info_skills)

        # Initialize SystemSkill without injecting custom window_manager/system_monitor
        self.facade = SystemSkill(container=self.container)

    def test_priorities_invariants(self) -> None:
        self.assertEqual(self.facade.priority, 50)
        self.assertEqual(AppSkills.priority, 60)
        self.assertEqual(BrowserSkills.priority, 60)
        self.assertEqual(FileSkills.priority, 60)
        self.assertEqual(WindowSkills.priority, 55)
        self.assertGreater(WindowSkills.priority, self.facade.priority)
        self.assertEqual(SystemControlSkills.priority, 60)
        self.assertEqual(SystemInfoSkills.priority, 60)

    def test_open_app_delegates_and_preserves_conversational_response(self) -> None:
        resp = self.facade.execute("open notepad")
        self.assertEqual(resp, "I have launched notepad for you.")
        self.mock_app_skills.execute.assert_called_once_with({"action": "open_app", "target": "notepad"})

    def test_kill_process_delegates_and_preserves_conversational_response(self) -> None:
        resp = self.facade.execute("kill process notepad")
        self.assertEqual(resp, "Successfully terminated 1 process(es) matching 'notepad'.")
        self.mock_app_skills.execute.assert_called_once_with({"action": "close_app", "target": "notepad"})

    def test_cpu_delegates_and_preserves_conversational_response(self) -> None:
        resp = self.facade.execute("what is cpu usage?")
        self.assertIn("CPU utilization is currently 18.5%", resp)
        self.assertIn("8 cores", resp)
        self.assertIn("3200 MHz", resp)
        self.mock_sys_info_skills.execute.assert_called_once_with({"action": "get_cpu_info"})

    def test_can_handle_intents_preserved(self) -> None:
        self.assertTrue(self.facade.can_handle("open notepad"))
        self.assertTrue(self.facade.can_handle("kill process chrome"))
        self.assertTrue(self.facade.can_handle("system status"))
        self.assertTrue(self.facade.can_handle("cpu"))
        self.assertTrue(self.facade.can_handle("ram usage"))
        self.assertTrue(self.facade.can_handle("disk space"))
        self.assertTrue(self.facade.can_handle("battery level"))


class TestMetacognitiveTelemetryIntegration(unittest.TestCase):
    """Verify MetacognitiveController observes SystemSkill events and updates Evolution Engine & KG."""

    def setUp(self) -> None:
        self.event_bus = PlannerEventBus()
        self.knowledge_graph = SemanticKnowledgeGraph()
        self.compiler = MagicMock(spec=MacroSkillCompiler)
        self.reflection = MagicMock(spec=CausalReflectionEngine)
        self.synthesizer = MagicMock(spec=ToolSynthesizer)

        self.evolution_engine = SkillEvolutionEngine(
            compiler=self.compiler,
            reflection_engine=self.reflection,
            synthesizer=self.synthesizer,
            event_bus=self.event_bus,
        )

        self.controller = MetacognitiveController(
            tool_synthesizer=self.synthesizer,
            reflection_engine=self.reflection,
            skill_compiler=self.compiler,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
            evolution_engine=self.evolution_engine,
        )

    def tearDown(self) -> None:
        self.controller.shutdown()

    def test_metacognitive_success_telemetry(self) -> None:
        event = SystemSkillCompleted(
            skill_name="file",
            operation="create_file",
            target="C:\\test\\file.txt",
            duration=0.045,
            success=True,
            execution_id="exec_succ_1",
        )
        self.event_bus.publish(event)

        # Verify SkillEvolutionEngine
        metrics = self.evolution_engine.get_metrics("file")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.failure_count, 0)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertAlmostEqual(metrics.average_latency, 45.0, places=1)

        # Verify SemanticKnowledgeGraph
        triplets = self.knowledge_graph.query(subject="file", predicate="executed_operation")
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0].object_, "create_file")

    def test_metacognitive_failure_telemetry(self) -> None:
        event = SystemSkillFailed(
            skill_name="app",
            operation="open_app",
            target="nonexistent_app",
            duration=0.015,
            error="Application not allowlisted",
            execution_id="exec_fail_1",
        )
        self.event_bus.publish(event)

        # Verify SkillEvolutionEngine
        metrics = self.evolution_engine.get_metrics("app")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.failure_count, 1)
        self.assertEqual(metrics.success_count, 0)
        self.assertEqual(metrics.consecutive_failures, 1)
        self.assertEqual(metrics.success_rate, 0.0)

        # Verify SemanticKnowledgeGraph
        triplets = self.knowledge_graph.query(subject="app", predicate="failed_operation")
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0].object_, "open_app")

    def test_duplicate_telemetry_protection_event_then_trajectory(self) -> None:
        # 1. Direct SystemSkillCompleted event
        event = SystemSkillCompleted(
            skill_name="browser",
            operation="open_url",
            target="https://example.com",
            duration=0.03,
            execution_id="task_exec_99",
            metadata={"task_id": "task_99"},
        )
        self.event_bus.publish(event)

        metrics_after_event = self.evolution_engine.get_metrics("browser")
        self.assertIsNotNone(metrics_after_event)
        self.assertEqual(metrics_after_event.invocation_count, 1)
        self.assertEqual(metrics_after_event.success_count, 1)

        # 2. Trajectory completion for the SAME execution
        trajectory = {
            "trajectory_id": "traj_99",
            "goal": "open url in browser",
            "success": True,
            "tasks": [
                {
                    "id": "task_99",
                    "action": "open_url",
                    "target": "https://example.com",
                    "skill_name": "browser",
                    "execution_id": "task_exec_99",
                }
            ],
            "duration_ms": 30.0,
        }
        self.controller.on_trajectory_completed(trajectory)

        # Verify not counted twice!
        metrics_after_traj = self.evolution_engine.get_metrics("browser")
        self.assertEqual(metrics_after_traj.invocation_count, 1)
        self.assertEqual(metrics_after_traj.success_count, 1)

    def test_duplicate_telemetry_protection_trajectory_then_event(self) -> None:
        # 1. Register skill
        self.evolution_engine.register_skill("window")
        self.assertEqual(self.evolution_engine.get_metrics("window").invocation_count, 0)

        # 2. Trajectory completion runs first
        trajectory = {
            "trajectory_id": "traj_100",
            "goal": "window",
            "success": True,
            "tasks": [
                {
                    "id": "task_100",
                    "action": "list_windows",
                    "target": None,
                    "skill_name": "window",
                    "execution_id": "task_exec_100",
                }
            ],
            "duration_ms": 20.0,
        }
        self.controller.on_trajectory_completed(trajectory)

        metrics_after_traj = self.evolution_engine.get_metrics("window")
        self.assertEqual(metrics_after_traj.invocation_count, 1)

        # 3. Direct event arrives after
        event = SystemSkillCompleted(
            skill_name="window",
            operation="list_windows",
            target=None,
            duration=0.02,
            execution_id="task_exec_100",
            metadata={"task_id": "task_100"},
        )
        self.event_bus.publish(event)

        # Verify not counted twice for window
        metrics = self.evolution_engine.get_metrics("window")
        self.assertEqual(metrics.invocation_count, 1)

    def test_different_executions_same_action_both_recorded(self) -> None:
        """Scenario F: distinct executions of the same action must BOTH be recorded."""
        self.evolution_engine.register_skill("app")

        event1 = SystemSkillCompleted(
            skill_name="app",
            operation="open_app",
            target="notepad",
            duration=0.03,
            execution_id="exec_app_1",
            metadata={"task_id": "task_app_1"},
        )
        self.event_bus.publish(event1)

        event2 = SystemSkillCompleted(
            skill_name="app",
            operation="open_app",
            target="calc",
            duration=0.04,
            execution_id="exec_app_2",
            metadata={"task_id": "task_app_2"},
        )
        self.event_bus.publish(event2)

        metrics = self.evolution_engine.get_metrics("app")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.invocation_count, 2)
        self.assertEqual(metrics.success_count, 2)

    def test_different_file_operations_same_action_both_recorded(self) -> None:
        """Scenario F (file): distinct file creations must BOTH be recorded."""
        self.evolution_engine.register_skill("file")

        event1 = SystemSkillCompleted(
            skill_name="file",
            operation="create_file",
            target="a.txt",
            duration=0.02,
            execution_id="exec_file_1",
            metadata={"task_id": "task_file_1"},
        )
        self.event_bus.publish(event1)

        event2 = SystemSkillCompleted(
            skill_name="file",
            operation="create_file",
            target="b.txt",
            duration=0.02,
            execution_id="exec_file_2",
            metadata={"task_id": "task_file_2"},
        )
        self.event_bus.publish(event2)

        metrics = self.evolution_engine.get_metrics("file")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.invocation_count, 2)
        self.assertEqual(metrics.success_count, 2)

    def test_missing_ids_both_recorded_fail_open(self) -> None:
        """Scenario E: events lacking correlation IDs must both be recorded (fail-open telemetry)."""
        self.evolution_engine.register_skill("browser")

        event1 = SystemSkillCompleted(
            skill_name="browser",
            operation="open_url",
            target="https://site1.com",
            duration=0.03,
            execution_id="",
            metadata={},
        )
        self.event_bus.publish(event1)

        event2 = SystemSkillCompleted(
            skill_name="browser",
            operation="open_url",
            target="https://site2.com",
            duration=0.03,
            execution_id="",
            metadata={},
        )
        self.event_bus.publish(event2)

        metrics = self.evolution_engine.get_metrics("browser")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.invocation_count, 2)
        self.assertEqual(metrics.success_count, 2)

    def test_cache_pruning_expired_entries(self) -> None:
        """Expired entries older than dedup_ttl_seconds must be pruned on next check."""
        import time
        now = time.time()
        self.controller._recorded_executions["old_key"] = now - 75.0
        self.controller._recorded_executions["recent_key"] = now - 10.0

        is_dup = self.controller._prune_and_check_dedup(["new_key"], now=now)
        self.assertFalse(is_dup)
        self.assertNotIn("old_key", self.controller._recorded_executions)
        self.assertIn("recent_key", self.controller._recorded_executions)
        self.assertIn("new_key", self.controller._recorded_executions)

    def test_cache_size_bound_enforced(self) -> None:
        """Deduplication cache size cannot exceed max_dedup_cache_size."""
        self.controller.max_dedup_cache_size = 5
        self.controller._recorded_executions.clear()

        for i in range(12):
            self.controller._prune_and_check_dedup([f"exec:key_{i}"])

        self.assertLessEqual(len(self.controller._recorded_executions), 5)
        # Most recent keys must be retained
        self.assertIn("exec:key_11", self.controller._recorded_executions)

    def test_executor_task_id_flows_to_system_skill_event(self) -> None:
        """Executor task ID must be passed into BaseSystemSkill and propagate to event bus."""
        published_events = []
        self.event_bus.subscribe(SystemSkillCompleted, lambda e: published_events.append(e))

        class MockAppSkillWithBus(BaseSystemSkill):
            name = "app"
            def can_handle(self, cmd: Any) -> bool:
                return True
            def _execute_operation(self, op, target, params):
                return {"status": "launched", "target": target}

        container = ServiceContainer()
        sm = SkillManager(container_instance=container)
        skill = MockAppSkillWithBus(event_bus=self.event_bus)
        sm.register(skill)

        executor = Executor(
            container_instance=container,
            skill_manager_instance=sm,
            planner_event_bus=self.event_bus,
            auto_register_in_container=False,
        )

        task = Task(
            id="task_flow_test_123",
            action="open_app",
            target="notepad",
            parameters={"custom_param": 42},
        )
        handler = executor.resolve_handler(task)
        self.assertIsNotNone(handler)
        handler(task)

        self.assertEqual(len(published_events), 1)
        ev = published_events[0]
        self.assertEqual(ev.execution_id, "task_flow_test_123")
        self.assertEqual(ev.metadata.get("task_id"), "task_flow_test_123")
        self.assertEqual(task.parameters, {"custom_param": 42})


if __name__ == "__main__":
    unittest.main()
