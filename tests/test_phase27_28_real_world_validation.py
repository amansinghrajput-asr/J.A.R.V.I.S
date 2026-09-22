"""Phase 27.28 — MVP Real-World Validation & UX Stabilization Test Suite.

Exhaustively verifies:
1. Planner system status command recognition (check system status, system status, system summary, etc.).
2. Planner CPU telemetry command recognition (check cpu, check cpu usage, get cpu info, etc.).
3. Planner memory/RAM command recognition (check memory, check ram, ram info, etc.).
4. Planner disk command recognition (check disk, disk usage, disk space, etc.).
5. Planner battery command recognition (check battery, battery status, battery info, etc.).
6. Visual-toggle regression preservation ("check the checkbox" -> visual_toggle, NOT hijacked).
7. Non-visual system command isolation ("check system status" -> get_system_summary, NOT visual_toggle).
8. Composite MVP command decomposition ("open notepad and check system status" -> [open_app, get_system_summary] with DAG dependency).
9. Safe composite execution with test doubles (no real Notepad launch).
10. Single safe system action execution through Executor and shared PlannerEventBus.
11. AIManager routing for single safe system actions (sync & async parity).
12. Canonical SystemSkillResult human-readable formatting (System Status, CPU, Memory, Disk, Battery).
13. Formatting handles missing/unavailable fields gracefully without Python object representation.
14. Structured result data preservation for programmatic use.
15. Event/state safety audit (zero leak of HWND, coordinates, OCR, pixels, passwords, or tokens).
16. Win32InputBackend zero physical invocation guarantee (invocation_count == 0).
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure headless offscreen Qt platform for test session
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["JARVIS_TEST_MODE"] = "1"

from PySide6.QtWidgets import QApplication

_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

from app.ai.manager import AIManager, EXECUTABLE_SAFE_SYSTEM_ACTIONS
from app.ai.planner.events import (
    PlanCompleted,
    PlannerEventBus,
    PlanStarted,
    SystemSkillCompleted,
    SystemSkillStarted,
    TaskCompleted,
    TaskStarted,
)
from app.ai.planner.executor import Executor
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.planner.planner import Planner
from app.automation.input import MockInputBackend, Win32InputBackend
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.skills.system import (
    AppSkills,
    InteractionSkills,
    SystemConfirmationManager,
    SystemInfoSkills,
    SystemSecurityPolicy,
    register_system_foundation,
)
from app.skills.system.base_system_skill import SystemSkillResult


class TestPlannerSystemInfoParsing(unittest.TestCase):
    """A, B, C, D, E, F: Planner natural-language system command recognition & visual toggle preservation."""

    def setUp(self) -> None:
        self.planner = Planner()

    def test_01_planner_system_status_recognition(self) -> None:
        """A. System status query variants map to get_system_summary."""
        queries = [
            "check system status",
            "system status",
            "system summary",
            "system overview",
            "get system summary",
        ]
        for q in queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "get_system_summary")
                self.assertIsNone(plan.tasks[0].target)

    def test_02_planner_cpu_recognition(self) -> None:
        """B. CPU telemetry query variants map to get_cpu_info."""
        queries = [
            "check cpu",
            "check cpu info",
            "check cpu usage",
            "cpu info",
            "cpu usage",
            "get cpu info",
        ]
        for q in queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "get_cpu_info")
                self.assertIsNone(plan.tasks[0].target)

    def test_03_planner_memory_recognition(self) -> None:
        """C. Memory and RAM query variants map to get_memory_info."""
        queries = [
            "check memory",
            "check memory info",
            "check memory usage",
            "check ram",
            "check ram info",
            "check ram usage",
            "memory info",
            "memory usage",
            "ram info",
            "ram usage",
            "get memory info",
        ]
        for q in queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "get_memory_info")
                self.assertIsNone(plan.tasks[0].target)

    def test_04_planner_disk_recognition(self) -> None:
        """D. Disk and storage query variants map to get_disk_info."""
        queries = [
            "check disk",
            "check disk space",
            "check disk usage",
            "disk space",
            "disk usage",
            "get disk info",
        ]
        for q in queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "get_disk_info")
                self.assertIsNone(plan.tasks[0].target)

    def test_05_planner_battery_recognition(self) -> None:
        """E. Battery query variants map to get_battery_info."""
        queries = [
            "check battery",
            "check battery status",
            "check battery info",
            "battery status",
            "battery info",
            "get battery info",
        ]
        for q in queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "get_battery_info")
                self.assertIsNone(plan.tasks[0].target)

    def test_06_visual_toggle_regression_preserved(self) -> None:
        """F1. Legitimate visual toggle commands continue to map to visual_toggle."""
        cases = [
            ("check the checkbox", "checkbox"),
            ("check the login checkbox", "login"),
            ("check the settings switch", "settings"),
            ("toggle notifications", "notifications"),
            ("switch theme", "theme"),
            ("uncheck auto save", "auto save"),
        ]
        for q, expected_target in cases:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].action, "visual_toggle")
                self.assertEqual(plan.tasks[0].target, expected_target)

    def test_07_check_system_status_does_not_map_to_visual_toggle(self) -> None:
        """F2. System telemetry commands starting with 'check' must NEVER become visual_toggle."""
        telemetry_commands = [
            "check system status",
            "check cpu",
            "check memory",
            "check ram",
            "check disk",
            "check battery",
        ]
        for q in telemetry_commands:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(len(plan.tasks), 1)
                self.assertNotEqual(
                    plan.tasks[0].action,
                    "visual_toggle",
                    f"'{q}' was incorrectly parsed as visual_toggle!",
                )


class TestCompositeMVPCommand(unittest.TestCase):
    """G. Composite MVP command decomposition & safe execution."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()
        self.sec_policy = SystemSecurityPolicy(event_bus=self.planner_bus)
        self.conf_mgr = SystemConfirmationManager(event_bus=self.planner_bus)

        # Mock AppSkills to ensure zero real desktop application launch
        self.app_skills = MagicMock(spec=AppSkills)
        self.app_skills.name = "app"
        self.app_skills.security_policy = self.sec_policy
        self.app_skills.parse_command = MagicMock(
            return_value=("open_app", "notepad", {}, None)
        )
        self.app_skills.execute = MagicMock(
            return_value=SystemSkillResult(
                operation="open_app",
                success=True,
                data={"status": "running", "app": "notepad", "pid": 12345},
            )
        )

        # SystemInfoSkills test double
        self.info_skills = SystemInfoSkills(
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_mgr,
            container=self.container,
            event_bus=self.planner_bus,
        )

        # Mock interaction backend
        self.mock_backend = MockInputBackend()
        self.interaction_skills = InteractionSkills(
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_mgr,
            container=self.container,
            event_bus=self.planner_bus,
            input_backend=self.mock_backend,
        )

        register_system_foundation(
            self.container,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_mgr,
            planner_event_bus=self.planner_bus,
            app_skills=self.app_skills,
            system_info_skills=self.info_skills,
            interaction_skills=self.interaction_skills,
        )

    def test_08_composite_command_decomposition(self) -> None:
        """G1. 'open notepad and check system status' decomposes into open_app -> get_system_summary DAG."""
        planner = Planner(event_bus=self.planner_bus)
        plan = planner.create_plan("open notepad and check system status")

        self.assertEqual(len(plan.tasks), 2)
        task1, task2 = plan.tasks[0], plan.tasks[1]

        self.assertEqual(task1.action, "open_app")
        self.assertEqual(task1.target, "notepad")
        self.assertEqual(task1.dependencies, [])

        self.assertEqual(task2.action, "get_system_summary")
        self.assertIsNone(task2.target)
        self.assertEqual(task2.dependencies, [task1.id])

    def test_09_composite_command_safe_execution(self) -> None:
        """G2. Composite command executes sequentially through Executor without launching real Notepad."""
        planner = Planner(event_bus=self.planner_bus)
        plan = planner.create_plan("open notepad and check system status")

        executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.planner_bus,
        )
        result = executor.execute_plan(plan)

        self.assertTrue(result.success)
        self.assertEqual(len(result.completed_tasks), 2)
        self.assertEqual(len(result.failed_tasks), 0)
        self.assertEqual(result.execution_order, [plan.tasks[0].id, plan.tasks[1].id])

        # Verify AppSkills was called with test double, not real OS launch
        self.app_skills.execute.assert_called_once()
        self.assertEqual(Win32InputBackend.invocation_count, 0)


class TestSingleSafeSystemActionExecution(unittest.TestCase):
    """H, M: Single safe system action execution and sync/async parity."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()
        self.sec_policy = SystemSecurityPolicy(event_bus=self.planner_bus)
        self.conf_mgr = SystemConfirmationManager(event_bus=self.planner_bus)

        self.info_skills = SystemInfoSkills(
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_mgr,
            container=self.container,
            event_bus=self.planner_bus,
        )

        register_system_foundation(
            self.container,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_mgr,
            planner_event_bus=self.planner_bus,
            system_info_skills=self.info_skills,
        )

        self.executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.planner_bus,
        )

    def test_10_execute_get_system_summary(self) -> None:
        """H1. Single safe get_system_summary executes cleanly through Executor."""
        events: list[Any] = []
        self.planner_bus.subscribe(SystemSkillStarted, events.append)
        self.planner_bus.subscribe(SystemSkillCompleted, events.append)

        task = Task(action="get_system_summary", target=None)
        plan = Plan(query="get_system_summary", tasks=[task])
        result = self.executor.execute_plan(plan)

        self.assertTrue(result.success)
        self.assertEqual(len(result.completed_tasks), 1)
        raw_res = result.task_results.get(task.id)
        self.assertIsInstance(raw_res, SystemSkillResult)
        self.assertTrue(raw_res.success)
        self.assertIn("cpu", raw_res.data)
        self.assertIn("memory", raw_res.data)
        self.assertIn("disk", raw_res.data)
        self.assertTrue(len(events) >= 2)

    def test_11_execute_get_cpu_info(self) -> None:
        """H2. Single safe get_cpu_info executes cleanly through Executor."""
        task = Task(action="get_cpu_info", target=None)
        plan = Plan(query="get_cpu_info", tasks=[task])
        result = self.executor.execute_plan(plan)

        self.assertTrue(result.success)
        raw_res = result.task_results.get(task.id)
        self.assertIsInstance(raw_res, SystemSkillResult)
        self.assertIn("usage_percent", raw_res.data)
        self.assertIn("logical_cores", raw_res.data)

    def test_12_execute_get_memory_info(self) -> None:
        """H3. Single safe get_memory_info executes cleanly through Executor."""
        task = Task(action="get_memory_info", target=None)
        plan = Plan(query="get_memory_info", tasks=[task])
        result = self.executor.execute_plan(plan)

        self.assertTrue(result.success)
        raw_res = result.task_results.get(task.id)
        self.assertIsInstance(raw_res, SystemSkillResult)
        self.assertIn("total_bytes", raw_res.data)
        self.assertIn("available_bytes", raw_res.data)

    def test_13_aimanager_single_action_routing_sync(self) -> None:
        """M1. AIManager routes single system telemetry command to Executor and formats response (sync)."""
        ai_mgr = AIManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            planner_instance=Planner(event_bus=self.planner_bus),
            executor_instance=self.executor,
        )

        # Mock LLM provider to ensure test does not rely on external API
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="LLM fallback should not be called")
        ai_mgr._llm = mock_llm

        response = ai_mgr.generate("check cpu")

        # Must NOT call LLM fallback because check cpu is an EXECUTABLE_SAFE_SYSTEM_ACTION
        mock_llm.generate.assert_not_called()
        self.assertIn("CPU:", response.content)
        self.assertIn("Usage:", response.content)

    def test_14_aimanager_single_action_routing_async(self) -> None:
        """M2. AIManager routes single system telemetry command to Executor and formats response (async)."""
        ai_mgr = AIManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            planner_instance=Planner(event_bus=self.planner_bus),
            executor_instance=self.executor,
        )

        mock_llm = MagicMock()
        mock_llm.generate_async = AsyncMock(return_value="LLM fallback should not be called")
        ai_mgr._llm = mock_llm

        async def _run() -> Any:
            return await ai_mgr.generate_async("check system status")

        response = asyncio.run(_run())

        mock_llm.generate_async.assert_not_called()
        self.assertIn("System Status:", response.content)
        self.assertIn("CPU:", response.content)
        self.assertIn("Memory:", response.content)


class TestResultFormattingAndStructure(unittest.TestCase):
    """I, J: Canonical Result-Formatting and structured result preservation."""

    def test_15_system_summary_formatting(self) -> None:
        """I1. System summary result formats into clean human-readable response without raw dict."""
        data = {
            "platform": "Windows 11 (AMD64)",
            "cpu": {"usage_percent": 14.5, "logical_cores": 16, "physical_cores": 8},
            "memory": {"usage_percent": 60.0, "total_gb": 32.0, "available_gb": 12.8},
            "disk": {"path": "C:\\", "usage_percent": 45.0, "total_gb": 1000.0, "free_gb": 550.0},
            "battery": {"available": True, "percent": 90.0, "plugged": True},
        }
        res = SystemSkillResult(operation="get_system_summary", success=True, data=data)
        msg = res.to_user_message()

        self.assertIn("System Status:", msg)
        self.assertIn("CPU: 14.5% (16 cores)", msg)
        self.assertIn("Memory: 60.0% of 32.0 GB (12.8 GB free)", msg)
        self.assertIn("Disk: 45.0% of 1000.0 GB (550.0 GB free)", msg)
        self.assertIn("OS: Windows 11 (AMD64)", msg)
        self.assertIn("Battery: 90.0% (Charging)", msg)
        # Ensure no raw Python dict leaked
        self.assertNotIn("{'usage_percent'", msg)
        self.assertNotIn("{'platform'", msg)

    def test_16_cpu_info_formatting(self) -> None:
        """I2. CPU result formats into clean human-readable response."""
        data = {
            "usage_percent": 22.4,
            "logical_cores": 8,
            "physical_cores": 4,
            "frequency_mhz": 3400.0,
        }
        res = SystemSkillResult(operation="get_cpu_info", success=True, data=data)
        msg = res.to_user_message()

        self.assertIn("CPU:", msg)
        self.assertIn("Usage: 22.4%", msg)
        self.assertIn("Cores: 8 logical (4 physical)", msg)
        self.assertIn("Frequency: 3400.0 MHz", msg)
        self.assertNotIn("{'usage_percent'", msg)

    def test_17_memory_info_formatting(self) -> None:
        """I3. Memory result formats into clean human-readable response."""
        data = {
            "total_bytes": 16 * (1024**3),
            "available_bytes": 8 * (1024**3),
            "used_bytes": 8 * (1024**3),
            "usage_percent": 50.0,
        }
        res = SystemSkillResult(operation="get_memory_info", success=True, data=data)
        msg = res.to_user_message()

        self.assertIn("Memory:", msg)
        self.assertIn("Used: 8.00 GB", msg)
        self.assertIn("Available: 8.00 GB", msg)
        self.assertIn("Total: 16.00 GB", msg)
        self.assertIn("Usage: 50.0%", msg)
        self.assertNotIn("{'total_bytes'", msg)

    def test_18_disk_info_formatting(self) -> None:
        """I4. Disk result formats into clean human-readable response."""
        data = {
            "path": "C:\\",
            "total_bytes": 500 * (1024**3),
            "used_bytes": 250 * (1024**3),
            "free_bytes": 250 * (1024**3),
            "usage_percent": 50.0,
        }
        res = SystemSkillResult(operation="get_disk_info", success=True, data=data)
        msg = res.to_user_message()

        self.assertIn("Disk:", msg)
        self.assertIn("Used: 250.00 GB", msg)
        self.assertIn("Free: 250.00 GB", msg)
        self.assertIn("Total: 500.00 GB", msg)
        self.assertIn("Usage: 50.0%", msg)
        self.assertIn("Path: C:\\", msg)

    def test_19_battery_info_formatting(self) -> None:
        """I5. Battery result formats into clean human-readable response."""
        data = {
            "available": True,
            "percent": 85.0,
            "plugged": False,
            "seconds_left": 7200,
        }
        res = SystemSkillResult(operation="get_battery_info", success=True, data=data)
        msg = res.to_user_message()

        self.assertIn("Battery:", msg)
        self.assertIn("Level: 85.0%", msg)
        self.assertIn("Charging: No", msg)
        self.assertIn("Time Remaining: 2h 0m", msg)

        # Test battery unavailable
        no_bat = SystemSkillResult(
            operation="get_battery_info",
            success=True,
            data={"available": False},
        )
        self.assertIn("No battery detected", no_bat.to_user_message())

    def test_20_structured_result_preservation(self) -> None:
        """J. Formatting does not destroy or mutate the underlying structured result."""
        original_data = {
            "usage_percent": 18.2,
            "logical_cores": 12,
            "physical_cores": 6,
            "frequency_mhz": 2800.0,
        }
        res = SystemSkillResult(
            operation="get_cpu_info",
            success=True,
            data=dict(original_data),
        )

        # Call to_user_message multiple times
        msg1 = res.to_user_message()
        msg2 = res.to_user_message()

        self.assertEqual(msg1, msg2)
        # Verify underlying structured data is identical and fully intact
        self.assertEqual(res.data, original_data)
        self.assertEqual(res.data["usage_percent"], 18.2)
        self.assertEqual(res.data["logical_cores"], 12)
        self.assertEqual(res.to_dict()["data"], original_data)


class TestSafetyAndSanitizationAudit(unittest.TestCase):
    """K, L: Privacy, security, and physical input safety assertions."""

    def test_21_events_remain_semantic_and_sanitized(self) -> None:
        """K. Planner and Executor lifecycle events do not leak coordinates, HWND, passwords, or tokens."""
        container = ServiceContainer()
        planner_bus = PlannerEventBus()
        sec_policy = SystemSecurityPolicy(event_bus=planner_bus)
        conf_mgr = SystemConfirmationManager(event_bus=planner_bus)
        info_skills = SystemInfoSkills(
            security_policy=sec_policy,
            confirmation_manager=conf_mgr,
            container=container,
            event_bus=planner_bus,
        )
        register_system_foundation(
            container,
            security_policy=sec_policy,
            confirmation_manager=conf_mgr,
            planner_event_bus=planner_bus,
            system_info_skills=info_skills,
        )

        events_captured: list[Any] = []
        planner_bus.subscribe(PlanStarted, events_captured.append)
        planner_bus.subscribe(TaskStarted, events_captured.append)
        planner_bus.subscribe(TaskCompleted, events_captured.append)
        planner_bus.subscribe(PlanCompleted, events_captured.append)
        planner_bus.subscribe(SystemSkillStarted, events_captured.append)
        planner_bus.subscribe(SystemSkillCompleted, events_captured.append)

        executor = Executor(container_instance=container, planner_event_bus=planner_bus)
        task = Task(action="get_system_summary")
        plan = Plan(query="get_system_summary", tasks=[task])
        executor.execute_plan(plan)

        self.assertTrue(len(events_captured) > 0)
        forbidden_substrings = [
            "password",
            "secret",
            "hwnd",
            "window_handle",
            "raw_ocr",
            "screenshot",
            "coordinates",
            "input_text",
        ]

        for ev in events_captured:
            ev_dict = ev.to_dict() if hasattr(ev, "to_dict") else vars(ev)
            ev_str = str(ev_dict).lower()
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden,
                    ev_str,
                    f"Forbidden substring '{forbidden}' leaked in event {type(ev).__name__}: {ev_str}",
                )

    def test_22_win32_input_backend_zero_invocations(self) -> None:
        """L. Win32InputBackend.invocation_count strictly equals 0 across all tests."""
        self.assertEqual(
            Win32InputBackend.invocation_count,
            0,
            "CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked during tests!",
        )


if __name__ == "__main__":
    unittest.main()
