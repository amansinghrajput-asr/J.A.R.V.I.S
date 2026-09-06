"""Comprehensive unit tests for the SystemSkill and Command Router integration.

Tests intent classification, telemetry reporting, application management,
priority routing over AISkill (-100 vs 50), and automatic skill discovery.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.application import JarvisApplication
from app.automation.guardrails import SecurityGuard
from app.automation.models import (
    BatteryMetrics,
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    ProcessInfo,
    SecurityBlockedError,
    SystemTelemetry,
)
from app.automation.system import SystemMonitor
from app.automation.windows import WindowManager
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.router.router import CommandRouter
from app.skills.ai_skill import AISkill, DEFAULT_AI_SKILL_PRIORITY
from app.skills.manager import SkillManager
from app.skills.system_skill import (
    DEFAULT_SYSTEM_SKILL_NAME,
    DEFAULT_SYSTEM_SKILL_PRIORITY,
    SystemSkill,
)


class TestSystemSkillBasics(unittest.TestCase):
    """Test SystemSkill metadata, priority, and default initialization."""

    def test_default_metadata(self) -> None:
        skill = SystemSkill()
        self.assertEqual(skill.name, DEFAULT_SYSTEM_SKILL_NAME)
        self.assertEqual(skill.priority, 50)
        self.assertEqual(skill.priority, DEFAULT_SYSTEM_SKILL_PRIORITY)
        self.assertTrue(skill.enabled)
        self.assertEqual(skill.version, "1.0.0")

    def test_priority_relative_to_ai_skill(self) -> None:
        """Ensure SystemSkill (50) is higher priority than AISkill (-100)."""
        sys_skill = SystemSkill()
        ai = AISkill()
        self.assertEqual(ai.priority, -100)
        self.assertEqual(DEFAULT_AI_SKILL_PRIORITY, -100)
        self.assertGreater(sys_skill.priority, ai.priority)


class TestSystemSkillIntentMatching(unittest.TestCase):
    """Test can_handle() intent classification."""

    def setUp(self) -> None:
        self.skill = SystemSkill()

    def test_can_handle_resource_metrics(self) -> None:
        positive_queries = [
            "what is my cpu usage?",
            "cpu",
            "check cpu load",
            "how many cores do I have?",
            "ram usage",
            "how much memory is free?",
            "check disk space",
            "storage capacity",
            "battery percentage",
            "battery status",
        ]
        for q in positive_queries:
            self.assertTrue(self.skill.can_handle(q), f"Failed to match: {q}")

    def test_can_handle_system_status(self) -> None:
        for q in ("system status", "system info", "system diagnostics", "health check", "specs"):
            self.assertTrue(self.skill.can_handle(q), f"Failed to match: {q}")

    def test_can_handle_app_launch(self) -> None:
        for q in ("open notepad", "launch calculator", "start chrome", "run code"):
            self.assertTrue(self.skill.can_handle(q), f"Failed to match: {q}")

    def test_can_handle_processes(self) -> None:
        for q in ("top processes", "list processes", "running tasks", "kill process notepad"):
            self.assertTrue(self.skill.can_handle(q), f"Failed to match: {q}")

    def test_can_handle_rejects_conversational_queries(self) -> None:
        negative_queries = [
            "hello jarvis",
            "how are you today?",
            "tell me a joke",
            "what is the capital of France?",
            "explain quantum physics",
            "",
            None,
        ]
        for q in negative_queries:
            self.assertFalse(self.skill.can_handle(q), f"Incorrectly matched: {q}")


class TestSystemSkillExecution(unittest.TestCase):
    """Test skill command dispatching and response generation."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.monitor = MagicMock(spec=SystemMonitor)
        self.win_mgr = MagicMock(spec=WindowManager)
        self.guard = MagicMock(spec=SecurityGuard)

        self.skill = SystemSkill(
            system_monitor_instance=self.monitor,
            window_manager_instance=self.win_mgr,
            security_guard_instance=self.guard,
            container=self.container,
            event_bus=self.event_bus,
        )

    def test_execute_cpu_query(self) -> None:
        self.monitor.get_cpu_metrics.return_value = CpuMetrics(
            percent=14.5, logical_cores=8, physical_cores=4, frequency_mhz=3500.0
        )
        res = self.skill.execute("what is cpu usage?")
        self.assertIn("14.5%", res)
        self.assertIn("8 cores", res)
        self.assertEqual(self.skill.execution_count, 1)

    def test_execute_memory_query(self) -> None:
        self.monitor.get_memory_metrics.return_value = MemoryMetrics(
            total_gb=32.0, used_gb=16.0, free_gb=16.0, percent=50.0
        )
        res = self.skill.execute("check ram usage")
        self.assertIn("50.0%", res)
        self.assertIn("16.0 GB", res)
        self.assertIn("32.0 GB", res)

    def test_execute_disk_query(self) -> None:
        self.monitor.get_disk_metrics.return_value = DiskMetrics(
            total_gb=500.0, used_gb=200.0, free_gb=300.0, percent=40.0, mount_point="C:\\"
        )
        res = self.skill.execute("how much disk storage is free?")
        self.assertIn("40.0%", res)
        self.assertIn("300.0 GB free", res)

    def test_execute_battery_query(self) -> None:
        self.monitor.get_battery_metrics.return_value = BatteryMetrics(
            percent=95.0, power_plugged=True
        )
        res = self.skill.execute("battery level")
        self.assertIn("95%", res)
        self.assertIn("Plugged in", res)

    def test_execute_battery_none(self) -> None:
        self.monitor.get_battery_metrics.return_value = None
        res = self.skill.execute("battery")
        self.assertIn("No battery detected", res)

    def test_execute_system_status(self) -> None:
        self.monitor.collect_telemetry.return_value = SystemTelemetry(
            cpu=CpuMetrics(percent=10.0, logical_cores=4, physical_cores=4),
            memory=MemoryMetrics(total_gb=16.0, used_gb=8.0, free_gb=8.0, percent=50.0),
            disk=DiskMetrics(total_gb=500.0, used_gb=250.0, free_gb=250.0, percent=50.0),
            battery=None,
        )
        res = self.skill.execute("system status")
        self.assertIn("CPU: 10.0%", res)
        self.assertIn("RAM: 50.0%", res)

    def test_execute_top_processes(self) -> None:
        self.monitor.get_top_processes.return_value = [
            ProcessInfo(pid=1001, name="chrome.exe", cpu_percent=12.5, memory_percent=4.2),
            ProcessInfo(pid=1002, name="code.exe", cpu_percent=5.1, memory_percent=3.0),
        ]
        res = self.skill.execute("top processes")
        self.assertIn("chrome.exe", res)
        self.assertIn("PID 1001", res)
        self.assertIn("code.exe", res)

    def test_execute_open_app(self) -> None:
        self.win_mgr.launch_application.return_value = True
        res = self.skill.execute("open notepad")
        self.assertIn("launched notepad", res)
        self.win_mgr.launch_application.assert_called_once_with("notepad")

    def test_execute_kill_process(self) -> None:
        self.win_mgr.terminate_process.return_value = 1
        res = self.skill.execute("kill process notepad")
        self.assertIn("Successfully terminated 1 process(es)", res)
        self.win_mgr.terminate_process.assert_called_once_with("notepad")

    def test_execute_handles_security_blocked_error(self) -> None:
        self.win_mgr.terminate_process.side_effect = SecurityBlockedError("Protected process")
        res = self.skill.execute("kill process explorer.exe")
        self.assertIn("System operation failed", res)
        self.assertIn("Protected process", res)
        self.assertEqual(self.skill.error_count, 1)

    def test_execute_async(self) -> None:
        self.monitor.get_cpu_metrics.return_value = CpuMetrics(percent=20.0, logical_cores=4, physical_cores=4)
        res = asyncio.run(self.skill.execute_async("cpu"))
        self.assertIn("20.0%", res)

    def test_event_bus_publishing(self) -> None:
        events = []
        self.event_bus.subscribe("system.skill.started", lambda e: events.append(e))
        self.event_bus.subscribe("system.skill.completed", lambda e: events.append(e))

        self.monitor.get_cpu_metrics.return_value = CpuMetrics(percent=15.0, logical_cores=4, physical_cores=4)
        self.skill.execute("cpu")

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "system.skill.started")
        self.assertEqual(events[1].name, "system.skill.completed")


class TestSystemSkillRouterIntegration(unittest.TestCase):
    """Test priority routing between SystemSkill (priority 50) and AISkill (priority -100)."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.skill_manager = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )
        self.router = CommandRouter(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            skill_manager_instance=self.skill_manager,
            auto_register_in_container=True,
        )

        # Mock dependencies for skills
        self.system_monitor = MagicMock(spec=SystemMonitor)
        self.system_monitor.get_cpu_metrics.return_value = CpuMetrics(percent=22.2, logical_cores=8, physical_cores=4)
        self.system_skill = SystemSkill(
            system_monitor_instance=self.system_monitor,
            container=self.container,
            event_bus=self.event_bus,
        )

        self.ai_manager = MagicMock()
        ai_resp = MagicMock()
        ai_resp.content = "Conversational response from Gemini."
        ai_resp.__str__.return_value = "Conversational response from Gemini."
        self.ai_manager.generate.return_value = ai_resp
        self.ai_skill = AISkill(
            ai_manager_instance=self.ai_manager,
            container=self.container,
            event_bus=self.event_bus,
        )

        # Register both skills
        self.skill_manager.register(self.system_skill)
        self.skill_manager.register(self.ai_skill)

    def test_system_skill_intercepts_system_query(self) -> None:
        """A system query routes to SystemSkill (priority 50), NOT AISkill (priority -100)."""
        result = self.router.route("what is my cpu usage?")
        self.assertIn("22.2%", str(result))
        self.ai_manager.generate.assert_not_called()
        self.assertEqual(self.system_skill.execution_count, 1)

    def test_conversational_query_falls_back_to_ai_skill(self) -> None:
        """A conversational query bypasses SystemSkill and routes to AISkill fallback."""
        result = self.router.route("write a poem about space")
        self.ai_manager.generate.assert_called_once()
        self.assertEqual(result.content, "Conversational response from Gemini.")
        self.assertEqual(self.system_skill.execution_count, 0)


class TestJarvisApplicationAutoDiscovery(unittest.TestCase):
    """Test automatic discovery of SystemSkill in JarvisApplication."""

    def test_discover_skills_registers_system_and_ai(self) -> None:
        app = JarvisApplication(print_ready=False)
        self.assertTrue(app.skill_manager.has_skill("system"))
        self.assertTrue(app.skill_manager.has_skill("ai"))

        sys_skill = app.skill_manager.get("system")
        ai_skill = app.skill_manager.get("ai")

        self.assertIsNotNone(sys_skill)
        self.assertIsNotNone(ai_skill)
        self.assertEqual(sys_skill.priority, 50)
        self.assertEqual(ai_skill.priority, -100)


if __name__ == "__main__":
    unittest.main()
