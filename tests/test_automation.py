"""Comprehensive unit tests for the OS Automation subsystem.

Tests data models, security guardrails, system telemetry monitoring,
and window/process management.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.automation.guardrails import CRITICAL_SYSTEM_PROCESSES, SecurityGuard
from app.automation.models import (
    AutomationError,
    BatteryMetrics,
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    ProcessInfo,
    ProcessOperationError,
    SecurityBlockedError,
    SystemTelemetry,
    WindowInfo,
)
from app.automation.system import EVENT_SYSTEM_TELEMETRY, SystemMonitor
from app.automation.windows import DEFAULT_APP_ALIASES, WindowManager
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus


class TestAutomationModels(unittest.TestCase):
    """Test telemetry models and dataclass formatting."""

    def test_telemetry_summary_with_battery(self) -> None:
        cpu = CpuMetrics(percent=25.5, logical_cores=8, physical_cores=4, frequency_mhz=3200.0)
        mem = MemoryMetrics(total_gb=16.0, used_gb=8.0, free_gb=8.0, percent=50.0)
        disk = DiskMetrics(total_gb=500.0, used_gb=250.0, free_gb=250.0, percent=50.0, mount_point="C:\\")
        bat = BatteryMetrics(percent=88.0, power_plugged=True, remaining_seconds=3600)

        telemetry = SystemTelemetry(cpu=cpu, memory=mem, disk=disk, battery=bat, platform="Windows 11")
        summary = telemetry.summary()

        self.assertIn("CPU: 25.5%", summary)
        self.assertIn("8 cores", summary)
        self.assertIn("RAM: 50.0%", summary)
        self.assertIn("Disk: 50.0%", summary)
        self.assertIn("Battery: 88%", summary)
        self.assertIn("Plugged in", summary)

    def test_telemetry_summary_without_battery(self) -> None:
        cpu = CpuMetrics(percent=10.0, logical_cores=4, physical_cores=4)
        mem = MemoryMetrics(total_gb=8.0, used_gb=4.0, free_gb=4.0, percent=50.0)
        disk = DiskMetrics(total_gb=200.0, used_gb=100.0, free_gb=100.0, percent=50.0)

        telemetry = SystemTelemetry(cpu=cpu, memory=mem, disk=disk, battery=None, platform="Linux")
        summary = telemetry.summary()

        self.assertIn("CPU: 10.0%", summary)
        self.assertNotIn("Battery", summary)

    def test_process_and_window_info(self) -> None:
        proc = ProcessInfo(pid=1234, name="notepad.exe", cpu_percent=1.2, memory_percent=0.5)
        self.assertEqual(proc.pid, 1234)
        self.assertEqual(proc.name, "notepad.exe")

        win = WindowInfo(hwnd=5678, title="Untitled - Notepad", pid=1234, process_name="notepad.exe")
        self.assertEqual(win.hwnd, 5678)
        self.assertEqual(win.title, "Untitled - Notepad")
        self.assertTrue(win.is_visible)


class TestSecurityGuardrails(unittest.TestCase):
    """Test security policies and protection rules."""

    def setUp(self) -> None:
        self.guard = SecurityGuard()

    def test_critical_processes_are_protected(self) -> None:
        for proc in ("explorer.exe", "svchost.exe", "csrss.exe", "system", "EXPLORER", "winlogon"):
            self.assertTrue(
                self.guard.is_protected_process(proc),
                f"Expected {proc} to be protected.",
            )

    def test_safe_process_is_not_protected(self) -> None:
        self.assertFalse(self.guard.is_protected_process("notepad.exe"))
        self.assertFalse(self.guard.is_protected_process("calc.exe"))
        self.assertFalse(self.guard.is_protected_process("my_custom_script.py"))

    def test_validate_kill_target_blocks_critical(self) -> None:
        with self.assertRaises(SecurityBlockedError):
            self.guard.validate_kill_target("svchost.exe")

        with self.assertRaises(SecurityBlockedError):
            self.guard.validate_kill_target(0)  # System idle PID

    def test_validate_kill_target_allows_safe(self) -> None:
        try:
            self.guard.validate_kill_target("notepad.exe")
            self.guard.validate_kill_target(99999)
        except SecurityBlockedError:
            self.fail("SecurityGuard blocked a safe process.")

    def test_validate_app_launch_blocks_destructive(self) -> None:
        with self.assertRaises(SecurityBlockedError):
            self.guard.validate_app_launch("format C:")

        with self.assertRaises(SecurityBlockedError):
            self.guard.validate_app_launch("rm -rf /")

    def test_validate_app_launch_allows_safe(self) -> None:
        try:
            self.guard.validate_app_launch("notepad.exe")
            self.guard.validate_app_launch("calc")
        except SecurityBlockedError:
            self.fail("SecurityGuard blocked a safe app launch.")


class TestSystemMonitor(unittest.TestCase):
    """Test host telemetry monitoring."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.monitor = SystemMonitor(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

    def test_container_registration(self) -> None:
        self.assertTrue(self.container.exists("system_monitor"))
        self.assertIs(self.container.resolve("system_monitor"), self.monitor)

    def test_get_cpu_metrics(self) -> None:
        cpu = self.monitor.get_cpu_metrics()
        self.assertIsInstance(cpu, CpuMetrics)
        self.assertGreaterEqual(cpu.percent, 0.0)
        self.assertGreaterEqual(cpu.logical_cores, 1)

    def test_get_memory_metrics(self) -> None:
        mem = self.monitor.get_memory_metrics()
        self.assertIsInstance(mem, MemoryMetrics)
        self.assertGreater(mem.total_gb, 0.0)
        self.assertGreaterEqual(mem.percent, 0.0)
        self.assertLessEqual(mem.percent, 100.0)

    def test_get_disk_metrics(self) -> None:
        disk = self.monitor.get_disk_metrics()
        self.assertIsInstance(disk, DiskMetrics)
        self.assertGreater(disk.total_gb, 0.0)
        self.assertGreaterEqual(disk.percent, 0.0)

    def test_collect_telemetry_publishes_event(self) -> None:
        events = []
        self.event_bus.subscribe(EVENT_SYSTEM_TELEMETRY, lambda event: events.append(event))

        telemetry = self.monitor.collect_telemetry(publish=True)
        self.assertIsInstance(telemetry, SystemTelemetry)
        self.assertEqual(len(events), 1)
        self.assertIn("cpu_percent", events[0].payload)
        self.assertIn("summary", events[0].payload)

    def test_get_top_processes(self) -> None:
        procs = self.monitor.get_top_processes(limit=3)
        self.assertIsInstance(procs, list)
        for p in procs:
            self.assertIsInstance(p, ProcessInfo)


class TestWindowManager(unittest.TestCase):
    """Test window and process automation management."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.guard = SecurityGuard()
        self.win_mgr = WindowManager(
            security_guard_instance=self.guard,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

    def test_container_registration(self) -> None:
        self.assertTrue(self.container.exists("window_manager"))
        self.assertIs(self.container.resolve("window_manager"), self.win_mgr)

    @patch("subprocess.Popen")
    def test_launch_application_success(self, mock_popen: MagicMock) -> None:
        events = []
        self.event_bus.subscribe("automation.app.launched", lambda e: events.append(e))

        result = self.win_mgr.launch_application("notepad")
        self.assertTrue(result)
        mock_popen.assert_called_once()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["app"], "notepad")

    def test_launch_application_empty_fails(self) -> None:
        with self.assertRaises(ProcessOperationError):
            self.win_mgr.launch_application("   ")

    def test_launch_application_blocked_by_guard(self) -> None:
        with self.assertRaises(SecurityBlockedError):
            self.win_mgr.launch_application("format c:")

    def test_terminate_process_blocked_by_guard(self) -> None:
        with self.assertRaises(SecurityBlockedError):
            self.win_mgr.terminate_process("explorer.exe")

    @patch.object(WindowManager, "find_processes")
    def test_terminate_process_not_found(self, mock_find: MagicMock) -> None:
        mock_find.return_value = []
        with self.assertRaises(ProcessOperationError):
            self.win_mgr.terminate_process("non_existent_process_12345")


if __name__ == "__main__":
    unittest.main()
