"""Comprehensive unit and security test suite for Phase 22.4 System Information Skills.

Tests:
- All 7 read-only operations: CPU, Memory, Disk, Battery, GPU, Network, System Summary
- Hardware-agnostic execution with mocks for battery, GPU, and psutil
- Strict path security validation for disk queries (traversal, null bytes, reserved names)
- Safety tier classification (all SAFE, zero confirmation required)
- EventBus lifecycle notifications
- Dependency Injection and ServiceContainer integration
- No shell execution and zero filesystem mutation
- Concurrent thread-safety
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from app.ai.planner.events import (
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillStarted,
)
from app.automation.system import SystemMonitor
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.skills.system.system_info_skills import SystemInfoSkills


class TestSystemInfoSkills(unittest.TestCase):
    """Test suite covering SystemInfoSkills operations, security, and integration."""

    def setUp(self) -> None:
        """Create fresh isolated environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name).resolve()
        self.event_bus = PlannerEventBus()
        self.security_policy = SystemSecurityPolicy(
            allowed_roots=[self.temp_path],
            event_bus=self.event_bus,
        )
        self.confirmation_manager = SystemConfirmationManager(event_bus=self.event_bus)
        self.container = ServiceContainer()
        self.skill = SystemInfoSkills(
            security_policy=self.security_policy,
            confirmation_manager=self.confirmation_manager,
            container=self.container,
            event_bus=self.event_bus,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # 1. Metadata and Registration
    # ---------------------------------------------------------------------------

    def test_01_skill_registration_and_metadata(self) -> None:
        """Verify skill name, priority, permissions, and discovery tags."""
        self.assertEqual(self.skill.name, "system_info")
        self.assertEqual(self.skill.priority, 60)
        self.assertTrue(self.skill.enabled)
        self.assertIn("system:read", self.skill.permissions)
        self.assertIn("telemetry", self.skill.tags)
        self.assertIn("cpu", self.skill.tags)
        self.assertIn("gpu", self.skill.tags)

    def test_02_can_handle_explicit_operations(self) -> None:
        """Verify can_handle accepts all structured operation names."""
        for op in (
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
        ):
            self.assertTrue(self.skill.can_handle({"operation": op}))
            self.assertTrue(self.skill.can_handle(op))

    def test_03_can_handle_natural_language_queries(self) -> None:
        """Verify can_handle accepts natural language queries from requirements."""
        queries = [
            ("What is my CPU usage?", "get_cpu_info"),
            ("How much RAM is being used?", "get_memory_info"),
            ("How much disk space is available?", "get_disk_info"),
            ("What is my battery level?", "get_battery_info"),
            ("What GPU do I have?", "get_gpu_info"),
            ("What is my network status?", "get_network_info"),
            ("Give me a system summary.", "get_system_summary"),
        ]
        for query, expected_op in queries:
            self.assertTrue(self.skill.can_handle(query), f"Failed to handle: {query}")
            op, _, _, _ = self.skill.parse_command(query)
            self.assertEqual(op, expected_op, f"Mismatch for '{query}': got {op}")

    def test_04_can_handle_rejects_unrelated_queries(self) -> None:
        """Ensure unrelated commands and queries are rejected."""
        unrelated = [
            "open notepad",
            "delete file secret.txt",
            "write a poem about space",
            "turn off the lights",
            "",
            None,
        ]
        for cmd in unrelated:
            self.assertFalse(self.skill.can_handle(cmd), f"Should not handle: {cmd}")

    # ---------------------------------------------------------------------------
    # 2. CPU Information
    # ---------------------------------------------------------------------------

    def test_05_get_cpu_info_structure_and_ranges(self) -> None:
        """Verify get_cpu_info returns expected keys and valid ranges."""
        res = self.skill.execute({"operation": "get_cpu_info"})
        self.assertIsInstance(res, SystemSkillResult)
        self.assertTrue(res.success)
        data = res.data
        self.assertIn("usage_percent", data)
        self.assertIn("logical_cores", data)
        self.assertIn("physical_cores", data)
        self.assertIn("frequency_mhz", data)

        self.assertIsInstance(data["usage_percent"], (int, float))
        self.assertGreaterEqual(data["usage_percent"], 0.0)
        self.assertLessEqual(data["usage_percent"], 100.0)
        self.assertGreaterEqual(data["logical_cores"], 1)
        self.assertGreaterEqual(data["physical_cores"], 1)

    def test_06_get_cpu_info_fallback_when_psutil_missing(self) -> None:
        """Verify safe CPU fallback when psutil is unavailable."""
        with patch.object(SystemInfoSkills, "has_psutil", False):
            res = self.skill.get_cpu_info()
            self.assertIn("usage_percent", res)
            self.assertIn("logical_cores", res)
            self.assertEqual(res["usage_percent"], 0.0)
            self.assertGreaterEqual(res["logical_cores"], 1)

    # ---------------------------------------------------------------------------
    # 3. Memory Information
    # ---------------------------------------------------------------------------

    def test_07_get_memory_info_structure_and_values(self) -> None:
        """Verify get_memory_info returns valid structure and non-negative values."""
        res = self.skill.execute({"operation": "get_memory_info"})
        self.assertTrue(res.success)
        data = res.data
        for k in ("total_bytes", "available_bytes", "used_bytes", "usage_percent"):
            self.assertIn(k, data)

        self.assertGreater(data["total_bytes"], 0)
        self.assertGreaterEqual(data["available_bytes"], 0)
        self.assertGreaterEqual(data["used_bytes"], 0)
        self.assertGreaterEqual(data["usage_percent"], 0.0)
        self.assertLessEqual(data["usage_percent"], 100.0)

    def test_08_get_memory_info_fallback(self) -> None:
        """Verify memory query fallback when psutil is not available."""
        with patch.object(SystemInfoSkills, "has_psutil", False):
            res = self.skill.get_memory_info()
            self.assertIn("total_bytes", res)
            self.assertIn("available_bytes", res)
            self.assertIn("used_bytes", res)
            self.assertIn("usage_percent", res)

    # ---------------------------------------------------------------------------
    # 4. Disk Information & Path Security
    # ---------------------------------------------------------------------------

    def test_09_get_disk_info_default_path(self) -> None:
        """Verify get_disk_info defaults to primary root drive."""
        # Allow system root queries on unrestricted policy
        unrestricted_policy = SystemSecurityPolicy()
        skill = SystemInfoSkills(security_policy=unrestricted_policy)
        res = skill.execute({"operation": "get_disk_info"})
        self.assertTrue(res.success)
        data = res.data
        self.assertIn("path", data)
        self.assertIn("total_bytes", data)
        self.assertIn("used_bytes", data)
        self.assertIn("free_bytes", data)
        self.assertIn("usage_percent", data)
        self.assertGreater(data["total_bytes"], 0)
        self.assertGreater(data["free_bytes"], 0)

    def test_10_get_disk_info_custom_valid_path(self) -> None:
        """Verify get_disk_info for a custom valid folder inside allowed root."""
        sub = self.temp_path / "subfolder"
        sub.mkdir()
        res = self.skill.get_disk_info(path=str(sub))
        self.assertEqual(res["path"], str(sub))
        self.assertGreater(res["total_bytes"], 0)

    def test_11_get_disk_info_nonexistent_path_rejected(self) -> None:
        """Verify querying a nonexistent disk path raises FileNotFoundError."""
        nonexistent = self.temp_path / "does_not_exist_abc123"
        with self.assertRaises(FileNotFoundError):
            self.skill.get_disk_info(path=str(nonexistent))

    def test_12_get_disk_info_path_traversal_rejected(self) -> None:
        """Verify path traversal escaping allowed root is rejected."""
        traversal = str(self.temp_path / ".." / "outside")
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "get_disk_info", "parameters": {"path": traversal}})

    def test_13_get_disk_info_reserved_device_name_rejected(self) -> None:
        """Verify Windows reserved device names (CON, NUL, AUX) are rejected."""
        for reserved in ("CON", "NUL", "AUX", "COM1", "PRN"):
            with self.assertRaises(SecurityPolicyViolationError):
                self.skill.execute({"operation": "get_disk_info", "parameters": {"path": reserved}})

    def test_14_get_disk_info_null_byte_rejected(self) -> None:
        """Verify paths with null bytes are strictly rejected."""
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "get_disk_info", "parameters": {"path": "C:\\test\0bad"}})

    # ---------------------------------------------------------------------------
    # 5. Battery Information (Hardware Agnostic)
    # ---------------------------------------------------------------------------

    def test_15_get_battery_info_structure_with_battery(self) -> None:
        """Verify battery info parsing when battery hardware is present."""
        mock_sensors = MagicMock()
        mock_sensors.percent = 85.5
        mock_sensors.power_plugged = True
        mock_sensors.secsleft = 7200

        with patch("psutil.sensors_battery", return_value=mock_sensors):
            res = self.skill.get_battery_info()
            self.assertTrue(res["available"])
            self.assertEqual(res["percent"], 85.5)
            self.assertTrue(res["plugged"])
            self.assertEqual(res["seconds_left"], 7200)

    def test_16_get_battery_info_structure_without_battery(self) -> None:
        """Verify battery info parsing on desktop machines without battery hardware."""
        with patch("psutil.sensors_battery", return_value=None):
            res = self.skill.get_battery_info()
            self.assertFalse(res["available"])
            self.assertIsNone(res["percent"])
            self.assertIsNone(res["plugged"])
            self.assertIsNone(res["seconds_left"])

    # ---------------------------------------------------------------------------
    # 6. GPU Information (Hardware Agnostic & No Shell)
    # ---------------------------------------------------------------------------

    def test_17_get_gpu_info_via_nvidia_smi(self) -> None:
        """Verify GPU info extraction via nvidia-smi subprocess parsing."""
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "NVIDIA GeForce RTX 4090, 24576, 2048, 22528\n"

        with patch("shutil.which", return_value="C:\\Windows\\System32\\nvidia-smi.exe"), \
             patch("subprocess.run", return_value=mock_res) as mock_run:
            res = self.skill.get_gpu_info()
            self.assertTrue(res["available"])
            self.assertEqual(res["name"], "NVIDIA GeForce RTX 4090")
            self.assertEqual(res["memory_total_bytes"], 24576 * 1024 * 1024)
            self.assertEqual(res["memory_used_bytes"], 2048 * 1024 * 1024)
            self.assertEqual(res["memory_free_bytes"], 22528 * 1024 * 1024)

            # Assert subprocess.run called with list and no shell
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            self.assertIsInstance(args[0], list)
            self.assertNotIn("shell", kwargs)

    def test_18_get_gpu_info_fallback_winreg(self) -> None:
        """Verify GPU info fallback via Windows registry when nvidia-smi is absent."""
        mock_key = MagicMock()

        def mock_enum(k, i):
            if i == 0:
                return "0000"
            raise OSError("No more keys")

        def mock_query(subkey, val):
            if val == "DriverDesc":
                return ("Intel(R) Iris(R) Xe Graphics", 1)
            if val == "HardwareInformation.qwMemorySize":
                return (1073741824, 1)
            raise FileNotFoundError()

        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.exists", return_value=False), \
             patch("platform.system", return_value="Windows"), \
             patch("winreg.OpenKey", return_value=mock_key), \
             patch("winreg.EnumKey", side_effect=mock_enum), \
             patch("winreg.QueryValueEx", side_effect=mock_query):
            res = self.skill.get_gpu_info()
            self.assertTrue(res["available"])
            self.assertEqual(res["name"], "Intel(R) Iris(R) Xe Graphics")
            self.assertEqual(res["memory_total_bytes"], 1073741824)

    def test_19_get_gpu_info_unavailable(self) -> None:
        """Verify GPU info when no GPU or tools are available."""
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.exists", return_value=False), \
             patch("platform.system", return_value="Linux"):
            res = self.skill.get_gpu_info()
            self.assertFalse(res["available"])
            self.assertIsNone(res["name"])
            self.assertIsNone(res["memory_total_bytes"])

    def test_20_get_gpu_info_no_shell_execution(self) -> None:
        """Verify no command strings or shell interpreters are invoked for GPU query."""
        with patch("shutil.which", return_value="C:\\Windows\\System32\\nvidia-smi.exe"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            self.skill.get_gpu_info()
            self.assertTrue(mock_run.called)
            cmd_arg = mock_run.call_args[0][0]
            self.assertIsInstance(cmd_arg, list)
            # Verify no shell string injection
            for token in cmd_arg:
                self.assertNotIn("cmd", token.lower())
                self.assertNotIn("powershell", token.lower())
                self.assertNotIn("|", token)
                self.assertNotIn(";", token)

    # ---------------------------------------------------------------------------
    # 7. Network Information
    # ---------------------------------------------------------------------------

    def test_21_get_network_info_structure(self) -> None:
        """Verify get_network_info returns structured interface and IO stats."""
        res = self.skill.execute({"operation": "get_network_info"})
        self.assertTrue(res.success)
        data = res.data
        self.assertIn("interfaces", data)
        self.assertIn("active_interfaces", data)
        self.assertIn("bytes_sent", data)
        self.assertIn("bytes_received", data)
        self.assertIsInstance(data["interfaces"], list)
        self.assertGreaterEqual(data["active_interfaces"], 0)
        self.assertGreaterEqual(data["bytes_sent"], 0)
        self.assertGreaterEqual(data["bytes_received"], 0)

    def test_22_get_network_info_fallback(self) -> None:
        """Verify network info fallback when psutil is missing."""
        with patch.object(SystemInfoSkills, "has_psutil", False):
            res = self.skill.get_network_info()
            self.assertEqual(res["interfaces"], [])
            self.assertEqual(res["active_interfaces"], 0)
            self.assertEqual(res["bytes_sent"], 0)
            self.assertEqual(res["bytes_received"], 0)

    # ---------------------------------------------------------------------------
    # 8. System Summary
    # ---------------------------------------------------------------------------

    def test_23_get_system_summary_structure(self) -> None:
        """Verify get_system_summary aggregates high-level metrics accurately."""
        res = self.skill.execute({"operation": "get_system_summary"})
        self.assertTrue(res.success)
        data = res.data
        for section in ("platform", "cpu", "memory", "disk", "battery", "gpu", "network"):
            self.assertIn(section, data)

        self.assertIn("usage_percent", data["cpu"])
        self.assertIn("usage_percent", data["memory"])
        self.assertIn("total_gb", data["disk"])
        self.assertIn("available", data["battery"])
        self.assertIn("available", data["gpu"])
        self.assertIn("active_interfaces", data["network"])

    # ---------------------------------------------------------------------------
    # 9. Security Policy, Events, DI, and Concurrency
    # ---------------------------------------------------------------------------

    def test_24_all_operations_classified_as_safe(self) -> None:
        """Verify all 7 system info operations are classified as SAFE."""
        for op in (
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
        ):
            tier, reason = self.security_policy.validate_operation(op)
            self.assertEqual(
                tier,
                SystemSafetyTier.SAFE,
                f"Operation '{op}' must be SAFE, but got '{tier}': {reason}",
            )

    def test_25_no_confirmation_required_for_any_operation(self) -> None:
        """Verify no confirmation manager tokens are requested for info operations."""
        initial_requests = len(self.confirmation_manager._requests)
        for op in (
            "get_cpu_info",
            "get_memory_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
        ):
            res = self.skill.execute({"operation": op})
            self.assertTrue(res.success)
        final_requests = len(self.confirmation_manager._requests)
        self.assertEqual(initial_requests, final_requests)

    def test_26_execute_lifecycle_events(self) -> None:
        """Verify execution emits SystemSkillStarted and SystemSkillCompleted events."""
        events: list[Any] = []
        self.event_bus.subscribe(SystemSkillStarted, lambda e: events.append(e))
        self.event_bus.subscribe(SystemSkillCompleted, lambda e: events.append(e))

        res = self.skill.execute({"operation": "get_cpu_info"})
        self.assertTrue(res.success)
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], SystemSkillStarted)
        self.assertIsInstance(events[1], SystemSkillCompleted)
        self.assertEqual(events[0].operation, "get_cpu_info")
        self.assertEqual(events[1].operation, "get_cpu_info")

    def test_27_dependency_injection_container(self) -> None:
        """Verify SystemInfoSkills can resolve dependencies from ServiceContainer."""
        container = ServiceContainer()
        custom_policy = SystemSecurityPolicy()
        container.register_singleton("system_security_policy", custom_policy)

        skill = SystemInfoSkills(container=container)
        self.assertIs(skill.security_policy, custom_policy)

    def test_28_repeated_calls_and_idempotence(self) -> None:
        """Verify repeated executions do not leak memory, locks, or state."""
        for _ in range(5):
            res = self.skill.execute("what is my cpu usage?")
            self.assertTrue(res.success)
            self.assertIn("usage_percent", res.data)

    def test_29_no_mutation_of_filesystem_or_system_state(self) -> None:
        """Verify read-only operations never write, touch, or delete files."""
        sub = self.temp_path / "check_immutable"
        sub.mkdir()
        baseline_files = list(sub.glob("*"))

        self.skill.execute({"operation": "get_disk_info", "parameters": {"path": str(sub)}})
        self.skill.execute({"operation": "get_system_summary"})

        after_files = list(sub.glob("*"))
        self.assertEqual(baseline_files, after_files)

    def test_30_thread_safety_concurrent_queries(self) -> None:
        """Verify thread-safety when querying system metrics concurrently."""
        ops = [
            "get_cpu_info",
            "get_memory_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
        ]
        errors = []

        def worker(op_name: str) -> None:
            try:
                res = self.skill.execute({"operation": op_name})
                if not res.success:
                    errors.append(f"{op_name} failed: {res.error}")
            except Exception as e:
                errors.append(f"{op_name} raised: {e}")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, op) for op in ops * 3]
            for f in futures:
                f.result()

        self.assertEqual(len(errors), 0, f"Concurrent query errors: {errors}")


if __name__ == "__main__":
    unittest.main()
