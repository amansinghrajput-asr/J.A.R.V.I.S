"""Comprehensive unit tests for Sprint 22.2: Application Skills.

Verifies:
1. App skill registration into SkillManager and BaseSystemSkill hierarchy.
2. open_app with valid allowlisted target.
3. open_app rejects invalid target.
4. open_app rejects shell injection.
5. open_app rejects cmd/PowerShell.
6. is_app_running returns correct mocked result.
7. list_running_apps filters/structures results correctly.
8. close_app requires appropriate confirmation.
9. close_app rejects critical process.
10. close_app handles missing process.
11. close_app handles permission failure.
12. restart_app composes safe close/open behavior.
13. restart_app does not bypass confirmation.
14. PID reuse / process identity protection is tested with mocks.
15. Policy rejection is propagated correctly.
16. EventBus lifecycle events are emitted.
17. Concurrent read-only application queries are safe.
18. Architecture isolation/no global mutable state.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import MagicMock, patch

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillFailed,
    SystemSkillPolicyRejected,
    SystemSkillStarted,
)
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.app_skills import (
    AppResolver,
    AppSkills,
    DEFAULT_SAFE_APP_ALIASES,
    ProcessManager,
)
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)


class TestAppSkills(unittest.TestCase):
    """Unit test suite for AppSkills, AppResolver, and ProcessManager."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: list[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, lambda ev: self.events.append(ev))

        self.policy = SystemSecurityPolicy(event_bus=self.bus)
        self.confirmation_mgr = SystemConfirmationManager(event_bus=self.bus)
        self.resolver = AppResolver()
        self.process_manager = ProcessManager(security_policy=self.policy)

        self.skill = AppSkills(
            app_resolver=self.resolver,
            process_manager=self.process_manager,
            security_policy=self.policy,
            confirmation_manager=self.confirmation_mgr,
            event_bus=self.bus,
        )

    def test_01_skill_registration(self) -> None:
        """1. App skill registers properly into SkillManager and adheres to BaseSkill contract."""
        container = ServiceContainer()
        manager = SkillManager(container_instance=container, auto_register_in_container=False)
        manager.register(self.skill)

        registered = manager.get("app")
        self.assertIs(registered, self.skill)
        self.assertEqual(registered.name, "app")
        self.assertEqual(registered.priority, 60)
        self.assertTrue(self.skill.can_handle("open notepad"))
        self.assertTrue(self.skill.can_handle("close calc"))
        self.assertTrue(self.skill.can_handle("restart notepad"))
        self.assertTrue(self.skill.can_handle("is notepad running"))
        self.assertTrue(self.skill.can_handle("list running apps"))

    @patch("subprocess.Popen")
    def test_02_open_app_valid_target(self, mock_popen: MagicMock) -> None:
        """2. open_app successfully launches an allowlisted application."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_popen.return_value = mock_proc

        result = self.skill.execute({"operation": "open_app", "target": "notepad"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["app_name"], "notepad")
        self.assertEqual(result.data["pid"], 9999)
        self.assertTrue(result.data["launched"])

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertEqual(args[0], ["notepad.exe"])
        self.assertFalse(kwargs.get("shell", True))

    def test_03_open_app_rejects_invalid_target(self) -> None:
        """3. open_app rejects unrecognized or unallowlisted application targets."""
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({"operation": "open_app", "target": "unrecognized_unknown_app_xyz"})
        self.assertIn("not recognized or not allowlisted", str(ctx.exception))

    def test_04_open_app_rejects_shell_injection(self) -> None:
        """4. open_app rejects targets containing command injection tokens."""
        injection_targets = [
            "notepad.exe; calc.exe",
            "notepad && whoami",
            "calc | dir",
            "notepad `id`",
            "notepad $(calc)",
        ]
        for target in injection_targets:
            with self.subTest(target=target):
                with self.assertRaises((SecurityPolicyViolationError, SkillExecutionError)):
                    self.skill.execute({"operation": "open_app", "target": target})

    def test_05_open_app_rejects_cmd_and_powershell(self) -> None:
        """5. open_app strictly rejects direct shell interpreters."""
        shell_targets = ["cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "bash", "sh"]
        for target in shell_targets:
            with self.subTest(target=target):
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.execute({"operation": "open_app", "target": target})

    @patch.object(ProcessManager, "is_running")
    def test_06_is_app_running(self, mock_is_running: MagicMock) -> None:
        """6. is_app_running returns correct structured status."""
        mock_is_running.return_value = (True, [1234, 5678])

        res = self.skill.execute({"operation": "is_app_running", "target": "notepad"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["running"])
        self.assertEqual(res.data["pids"], [1234, 5678])
        self.assertEqual(res.data["count"], 2)

    @patch.object(ProcessManager, "list_running")
    def test_07_list_running_apps(self, mock_list: MagicMock) -> None:
        """7. list_running_apps returns structured list of user applications."""
        mock_list.return_value = [
            {"pid": 100, "name": "notepad.exe", "cpu_percent": 0.5, "memory_percent": 1.2, "status": "running"},
            {"pid": 200, "name": "chrome.exe", "cpu_percent": 2.1, "memory_percent": 4.5, "status": "running"},
        ]

        res = self.skill.execute({"operation": "list_running_apps"})
        self.assertTrue(res.success)
        self.assertEqual(res.data["total_listed"], 2)
        self.assertEqual(len(res.data["apps"]), 2)
        self.assertEqual(res.data["apps"][0]["name"], "notepad.exe")

    @patch.object(ProcessManager, "terminate_processes")
    @patch.object(ProcessManager, "get_process_snapshots")
    def test_08_close_app_requires_confirmation(
        self, mock_snapshots: MagicMock, mock_term: MagicMock
    ) -> None:
        """8. close_app requires interactive confirmation before termination."""
        mock_snapshots.return_value = [{"pid": 4321, "name": "notepad.exe", "create_time": 100.0}]
        mock_term.return_value = {"target": "notepad", "closed": True, "terminated_count": 1, "pids": [4321]}

        # First call without confirmation_id raises ConfirmationRequiredError
        with self.assertRaises(ConfirmationRequiredError) as ctx:
            self.skill.execute({"operation": "close_app", "target": "notepad"})

        token = ctx.exception.confirmation_id
        self.assertTrue(token.startswith("sys_conf_"))

        # Confirm token
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        # Re-execute with token
        res = self.skill.execute({
            "operation": "close_app",
            "target": "notepad",
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertEqual(res.data["terminated_count"], 1)

    def test_09_close_app_rejects_critical_process(self) -> None:
        """9. close_app rejects critical OS processes immediately."""
        critical_targets = ["explorer.exe", "csrss.exe", "dwm.exe", "0", "4"]
        for target in critical_targets:
            with self.subTest(target=target):
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.execute({"operation": "close_app", "target": target})

    @patch.object(ProcessManager, "get_process_snapshots")
    def test_10_close_app_handles_missing_process(self, mock_snapshots: MagicMock) -> None:
        """10. close_app handles process-not-found cleanly without crashing."""
        mock_snapshots.return_value = []

        # When process is not running, close_app requires confirmation by policy, then reports not found
        token = self.confirmation_mgr.request_confirmation("close_app", target="nonexistent_app")
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        res = self.skill.execute({
            "operation": "close_app",
            "target": "nonexistent_app",
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertFalse(res.data["closed"])
        self.assertEqual(res.data["terminated_count"], 0)
        self.assertEqual(res.data["reason"], "Process not found")

    @patch.object(ProcessManager, "get_process_snapshots")
    @patch.object(ProcessManager, "terminate_processes")
    def test_11_close_app_handles_permission_failure(
        self, mock_term: MagicMock, mock_snapshots: MagicMock
    ) -> None:
        """11. close_app handles permission / access denied failure gracefully."""
        mock_snapshots.return_value = [{"pid": 5555, "name": "locked.exe", "create_time": 100.0}]
        mock_term.side_effect = PermissionError("Access denied by OS security descriptor")

        token = self.confirmation_mgr.request_confirmation("close_app", target="locked.exe")
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({
                "operation": "close_app",
                "target": "locked.exe",
                "confirmation_id": token,
            })
        self.assertIn("Access denied", str(ctx.exception))

    @patch.object(AppSkills, "close_app")
    @patch.object(AppSkills, "open_app")
    def test_12_restart_app_composition(
        self, mock_open: MagicMock, mock_close: MagicMock
    ) -> None:
        """12. restart_app composes safe close followed by open."""
        mock_close.return_value = {"target": "notepad", "closed": True, "terminated_count": 1}
        mock_open.return_value = {"app_name": "notepad", "pid": 7777, "launched": True}

        token = self.confirmation_mgr.request_confirmation("restart_app", target="notepad")
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        res = self.skill.execute({
            "operation": "restart_app",
            "target": "notepad",
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertTrue(res.data["restarted"])
        mock_close.assert_called_once()
        mock_open.assert_called_once()

    def test_13_restart_app_does_not_bypass_confirmation(self) -> None:
        """13. restart_app requires confirmation just like close_app."""
        with self.assertRaises(ConfirmationRequiredError):
            self.skill.execute({"operation": "restart_app", "target": "notepad"})

    def test_14_pid_reuse_protection(self) -> None:
        """14. ProcessManager guards against PID recycling/identity changes before termination."""
        pm = ProcessManager()

        # Simulate PID 999 was notepad when snapshotted
        stale_snapshots = [{"pid": 999, "name": "notepad.exe", "create_time": 10.0}]

        # But current process table shows PID 999 is now chrome.exe
        with patch.object(pm, "get_process_snapshots") as mock_current:
            mock_current.return_value = [{"pid": 999, "name": "chrome.exe", "create_time": 20.0}]

            res = pm.terminate_processes(
                target="notepad",
                expected_snapshots=stale_snapshots,
            )
            # Should have skipped terminating PID 999 because name changed from notepad.exe to chrome.exe!
            self.assertEqual(res["terminated_count"], 0)
            self.assertEqual(res["pids"], [])

    def test_15_policy_rejection_propagated_with_events(self) -> None:
        """15. Prohibited operations emit SystemSkillPolicyRejected and raise SecurityPolicyViolationError."""
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "close_app", "target": "smss.exe"})

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillPolicyRejected, event_types)

    @patch("subprocess.Popen")
    def test_16_event_bus_lifecycle_emission(self, mock_popen: MagicMock) -> None:
        """16. Successful execution emits SystemSkillStarted and SystemSkillCompleted."""
        mock_popen.return_value = MagicMock(pid=1111)

        self.skill.execute({"operation": "open_app", "target": "calc"})

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillCompleted, event_types)

    @patch.object(ProcessManager, "is_running")
    def test_17_concurrent_read_only_queries(self, mock_is_running: MagicMock) -> None:
        """17. Concurrent read-only queries (is_app_running, list_running_apps) are safe."""
        mock_is_running.return_value = (True, [123])

        def worker(idx: int) -> bool:
            res = self.skill.execute({"operation": "is_app_running", "target": "calc"})
            return res.success and res.data["running"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(30)))

        self.assertTrue(all(results))

    def test_18_architecture_isolation_and_di(self) -> None:
        """18. Verify independent instances have isolated configurations and zero global state."""
        skill1 = AppSkills()
        skill2 = AppSkills()

        skill1.resolver.register_alias("custom1", "c:\\custom1.exe")
        self.assertTrue(skill1.resolver.resolve("custom1")[0])
        self.assertFalse(skill2.resolver.resolve("custom1")[0])


if __name__ == "__main__":
    unittest.main()
