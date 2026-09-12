"""Comprehensive unit, security, and invariant tests for Phase 22.6 Window Skills.

SAFETY IS THE HIGHEST PRIORITY:
All Win32 state-changing APIs (SetForegroundWindow, ShowWindow, PostMessageW,
EnumWindows, IsWindow, etc.) are 100% mocked.
NO REAL WINDOW ACTIONS OCCUR.
NO REAL WINDOWS ARE CLOSED, MINIMIZED, MAXIMIZED, OR FOCUSED.
Zero subprocesses, shell commands, or power transitions are executed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import unittest
from unittest.mock import MagicMock, call, patch

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillPolicyRejected,
    SystemSkillStarted,
)
from app.automation.models import WindowInfo
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.skills.system.window_skills import (
    MAX_WINDOW_ENUM_LIMIT,
    SW_MAXIMIZE,
    SW_MINIMIZE,
    SW_RESTORE,
    WM_CLOSE,
    WindowSkills,
)


class TestWindowSkills(unittest.TestCase):
    """Unit and security test suite for WindowSkills."""

    def setUp(self) -> None:
        """Initialize test environment with isolated bus, security policy, and mocked APIs."""
        self.container = ServiceContainer()
        self.bus = PlannerEventBus()
        self.security_policy = SystemSecurityPolicy(event_bus=self.bus)
        self.confirmation_manager = SystemConfirmationManager(event_bus=self.bus)

        self.skill = WindowSkills(
            security_policy=self.security_policy,
            confirmation_manager=self.confirmation_manager,
            container=self.container,
            event_bus=self.bus,
        )

        # Intercept event publishing for inspection
        self.events = []
        self.bus.subscribe(PlannerEvent, lambda e: self.events.append(e))

        # Class / instance mocks for Win32 calls so NO real window action ever executes
        self.skill._api_is_window = MagicMock(return_value=True)
        self.skill._api_is_window_visible = MagicMock(return_value=True)
        self.skill._api_is_cloaked = MagicMock(return_value=False)
        self.skill._api_get_foreground_window = MagicMock(return_value=1001)
        self.skill._api_get_window_text = MagicMock(return_value="Document - Notepad")
        self.skill._api_get_window_pid = MagicMock(return_value=5432)
        self.skill._get_process_name = MagicMock(return_value="notepad.exe")
        self.skill._api_set_foreground_window = MagicMock(return_value=True)
        self.skill._api_show_window = MagicMock(return_value=True)
        self.skill._api_post_wm_close = MagicMock(return_value=True)
        self.skill._api_enum_windows = MagicMock(return_value=[1001, 1002, 1003])

    # -----------------------------------------------------------------------
    # 1. Routing & Intent Recognition Tests
    # -----------------------------------------------------------------------

    def test_01_can_handle_all_window_operations(self) -> None:
        """Verify can_handle recognizes all 7 window operations and NL phrasing."""
        valid_cmds = [
            {"operation": "list_windows"},
            {"operation": "list_open_windows"},
            {"operation": "get_active_window"},
            {"operation": "focus_window", "target": "Notepad"},
            {"operation": "minimize_window", "target": "Notepad"},
            {"operation": "maximize_window", "target": "Notepad"},
            {"operation": "restore_window", "target": "Notepad"},
            {"operation": "close_window", "target": "Notepad"},
            "list windows",
            "list open windows",
            "get active window",
            "focus window Notepad",
            "switch to window Chrome",
            "minimize window Notepad",
            "maximize window Notepad",
            "restore window Notepad",
            "close window Notepad",
        ]
        for cmd in valid_cmds:
            with self.subTest(cmd=cmd):
                self.assertTrue(self.skill.can_handle(cmd))

    def test_02_can_handle_rejects_unrelated_commands(self) -> None:
        """Verify can_handle ignores unrelated actions."""
        invalid_cmds = [
            {"operation": "open_app"},
            {"operation": "delete_file"},
            {"operation": "set_volume"},
            "open notepad",
            "reboot system",
            "shutdown",
            "format c:",
        ]
        for cmd in invalid_cmds:
            with self.subTest(cmd=cmd):
                self.assertFalse(self.skill.can_handle(cmd))

    # -----------------------------------------------------------------------
    # 2. Query Operations (SAFE Tier)
    # -----------------------------------------------------------------------

    def test_03_list_open_windows_success(self) -> None:
        """Verify listing open windows returns structured, bounded list."""
        def mock_text(hwnd):
            return {1001: "Document - Notepad", 1002: "Google Chrome", 1003: "Calculator"}.get(hwnd, "")
        def mock_pid(hwnd):
            return {1001: 5432, 1002: 8765, 1003: 9999}.get(hwnd, 0)
        def mock_proc(pid):
            return {5432: "notepad.exe", 8765: "chrome.exe", 9999: "calc.exe"}.get(pid, None)

        self.skill._api_get_window_text = MagicMock(side_effect=mock_text)
        self.skill._api_get_window_pid = MagicMock(side_effect=mock_pid)
        self.skill._get_process_name = MagicMock(side_effect=mock_proc)

        res = self.skill.execute({"operation": "list_windows"})
        self.assertTrue(res.success)
        self.assertEqual(res.data["count"], 3)
        self.assertEqual(len(res.data["windows"]), 3)
        first_win = res.data["windows"][0]
        self.assertEqual(first_win["hwnd"], 1001)
        self.assertEqual(first_win["title"], "Document - Notepad")
        self.assertEqual(first_win["pid"], 5432)
        self.assertEqual(first_win["process_name"], "notepad.exe")

    def test_04_list_open_windows_alias(self) -> None:
        """Verify list_open_windows alias returns identical structured format."""
        res = self.skill.execute({"operation": "list_open_windows"})
        self.assertTrue(res.success)
        self.assertIn("windows", res.data)
        self.assertIn("count", res.data)

    def test_05_get_active_window_success(self) -> None:
        """Verify get_active_window queries and returns current foreground window."""
        self.skill._api_get_foreground_window = MagicMock(return_value=2048)
        self.skill._api_get_window_text = MagicMock(return_value="J.A.R.V.I.S Console")
        self.skill._api_get_window_pid = MagicMock(return_value=7777)
        self.skill._get_process_name = MagicMock(return_value="python.exe")

        res = self.skill.execute({"operation": "get_active_window"})
        self.assertTrue(res.success)
        act = res.data["active_window"]
        self.assertIsNotNone(act)
        self.assertEqual(act["hwnd"], 2048)
        self.assertEqual(act["title"], "J.A.R.V.I.S Console")
        self.assertEqual(act["pid"], 7777)
        self.assertTrue(act["is_foreground"])

    def test_06_get_active_window_none_detected(self) -> None:
        """Verify get_active_window handles situation where no foreground window exists."""
        self.skill._api_get_foreground_window = MagicMock(return_value=0)
        res = self.skill.execute({"operation": "get_active_window"})
        self.assertTrue(res.success)
        self.assertIsNone(res.data["active_window"])
        self.assertEqual(res.data["status"], "no_active_window")

    # -----------------------------------------------------------------------
    # 3. State-Changing Window Operations (SAFE Tier)
    # -----------------------------------------------------------------------

    def test_07_focus_window_by_title_and_by_hwnd(self) -> None:
        """Verify focus_window resolves window, restores if needed, and calls SetForegroundWindow."""
        # Focus by title substring
        res = self.skill.execute({"operation": "focus_window", "target": "Notepad"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["focused"])
        self.skill._api_show_window.assert_called_with(1001, SW_RESTORE)
        self.skill._api_set_foreground_window.assert_called_with(1001)

        # Focus by direct HWND parameter
        self.skill._api_set_foreground_window.reset_mock()
        res_hwnd = self.skill.execute({
            "operation": "focus_window",
            "parameters": {"hwnd": 1001},
        })
        self.assertTrue(res_hwnd.success)
        self.skill._api_set_foreground_window.assert_called_with(1001)

    def test_08_minimize_window_success(self) -> None:
        """Verify minimize_window invokes ShowWindow with SW_MINIMIZE."""
        res = self.skill.execute({"operation": "minimize_window", "target": "Notepad"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["minimized"])
        self.skill._api_show_window.assert_called_with(1001, SW_MINIMIZE)

    def test_09_maximize_window_success(self) -> None:
        """Verify maximize_window invokes ShowWindow with SW_MAXIMIZE."""
        res = self.skill.execute({"operation": "maximize_window", "target": "Notepad"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["maximized"])
        self.skill._api_show_window.assert_called_with(1001, SW_MAXIMIZE)

    def test_10_restore_window_success(self) -> None:
        """Verify restore_window invokes ShowWindow with SW_RESTORE."""
        res = self.skill.execute({"operation": "restore_window", "target": "Notepad"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["restored"])
        self.skill._api_show_window.assert_called_with(1001, SW_RESTORE)

    # -----------------------------------------------------------------------
    # 4. close_window & Confirmation Protection
    # -----------------------------------------------------------------------

    def test_11_close_window_requires_confirmation(self) -> None:
        """Verify close_window is classified as CONFIRMATION_REQUIRED and halts without token."""
        cmd = {"operation": "close_window", "target": "Notepad"}
        with self.assertRaises(ConfirmationRequiredError) as ctx:
            self.skill.execute(cmd)

        token = ctx.exception.confirmation_id
        self.assertTrue(len(token) > 0)
        # Ensure WM_CLOSE was NOT posted
        self.skill._api_post_wm_close.assert_not_called()

    def test_12_close_window_with_approved_confirmation_token(self) -> None:
        """Verify close_window succeeds when approved confirmation token is supplied."""
        cmd = {"operation": "close_window", "target": "Notepad"}
        try:
            self.skill.execute(cmd)
        except ConfirmationRequiredError as exc:
            token = exc.confirmation_id

        # Approve token via manager
        self.confirmation_manager.resolve_confirmation(token, approved=True, decided_by="test_user")

        # Execute with confirmation token
        cmd_confirmed = {
            "operation": "close_window",
            "target": "Notepad",
            "confirmation_id": token,
        }
        res = self.skill.execute(cmd_confirmed)
        self.assertTrue(res.success)
        self.assertTrue(res.data["closed"])
        self.assertEqual(res.data["method"], "WM_CLOSE")
        self.skill._api_post_wm_close.assert_called_once_with(1001)

    def test_13_close_window_rejected_when_confirmation_denied(self) -> None:
        """Verify close_window fails if operator denies confirmation token."""
        cmd = {"operation": "close_window", "target": "Notepad"}
        try:
            self.skill.execute(cmd)
        except ConfirmationRequiredError as exc:
            token = exc.confirmation_id

        self.confirmation_manager.resolve_confirmation(token, approved=False, decided_by="test_user")

        cmd_rejected = {
            "operation": "close_window",
            "target": "Notepad",
            "confirmation_id": token,
        }
        with self.assertRaises(ConfirmationRejectedError):
            self.skill.execute(cmd_rejected)

        self.skill._api_post_wm_close.assert_not_called()

    # -----------------------------------------------------------------------
    # 5. Critical System Invariants & RESTRICTED Protections
    # -----------------------------------------------------------------------

    def test_14_close_window_critical_processes_strictly_restricted(self) -> None:
        """Verify attempts to close critical processes (dwm, explorer, PID 4, etc.) are RESTRICTED."""
        critical_targets = [
            "explorer.exe",
            "dwm.exe",
            "csrss.exe",
            "smss.exe",
            "wininit.exe",
            "services.exe",
            "lsass.exe",
            "svchost.exe",
            "0",
            "4",
            str(os.getpid()),
        ]
        for crit in critical_targets:
            with self.subTest(crit=crit):
                tier, reason = self.security_policy.validate_operation(
                    "close_window",
                    target=crit,
                )
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertIn("protected system process", reason)

        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "close_window", "target": "explorer.exe"})

    def test_15_close_window_toctou_critical_check_blocks_action(self) -> None:
        """Verify if a window belongs to PID 4 or dwm.exe, _handle_close_window rejects it."""
        self.skill._api_get_window_pid = MagicMock(return_value=4)
        self.skill._get_process_name = MagicMock(return_value="System")
        self.skill._api_get_window_text = MagicMock(return_value="System Root")

        # Even if confirmation were somehow pre-approved:
        token = self.confirmation_manager.request_confirmation(
            operation="close_window",
            target="System Root",
            parameters={},
        )
        self.confirmation_manager.resolve_confirmation(token, approved=True, decided_by="admin")

        cmd = {
            "operation": "close_window",
            "target": "System Root",
            "confirmation_id": token,
        }
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute(cmd)

        self.skill._api_post_wm_close.assert_not_called()

    def test_16_never_terminates_processes_or_calls_shell(self) -> None:
        """Verify window_skills does not import or call process termination APIs or shells."""
        import app.skills.system.window_skills as ws_module
        module_source = ""
        with open(ws_module.__file__, "r", encoding="utf-8") as f:
            module_source = f.read()

        prohibited_tokens = [
            "".join(["Term", "inateProcess"]),
            "".join(["task", "kill"]),
            "".join(["os.", "kill"]),
            "".join(["sub", "process"]),
            "shell=True",
            "".join(["cmd", ".exe"]),
            "".join(["power", "shell"]),
            "".join(["shut", "down"]),
            "".join(["Set", "SuspendState"]),
            "".join(["Exit", "WindowsEx"]),
            "".join(["Initiate", "SystemShutdown"]),
        ]
        for token in prohibited_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, module_source)

    def test_17_invalid_and_stale_hwnd_handled_gracefully(self) -> None:
        """Verify invalid or nonexistent HWND returns clear error without unhandled exception."""
        self.skill._api_is_window = MagicMock(return_value=False)
        self.skill._api_enum_windows = MagicMock(return_value=[])

        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({"operation": "focus_window", "target": "NonExistentApp"})
        self.assertIn("No active window found", str(ctx.exception))

    def test_18_enumeration_strictly_bounded(self) -> None:
        """Verify window enumeration caps at MAX_WINDOW_ENUM_LIMIT even with thousands of handles."""
        # Mock 500 window handles
        mock_handles = list(range(1, 501))
        self.skill._api_enum_windows = MagicMock(return_value=mock_handles)
        self.skill._api_get_window_text = MagicMock(return_value="Dummy Window")

        res = self.skill.execute({"operation": "list_windows"})
        self.assertTrue(res.success)
        self.assertLessEqual(res.data["count"], MAX_WINDOW_ENUM_LIMIT)
        self.assertEqual(res.data["count"], 200)

    def test_19_concurrent_operations_thread_safe(self) -> None:
        """Verify thread safety of WindowSkills across concurrent read and state operations."""
        def run_op(idx: int) -> bool:
            if idx % 2 == 0:
                res = self.skill.execute({"operation": "list_windows"})
                return res.success
            else:
                res = self.skill.execute({"operation": "get_active_window"})
                return res.success

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_op, i) for i in range(40)]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))
        self.assertEqual(len(results), 40)


if __name__ == "__main__":
    unittest.main()
