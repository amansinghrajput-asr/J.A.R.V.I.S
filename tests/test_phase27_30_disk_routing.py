"""Phase 27.30 — Focused Deterministic Tests for Disk Telemetry Routing.

Verifies:
1. All disk command variants map deterministically to Task(action="get_disk_info"):
   - "check disk"
   - "check disk."
   - "Check disc."
   - "check disc"
   - "check the disk"
   - "check the disc"
   - "check disk info"
   - "disk info"
   - "disk space"
   - "check the disk space"
   - "disk usage"
   - "disk status"
   - "disk drives"
2. Visual toggle commands strictly remain visual_toggle:
   - "check the checkbox"
   - "check the agree terms checkbox"
   - "uncheck the box"
   - "toggle notifications"
   - "switch the toggle button"
3. SystemInfoSkills matches "disc" alongside "disk"
4. Win32InputBackend invocation count strictly remains 0
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure headless test environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["JARVIS_TEST_MODE"] = "1"

from PySide6.QtWidgets import QApplication

_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

from app.automation.input import Win32InputBackend
from app.ai.planner.planner import Planner
from app.skills.system.system_info_skills import SystemInfoSkills


class TestPhase2730DiskRouting(unittest.TestCase):
    """Deterministic validation of disk telemetry routing and visual-toggle preservation."""

    def setUp(self) -> None:
        super().setUp()
        self.initial_invocations = getattr(Win32InputBackend, "invocation_count", 0)
        self.planner = Planner()
        self.system_info = SystemInfoSkills()

    def tearDown(self) -> None:
        current_invocations = getattr(Win32InputBackend, "invocation_count", 0)
        self.assertEqual(
            current_invocations,
            self.initial_invocations,
            "CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked!",
        )
        super().tearDown()

    def test_01_all_disk_command_variants_map_to_get_disk_info(self) -> None:
        """1. Verify all required disk command variants route to get_disk_info."""
        disk_queries = [
            "check disk",
            "check disk.",
            "Check disc.",
            "check disc",
            "check the disk",
            "check the disc",
            "check disk info",
            "disk info",
            "disk space",
            "check the disk space",
            "disk usage",
            "disk status",
            "disk drives",
        ]

        for q in disk_queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(
                    len(plan.tasks),
                    1,
                    f"Expected exactly 1 task for query '{q}', got {len(plan.tasks)}",
                )
                task = plan.tasks[0]
                self.assertEqual(
                    task.action,
                    "get_disk_info",
                    f"Query '{q}' routed to '{task.action}' instead of 'get_disk_info'",
                )
                self.assertIsNone(task.target)

    def test_02_visual_toggle_commands_remain_visual_toggle(self) -> None:
        """2. Verify visual UI toggle commands are NOT hijacked and remain visual_toggle."""
        visual_queries = [
            ("check the checkbox", "checkbox"),
            ("check the agree terms checkbox", "agree terms"),
            ("uncheck the box", "box"),
            ("toggle notifications", "notifications"),
            ("switch the toggle button", "toggle"),
        ]

        for q, expected_target in visual_queries:
            with self.subTest(query=q):
                plan = self.planner.create_plan(q)
                self.assertEqual(
                    len(plan.tasks),
                    1,
                    f"Expected exactly 1 task for query '{q}', got {len(plan.tasks)}",
                )
                task = plan.tasks[0]
                self.assertEqual(
                    task.action,
                    "visual_toggle",
                    f"Query '{q}' routed to '{task.action}' instead of 'visual_toggle'",
                )
                self.assertEqual(task.target, expected_target)

    def test_03_system_info_skills_handles_disc_alongside_disk(self) -> None:
        """3. Verify SystemInfoSkills directly recognizes 'disc' alongside 'disk'."""
        disc_queries = [
            "check disk",
            "check disc",
            "disk info",
            "disc info",
            "disk space",
            "disc space",
            "how much disk space is available",
            "how much disc space is available",
            "storage space",
        ]

        for q in disc_queries:
            with self.subTest(query=q):
                op, target, params, conf = self.system_info.parse_command(q)
                self.assertEqual(
                    op,
                    "get_disk_info",
                    f"SystemInfoSkills parsed '{q}' as op='{op}' instead of 'get_disk_info'",
                )
                self.assertTrue(
                    self.system_info.can_handle(q),
                    f"SystemInfoSkills.can_handle('{q}') returned False",
                )


if __name__ == "__main__":
    unittest.main()
