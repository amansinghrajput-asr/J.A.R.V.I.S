"""Phase 27.29 — Test Suite for Confirmed MVP UX Result-Formatting.

Validates that SystemSkillResult and other command completion results are cleanly
formatted using to_user_message(), .message, or graceful fallbacks, ensuring raw
SystemSkillResult.__repr__() never leaks into the user-facing UI or conversation memory.

Covers:
1. MainWindow result rendering uses to_user_message()
2. UIBridge conversation-memory result uses to_user_message()
3. Fallback behavior for normal strings/objects remains unchanged
4. CPU result is human-readable
5. Memory result is human-readable
6. Raw SystemSkillResult repr does not appear in user-facing conversation text
7. Sync/async behavior remains consistent
8. Win32InputBackend zero physical invocation guarantee (invocation_count == 0)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from unittest import mock

# Ensure headless offscreen Qt platform for test session
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["JARVIS_TEST_MODE"] = "1"

from PySide6.QtWidgets import QApplication

_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

from app.automation.input import Win32InputBackend
from app.core.presentation import PresentationAdapter
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.ui.bridge import UIBridge
from app.ui.components.left_panel import MessageBubble
from app.ui.main_window import JarvisMainWindow


class CustomUserMessageObj:
    """Test double implementing to_user_message()."""

    def __init__(self, msg: str) -> None:
        self._msg = msg

    def to_user_message(self) -> str:
        return self._msg

    def __repr__(self) -> str:
        return f"CustomUserMessageObj(raw={self._msg!r})"


class CustomMessageAttrObj:
    """Test double with string .message attribute."""

    def __init__(self, msg: str) -> None:
        self.message = msg

    def __repr__(self) -> str:
        return f"CustomMessageAttrObj(raw={self.message!r})"


class CustomNonStringMessageAttrObj:
    """Test double with non-string .message attribute."""

    def __init__(self, val: int) -> None:
        self.message = val

    def __repr__(self) -> str:
        return f"CustomNonStringMessageAttrObj({self.message})"


class Phase2729ResultFormattingTests(unittest.TestCase):
    """Deterministic tests for UI and conversation memory result formatting."""

    def setUp(self) -> None:
        super().setUp()
        self.initial_invocations = getattr(Win32InputBackend, "invocation_count", 0)

    def tearDown(self) -> None:
        # Strict safety check: Win32 physical input must never be triggered
        current_invocations = getattr(Win32InputBackend, "invocation_count", 0)
        self.assertEqual(
            current_invocations,
            self.initial_invocations,
            "CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked!",
        )
        super().tearDown()

    def test_01_main_window_rendering_uses_to_user_message(self) -> None:
        """1. MainWindow._on_command_completed uses to_user_message() if available."""
        window = JarvisMainWindow()
        test_obj = CustomUserMessageObj("All systems nominal.")

        window._on_command_completed(test_obj)

        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertGreater(len(bubbles), 0)
        latest_bubble = bubbles[-1]
        self.assertEqual(latest_bubble.text(), "All systems nominal.")
        self.assertNotIn("CustomUserMessageObj", latest_bubble.text())
        window.close()

    def test_02_bridge_conversation_memory_uses_to_user_message(self) -> None:
        """2. UIBridge.submit_command records to_user_message() in conversation memory."""
        state_mgr = AssistantStateManager()
        mock_router = mock.AsyncMock()
        test_result = CustomUserMessageObj("Custom memory response")
        mock_router.route_async.return_value = test_result

        adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
        mock_memory = mock.MagicMock()
        bridge = UIBridge(presentation_adapter=adapter, memory_manager=mock_memory)

        completed_events = []
        bridge.command_completed.connect(lambda res: completed_events.append(res))

        bridge.submit_command("test command")

        for _ in range(30):
            _qapp.processEvents()
            if completed_events:
                break
            time.sleep(0.02)

        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0], test_result)

        # Allow async memory worker thread to record
        time.sleep(0.05)
        _qapp.processEvents()

        self.assertEqual(mock_memory.add.call_count, 2)
        user_call, asst_call = mock_memory.add.call_args_list
        self.assertEqual(user_call.kwargs["content"], "test command")
        self.assertEqual(user_call.kwargs["role"], "user")
        self.assertEqual(asst_call.kwargs["content"], "Custom memory response")
        self.assertEqual(asst_call.kwargs["role"], "assistant")
        self.assertNotIn("CustomUserMessageObj", asst_call.kwargs["content"])

        bridge.close()
        adapter.close()
        state_mgr.close()

    def test_03_fallback_behavior_for_strings_and_objects(self) -> None:
        """3. Fallback behavior remains unchanged for strings, None, and normal objects."""
        window = JarvisMainWindow()

        # Plain string
        window._on_command_completed("Plain text response")
        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertEqual(bubbles[-1].text(), "Plain text response")

        # None -> Operation completed successfully.
        window._on_command_completed(None)
        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertEqual(bubbles[-1].text(), "Operation completed successfully.")

        # Object with .message string attribute
        msg_attr_obj = CustomMessageAttrObj("Message attribute text")
        window._on_command_completed(msg_attr_obj)
        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertEqual(bubbles[-1].text(), "Message attribute text")

        # Object with non-string .message falls back to str(obj)
        non_str_obj = CustomNonStringMessageAttrObj(404)
        window._on_command_completed(non_str_obj)
        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertEqual(bubbles[-1].text(), "CustomNonStringMessageAttrObj(404)")

        # Plain integer fallback
        window._on_command_completed(999)
        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertEqual(bubbles[-1].text(), "999")

        window.close()

    def test_04_cpu_result_is_human_readable(self) -> None:
        """4. CPU SystemSkillResult formats to human-readable text without raw repr."""
        cpu_result = SystemSkillResult(
            operation="get_cpu_info",
            success=True,
            data={
                "usage_percent": 18.5,
                "logical_cores": 16,
                "physical_cores": 8,
                "frequency_mhz": 3200.0,
            },
            duration_ms=45.2,
            metadata={"source": "psutil"},
        )

        window = JarvisMainWindow()
        window._on_command_completed(cpu_result)

        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        rendered_text = bubbles[-1].text()

        self.assertIn("CPU:", rendered_text)
        self.assertIn("Usage: 18.5%", rendered_text)
        self.assertIn("Cores: 16 logical (8 physical)", rendered_text)
        self.assertIn("Frequency: 3200.0 MHz", rendered_text)

        # Must NOT contain raw repr or metadata leaks
        self.assertNotIn("SystemSkillResult(", rendered_text)
        self.assertNotIn("duration_ms", rendered_text)
        self.assertNotIn("psutil", rendered_text)

        window.close()

    def test_05_memory_result_is_human_readable(self) -> None:
        """5. Memory SystemSkillResult formats to human-readable text without raw repr."""
        mem_result = SystemSkillResult(
            operation="get_memory_info",
            success=True,
            data={
                "total_bytes": 17179869184,  # 16 GB
                "used_bytes": 8589934592,    # 8 GB
                "available_bytes": 8589934592,
                "usage_percent": 50.0,
            },
            duration_ms=12.0,
            metadata={"cached": False},
        )

        window = JarvisMainWindow()
        window._on_command_completed(mem_result)

        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        rendered_text = bubbles[-1].text()

        self.assertIn("Memory:", rendered_text)
        self.assertIn("Used:", rendered_text)
        self.assertIn("Available:", rendered_text)

        # Must NOT contain raw repr or metadata leaks
        self.assertNotIn("SystemSkillResult(", rendered_text)
        self.assertNotIn("duration_ms", rendered_text)
        self.assertNotIn("cached", rendered_text)

        window.close()

    def test_06_raw_system_skill_result_repr_never_appears_in_conversation(self) -> None:
        """6. Exhaustively test system skill results to ensure raw repr never leaks."""
        sample_results = [
            SystemSkillResult(
                operation="get_system_summary",
                success=True,
                data={
                    "platform": "Windows",
                    "cpu_usage_percent": 22.0,
                    "memory_usage_percent": 45.0,
                    "battery_percent": 88.0,
                },
            ),
            SystemSkillResult(
                operation="get_disk_info",
                success=True,
                data={
                    "total_bytes": 512000000000,
                    "free_bytes": 256000000000,
                    "usage_percent": 50.0,
                },
            ),
            SystemSkillResult(
                operation="get_battery_info",
                success=True,
                data={
                    "percent": 85.0,
                    "power_plugged": True,
                },
            ),
        ]

        window = JarvisMainWindow()
        for res in sample_results:
            window._on_command_completed(res)
            bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
            rendered_text = bubbles[-1].text()
            self.assertNotIn("SystemSkillResult(", rendered_text)
            self.assertNotIn("operation=", rendered_text)
            self.assertNotIn("success=True", rendered_text)

        window.close()

    def test_07_sync_and_async_behavior_consistency(self) -> None:
        """7. Verify sync direct completion and async bridge submission produce identical text."""
        cpu_result = SystemSkillResult(
            operation="get_cpu_info",
            success=True,
            data={
                "usage_percent": 30.0,
                "logical_cores": 8,
            },
        )

        state_mgr = AssistantStateManager()
        mock_router = mock.AsyncMock()
        mock_router.route_async.return_value = cpu_result
        adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
        mock_memory = mock.MagicMock()
        bridge = UIBridge(presentation_adapter=adapter, memory_manager=mock_memory)

        window = JarvisMainWindow(bridge=bridge)

        # Capture the message text rendered when bridge finishes
        bridge.submit_command("check cpu")

        for _ in range(30):
            _qapp.processEvents()
            bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
            if bubbles:
                break
            time.sleep(0.02)

        time.sleep(0.05)
        _qapp.processEvents()

        bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
        self.assertGreater(len(bubbles), 0)
        async_text = bubbles[-1].text()

        # Direct synchronous call on window
        direct_window = JarvisMainWindow()
        direct_window._on_command_completed(cpu_result)
        direct_bubbles = direct_window.left_panel.conversation_card.findChildren(MessageBubble)
        sync_text = direct_bubbles[-1].text()

        self.assertEqual(async_text, sync_text)
        self.assertIn("CPU:\nUsage: 30.0%\nCores: 8", async_text)
        self.assertNotIn("SystemSkillResult(", async_text)

        # Also verify memory manager received identical text
        self.assertEqual(mock_memory.add.call_count, 2)
        user_call, asst_call = mock_memory.add.call_args_list
        self.assertEqual(user_call.kwargs["content"], "check cpu")
        self.assertEqual(user_call.kwargs["role"], "user")
        self.assertEqual(asst_call.kwargs["content"], async_text)
        self.assertEqual(asst_call.kwargs["role"], "assistant")

        direct_window.close()
        window.close()
        bridge.close()
        adapter.close()
        state_mgr.close()


if __name__ == "__main__":
    unittest.main()
