"""Unit Tests for Full PySide6 Desktop HUD Assembly.

Phase 23.2.4:
- HeaderBar, LeftPanel, CenterPanel, RightPanel, BottomBar instantiation
- CircularGauge radial dial rendering and value clamping
- GlassPanel container creation
- JarvisMainWindow layout assembly and component hierarchy
- UIBridge signal-to-slot wiring and state transitions
- Command submission flow from QuickActions and BottomBar
- Graceful shutdown on closeEvent
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock
import pytest

from PySide6.QtWidgets import QApplication

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState
from app.core.state_manager import AssistantStateManager
from app.ui.bridge import UIBridge
from app.ui.components.bottom_bar import BottomBar
from app.ui.components.center_panel import CenterPanel, StateIndicatorPills
from app.ui.components.dials import CircularGauge
from app.ui.components.glass_panel import GlassPanel
from app.ui.components.header import HeaderBar
from app.ui.components.left_panel import ConversationCard, LeftPanel, QuickActionsCard
from app.ui.components.right_panel import CurrentTaskCard, RightPanel, SystemStatusCard
from app.ui.main_window import JarvisMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a headless QApplication instance exists for widget testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_glass_panel_and_header(qapp):
    """Test GlassPanel creation, styling, and header addition."""
    panel = GlassPanel()
    assert panel.objectName() == "GlassPanel"
    header_layout = panel.add_card_header("TEST SECTION", icon_text="⚙")
    assert header_layout is not None
    assert panel.content_layout.count() >= 1


def test_circular_gauge_clamping_and_label(qapp):
    """Test CircularGauge value clamping and label updates."""
    gauge = CircularGauge(label="CPU", value=45.0)
    assert gauge.value == 45.0

    gauge.set_value(150.0)
    assert gauge.value == 100.0

    gauge.set_value(-20.0)
    assert gauge.value == 0.0

    gauge.set_label("GPU")
    assert gauge._label == "GPU"


def test_header_bar_tabs_and_status(qapp):
    """Test HeaderBar navigation tabs and status indicator updates."""
    header = HeaderBar()
    assert header.height() == 56

    captured_nav = []
    header.navigation_changed.connect(captured_nav.append)

    header._on_nav_clicked("ACTIVITY")
    assert captured_nav == ["ACTIVITY"]

    header.set_system_status(False, "OFFLINE")
    assert "OFFLINE" in header._online_badge.text()

    header.set_system_status(True, "SYSTEM ONLINE")
    assert "SYSTEM ONLINE" in header._online_badge.text()


def test_left_panel_conversation_and_quick_actions(qapp):
    """Test LeftPanel adding conversation messages and triggering quick actions."""
    left = LeftPanel()
    assert left.width() == 290

    # Add message
    left.conversation_card.add_message("You", "Hello J.A.R.V.I.S")
    assert left.conversation_card._message_count == 1
    assert left.conversation_card._messages_layout.count() >= 2

    # Quick action trigger
    dispatched = []
    left.action_dispatched.connect(dispatched.append)
    left.quick_actions_card.action_triggered.emit("open chrome")
    assert dispatched == ["open chrome"]


def test_center_panel_state_and_amplitude(qapp):
    """Test CenterPanel propagates state and amplitude to arc reactor and waveform."""
    center = CenterPanel()
    center.set_state("THINKING")
    assert center.arc_reactor.state == "THINKING"
    assert center.waveform._state == "THINKING"
    assert center.state_pills._active_state == "THINKING"

    center.set_amplitude(0.65)
    assert center.arc_reactor._amplitude == 0.65
    assert center.waveform._target_amplitude == 0.65


def test_right_panel_task_and_dials(qapp):
    """Test RightPanel task update and gauge dial values."""
    right = RightPanel()
    assert right.width() == 340

    right.current_task_card.set_task("Analyze code repository", 0.90)
    assert right.current_task_card._pbar.value() == 90
    assert right.current_task_card._title_lbl.text() == "Analyze code repository"

    assert right.system_status_card.cpu_dial.value == 32.0
    assert right.system_status_card.ram_dial.value == 48.0


def test_bottom_bar_send_and_status(qapp):
    """Test BottomBar input text emission and status text."""
    bar = BottomBar()
    captured_commands = []
    bar.command_submitted.connect(captured_commands.append)

    bar._input_edit.setText("test command")
    bar._on_send()

    assert captured_commands == ["test command"]
    assert bar._input_edit.text() == ""

    bar.set_status_message("System processing...")
    assert bar._status_lbl.text() == "System processing..."


def test_main_window_assembly_and_bridge_integration(qapp):
    """Test JarvisMainWindow complete layout assembly and bridge signal bindings."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)

    window = JarvisMainWindow(bridge=bridge)
    assert window.windowTitle() == "J.A.R.V.I.S — Tactical AI Assistant"
    assert window.header is not None
    assert window.left_panel is not None
    assert window.center_panel is not None
    assert window.right_panel is not None
    assert window.bottom_bar is not None

    # Test state transition from bridge
    bridge.state_changed.emit("SPEAKING")
    assert window.center_panel.arc_reactor.state == "SPEAKING"
    assert "SPEAKING" in window.bottom_bar._status_lbl.text()

    # Test amplitude update from bridge
    bridge.amplitude_updated.emit(0.70)
    assert window.center_panel.arc_reactor._amplitude == 0.70

    # Test command submission dispatch
    with mock.patch.object(bridge, "submit_command") as mock_submit:
        window._on_command_dispatched("open terminal")
        mock_submit.assert_called_once_with("open terminal")

    # Test command completed handler with string
    window._on_command_completed("Terminal launched.")
    assert window.center_panel.arc_reactor.state == "IDLE"

    # Test command completed handler with SystemSkillResult
    from app.skills.system.base_system_skill import SystemSkillResult
    from app.ui.components.left_panel import MessageBubble

    cpu_result = SystemSkillResult(
        operation="get_cpu_info",
        success=True,
        data={"usage_percent": 15.0, "logical_cores": 8},
    )
    window._on_command_completed(cpu_result)
    bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
    assert len(bubbles) >= 2
    assert "CPU:\nUsage: 15.0%\nCores: 8" in bubbles[-1].text()
    assert "SystemSkillResult(" not in bubbles[-1].text()

    # Test command completed handler with None
    window._on_command_completed(None)
    bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
    assert bubbles[-1].text() == "Operation completed successfully."

    # Test command failed handler
    window._on_command_failed("Launch failed.")
    assert window.center_panel.arc_reactor.state == "ERROR"

    window.close()
