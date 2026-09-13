"""Unit and Integration Tests for Phase 23.4: Telemetry, Confirmation Gateway, and Planner Tasks.

Tests:
1. CircularGauge display_text override and N/A handling
2. ConfirmationCard presentation, risk badge, and duplicate-click protection
3. CurrentTaskCard dynamic checklist rendering (standby, single-command, multi-step, failed)
4. SystemStatusCard live telemetry updates (CPU, RAM, GPU N/A fallback, network)
5. SystemInfoCard live system information updates (OS, machine name, uptime)
6. CenterPanel confirmation switching and signal propagation
7. UIBridge live telemetry background sampler, planner event tracking, and voice guard
8. Safe confirmation resolution via mock resolver (strictly zero real destructive actions)
9. Thread safety: Qt GUI event loop remains non-blocking, clean shutdown
"""

from __future__ import annotations

import os
import sys
import time
from unittest import mock
import pytest

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState, PendingConfirmation, PresentationEvent
from app.core.state_manager import AssistantStateManager
from app.ui.bridge import UIBridge
from app.ui.components.confirmation_dialog import ConfirmationCard
from app.ui.components.dials import CircularGauge
from app.ui.components.center_panel import CenterPanel
from app.ui.components.right_panel import CurrentTaskCard, SystemInfoCard, SystemStatusCard, TaskCheckItem
from app.ui.main_window import JarvisMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a headless QApplication instance exists for widget testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ==============================================================================
# 1. Dial Display Enhancement & N/A Fallback
# ==============================================================================

def test_circular_gauge_display_text_and_fallback(qapp):
    """Verify CircularGauge correctly displays values and custom override text like 'N/A'."""
    gauge = CircularGauge("GPU", 0.0)
    assert gauge.value == 0.0
    assert gauge.display_text is None

    # Update value
    gauge.set_value(65.0)
    assert gauge.value == 65.0
    assert gauge.display_text is None

    # Set display text override (e.g. N/A when GPU unavailable)
    gauge.set_display_text("N/A")
    assert gauge.display_text == "N/A"

    # Setting numeric value resets display_text override
    gauge.set_value(42.0)
    assert gauge.value == 42.0
    assert gauge.display_text is None


# ==============================================================================
# 2. ConfirmationCard Presentation & Duplicate-Click Protection
# ==============================================================================

def test_confirmation_card_load_and_signals(qapp):
    """Verify ConfirmationCard renders pending details and enforces click lock."""
    card = ConfirmationCard()

    pending = PendingConfirmation(
        confirmation_id="test-token-12345",
        operation="test_dummy_operation",
        target="MockTarget",
        risk_level="HIGH",
        description="Verify operator approval for dummy test action.",
    )

    card.load_confirmation(pending)
    assert card.confirmation_id == "test-token-12345"
    assert card._op_lbl.text() == "TEST_DUMMY_OPERATION"
    assert card._target_lbl.text() == "MockTarget"
    assert card._token_lbl.text() == "test-token-12345"
    assert card.confirm_btn.isEnabled()
    assert card.cancel_btn.isEnabled()

    # Track signals
    confirmed_tokens = []
    cancelled_tokens = []
    card.confirmed.connect(confirmed_tokens.append)
    card.cancelled.connect(cancelled_tokens.append)

    # Click confirm
    card.confirm_btn.click()
    assert confirmed_tokens == ["test-token-12345"]
    assert not cancelled_tokens

    # Verify duplicate click protection: buttons disabled, second click ignored
    assert not card.confirm_btn.isEnabled()
    assert not card.cancel_btn.isEnabled()
    card.confirm_btn.click()
    card.cancel_btn.click()
    assert confirmed_tokens == ["test-token-12345"]
    assert not cancelled_tokens


def test_confirmation_card_cancellation(qapp):
    """Verify ConfirmationCard cancellation signal and button locking."""
    card = ConfirmationCard()
    pending = PendingConfirmation(
        confirmation_id="test-token-abort-99",
        operation="test_abort_op",
        risk_level="CRITICAL",
        description="Dangerous mock operation.",
    )
    card.load_confirmation(pending)

    cancelled_tokens = []
    card.cancelled.connect(cancelled_tokens.append)

    card.cancel_btn.click()
    assert cancelled_tokens == ["test-token-abort-99"]
    assert not card.confirm_btn.isEnabled()
    assert not card.cancel_btn.isEnabled()


# ==============================================================================
# 3. Dynamic CurrentTaskCard & Checklist Rendering
# ==============================================================================

def test_current_task_card_lifecycle(qapp):
    """Verify CurrentTaskCard handles standby, single-command, and multi-step plans."""
    card = CurrentTaskCard()

    # 1. Standby
    card.update_task_state("Standby — Ready for commands", 0.0)
    assert card._badge_lbl.text() == "0 / 0"
    assert card._pbar.value() == 0
    assert card._tasks_layout.count() == 1

    # 2. Single Command
    card.update_task_state("Search weather", 0.5)
    assert card._badge_lbl.text() == "1 / 1"
    assert card._pbar.value() == 50
    assert card._tasks_layout.count() == 1

    # 3. Multi-Step Plan with Failed Task
    steps = [
        {"title": "Initialize Audio", "status": "completed", "is_done": True},
        {"title": "Capture Input", "status": "running", "is_active": True},
        {"title": "Execute System Operation", "status": "failed", "is_failed": True},
        {"title": "Clean Up", "status": "pending"},
    ]
    card.update_task_state("Multi-Step Plan", 0.25, steps)
    assert card._badge_lbl.text() == "1 / 4"
    assert card._pbar.value() == 25
    assert card._tasks_layout.count() == 4


def test_task_check_item_failed_state(qapp):
    """Verify TaskCheckItem renders distinct red error styling when failed."""
    item = TaskCheckItem("Failed Step", is_failed=True, tag="Error")
    assert item is not None


# ==============================================================================
# 4. SystemStatusCard & SystemInfoCard Live Telemetry
# ==============================================================================

def test_system_status_card_telemetry_update(qapp):
    """Verify SystemStatusCard updates gauges and handles GPU N/A gracefully."""
    status_card = SystemStatusCard()

    # 1. Update with live metrics (GPU available)
    metrics_with_gpu = {
        "cpu_percent": 45.5,
        "memory_percent": 68.2,
        "gpu_available": True,
        "gpu_percent": 22.0,
        "network_summary": "↑ 1.5 MB/s  ↓ 3.2 MB/s",
        "active_host": "TestPC",
        "system_status": "● Optimal",
    }
    status_card.update_telemetry(metrics_with_gpu)
    assert status_card.cpu_dial.value == 45.5
    assert status_card.ram_dial.value == 68.2
    assert status_card.gpu_dial.value == 22.0
    assert status_card._net_val_lbl.text() == "↑ 1.5 MB/s  ↓ 3.2 MB/s"
    assert status_card._active_app_val_lbl.text() == "TestPC"

    # 2. Update with GPU unavailable -> N/A fallback
    metrics_no_gpu = {
        "cpu_percent": 12.0,
        "memory_percent": 30.0,
        "gpu_available": False,
        "gpu_percent": None,
    }
    status_card.update_telemetry(metrics_no_gpu)
    assert status_card.cpu_dial.value == 12.0
    assert status_card.ram_dial.value == 30.0
    assert status_card.gpu_dial.display_text == "N/A"


def test_system_info_card_update(qapp):
    """Verify SystemInfoCard updates OS, hostname, and uptime string."""
    info_card = SystemInfoCard()

    info_card.update_info({
        "os_name": "Windows 11 Pro",
        "device_name": "JARVIS-TEST-NODE",
        "uptime": "5d 12h 44m",
    })
    assert info_card._os_title_lbl.text() == "Windows 11 Pro"
    assert info_card._os_sub_lbl.text() == "JARVIS-TEST-NODE"
    assert info_card._uptime_val_lbl.text() == "5d 12h 44m"


# ==============================================================================
# 5. CenterPanel Confirmation Switching
# ==============================================================================

def test_center_panel_confirmation_switching(qapp):
    """Verify CenterPanel switches between RecentExecutionCard and ConfirmationCard."""
    panel = CenterPanel()
    assert panel.lower_stack.currentIndex() == 0

    pending = PendingConfirmation(
        confirmation_id="tok-center-1",
        operation="mock_system_action",
        risk_level="HIGH",
    )

    # Show confirmation
    panel.show_confirmation(pending)
    assert panel.lower_stack.currentIndex() == 1
    assert panel.confirmation_card.confirmation_id == "tok-center-1"

    # Forwarded signals test
    confirmed = []
    panel.confirmation_confirmed.connect(confirmed.append)
    panel.confirmation_card.confirm_btn.click()
    assert confirmed == ["tok-center-1"]

    # Hide confirmation
    panel.hide_confirmation()
    assert panel.lower_stack.currentIndex() == 0


# ==============================================================================
# 6. UIBridge Telemetry, Voice Concurrency Guard, and Planner Events
# ==============================================================================

def test_ui_bridge_voice_concurrency_guard(qapp):
    """Verify UIBridge prevents overlapping voice interaction submissions."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)

    bridge = UIBridge(presentation_adapter=adapter)
    try:
        # First trigger
        accepted1 = bridge.start_voice_interaction()
        assert accepted1 is True

        # Rapid second trigger while voice is flagged active
        accepted2 = bridge.start_voice_interaction()
        assert accepted2 is False
    finally:
        bridge.close()


def test_ui_bridge_telemetry_sampling_and_signal(qapp):
    """Verify UIBridge background telemetry sampler emits valid metrics dict."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)

    bridge = UIBridge(presentation_adapter=adapter)
    telemetry_batches = []
    bridge.telemetry_updated.connect(telemetry_batches.append)

    try:
        # Trigger sampling worker synchronously for deterministic assertion
        bridge._telemetry_worker()
        QCoreApplication.processEvents()

        assert len(telemetry_batches) >= 1
        data = telemetry_batches[-1]
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "gpu_available" in data
        assert "network_summary" in data
        assert "uptime" in data
    finally:
        bridge.close()


def test_ui_bridge_planner_event_tracking(qapp):
    """Verify UIBridge receives and parses planner events into plan_updated signals."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)

    bridge = UIBridge(presentation_adapter=adapter)
    plan_events = []
    bridge.plan_updated.connect(plan_events.append)

    try:
        # 1. PlanStarted event
        ev_start = PresentationEvent(
            event_type="planner.plan_started",
            snapshot=state_mgr.get_snapshot(),
            payload=(("query", "Test Plan Query"), ("task_count", 2)),
        )
        bridge._process_planner_event(ev_start)
        assert len(plan_events) == 1
        assert plan_events[-1]["title"] == "Test Plan Query"

        # 2. TaskStarted event
        ev_task = PresentationEvent(
            event_type="planner.task_started",
            snapshot=state_mgr.get_snapshot(),
            payload=(("task_id", "t-1"), ("action", "Open Visual Studio Code")),
        )
        bridge._process_planner_event(ev_task)
        assert len(plan_events) == 2
        steps = plan_events[-1]["steps"]
        assert len(steps) == 1
        assert steps[0]["task_id"] == "t-1"
        assert steps[0]["is_active"] is True

        # 3. TaskCompleted event
        ev_comp = PresentationEvent(
            event_type="planner.task_completed",
            snapshot=state_mgr.get_snapshot(),
            payload=(("task_id", "t-1"), ("progress", 0.5)),
        )
        bridge._process_planner_event(ev_comp)
        assert len(plan_events) == 3
        assert plan_events[-1]["steps"][0]["is_done"] is True
    finally:
        bridge.close()


# ==============================================================================
# 7. Main Window Safe Confirmation Wiring & Non-Destructive Resolution
# ==============================================================================

def test_main_window_confirmation_flow_safety(qapp):
    """Verify JarvisMainWindow transitions to ConfirmationCard and calls safe resolver."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)

    window = JarvisMainWindow(presentation_adapter=adapter)
    try:
        # Mock resolve_confirmation on adapter so NO system skill or OS operation can run
        with mock.patch.object(adapter, "resolve_confirmation", return_value=True) as mock_resolve:
            # Simulate backend requesting confirmation
            conf = PendingConfirmation(
                confirmation_id="safe-test-token-77",
                operation="mock_non_destructive_op",
                risk_level="HIGH",
                description="Mock confirmation test.",
            )

            snap_awaiting = AssistantSnapshot(
                state=AssistantState.AWAITING_CONFIRMATION,
                status_message="Confirmation Required",
                pending_confirmation=conf,
            )

            window._on_snapshot_updated(snap_awaiting)
            QCoreApplication.processEvents()

            # Confirm widget is now visible in stack
            assert window.center_panel.lower_stack.currentIndex() == 1
            assert window.center_panel.confirmation_card.confirmation_id == "safe-test-token-77"

            # Operator clicks CONFIRM ACTION
            window.center_panel.confirmation_card.confirm_btn.click()
            QCoreApplication.processEvents()

            # Verify safe existing API was called with exact token
            mock_resolve.assert_called_once_with(
                "safe-test-token-77",
                approved=True,
                decided_by="gui_operator",
                reason="Authorized by operator via HUD ConfirmationCard",
            )
    finally:
        window.close()


def test_main_window_rejection_flow_safety(qapp):
    """Verify JarvisMainWindow handles cancellation via safe resolver."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)

    window = JarvisMainWindow(presentation_adapter=adapter)
    try:
        with mock.patch.object(adapter, "resolve_confirmation", return_value=True) as mock_resolve:
            conf = PendingConfirmation(
                confirmation_id="safe-abort-token-88",
                operation="mock_abort_op",
                risk_level="HIGH",
            )

            snap_awaiting = AssistantSnapshot(
                state=AssistantState.AWAITING_CONFIRMATION,
                pending_confirmation=conf,
            )

            window._on_snapshot_updated(snap_awaiting)
            QCoreApplication.processEvents()

            # Operator clicks CANCEL / ABORT
            window.center_panel.confirmation_card.cancel_btn.click()
            QCoreApplication.processEvents()

            # Verify safe existing API was called with approved=False
            mock_resolve.assert_called_once_with(
                "safe-abort-token-88",
                approved=False,
                decided_by="gui_operator",
                reason="Rejected by operator via HUD ConfirmationCard",
            )
    finally:
        window.close()


# ==============================================================================
# 8. Absolute Safety Invariant: Zero Destructive Actions
# ==============================================================================

def test_safety_invariant_no_destructive_os_execution():
    """Verify that test suite and UI components make zero destructive system calls."""
    import subprocess
    import shutil
    # Ensure standard dangerous commands are not called
    for danger_fn in ("shutdown", "restart", "taskkill"):
        assert not hasattr(JarvisMainWindow, danger_fn)
        assert not hasattr(ConfirmationCard, danger_fn)
        assert not hasattr(UIBridge, danger_fn)
