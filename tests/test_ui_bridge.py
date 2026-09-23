"""Unit Tests for UIBridge and PySide6 Headless Components.

Tests Phase 23.2.1 and 23.2.2:
- UIBridge initialization and presentation adapter binding
- Event draining into Qt signals
- State change signal emission
- Amplitude telemetry signal emission
- Asynchronous command dispatch delegation without blocking Qt
- ArcReactorCore instantiation and state/amplitude setting (headless)
- WaveformVisualizer instantiation and telemetry setting (headless)
"""

from __future__ import annotations

import os
import sys
import time
import unittest.mock as mock
import pytest

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState, PresentationEvent
from app.core.state_manager import AssistantStateManager
from app.ui.bridge import UIBridge
from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.waveform import WaveformVisualizer
from app.ui.styles import JarvisTheme, get_state_color


@pytest.fixture(scope="session")
def qapp():
    """Ensure a headless QApplication instance exists for widget testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_ui_styles_palette_and_state_colors():
    """Verify JarvisTheme constants and get_state_color mapping."""
    assert JarvisTheme.BG_MAIN == "#030812"
    assert JarvisTheme.CYAN_PRIMARY == "#00d4ff"

    assert get_state_color("LISTENING") == JarvisTheme.STATE_LISTENING
    assert get_state_color("THINKING") == JarvisTheme.STATE_THINKING
    assert get_state_color("ERROR") == JarvisTheme.STATE_ERROR
    assert get_state_color("UNKNOWN_STATE") == JarvisTheme.STATE_IDLE


def test_ui_bridge_signals_and_snapshot_sync(qapp):
    """Verify UIBridge emits snapshot_updated and state_changed signals."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)

    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=15)

    snapshots_received = []
    states_received = []
    amplitudes_received = []

    bridge.snapshot_updated.connect(lambda s: snapshots_received.append(s))
    bridge.state_changed.connect(lambda st: states_received.append(st))
    bridge.amplitude_updated.connect(lambda a: amplitudes_received.append(a))

    # Trigger state transition
    state_mgr.transition_to(AssistantState.LISTENING)
    qapp.processEvents()

    # Allow timer tick
    bridge._poll_backend_events()
    qapp.processEvents()

    assert len(snapshots_received) >= 1
    assert "LISTENING" in states_received
    assert bridge.current_snapshot.state == AssistantState.LISTENING

    # Trigger amplitude telemetry
    state_mgr.update_mic_amplitude(0.75)
    bridge._poll_backend_events()
    qapp.processEvents()

    assert any(abs(a - 0.75) < 0.05 for a in amplitudes_received)

    bridge.close()
    adapter.close()
    state_mgr.close()


def test_ui_bridge_submit_command_delegation(qapp):
    """Verify submit_command calls adapter.submit_command asynchronously."""
    state_mgr = AssistantStateManager()
    mock_router = mock.AsyncMock()
    mock_router.route_async.return_value = "Command done"

    adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
    bridge = UIBridge(presentation_adapter=adapter)

    completed_results = []
    bridge.command_completed.connect(lambda res: completed_results.append(res))

    bridge.submit_command("open chrome")

    # Wait briefly for thread pool future to complete
    for _ in range(60):
        qapp.processEvents()
        if completed_results:
            break
        time.sleep(0.05)

    assert len(completed_results) == 1
    assert completed_results[0] == "Command done"

    bridge.close()
    adapter.close()
    state_mgr.close()


def test_ui_bridge_submit_command_system_skill_result_memory_formatting(qapp):
    """Verify submit_command records to_user_message() in memory for SystemSkillResult."""
    from app.skills.system.base_system_skill import SystemSkillResult

    state_mgr = AssistantStateManager()
    mock_router = mock.AsyncMock()
    cpu_res = SystemSkillResult(
        operation="get_cpu_info",
        success=True,
        data={"usage_percent": 25.0, "logical_cores": 4},
    )
    mock_router.route_async.return_value = cpu_res

    adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
    mock_memory = mock.MagicMock()
    bridge = UIBridge(presentation_adapter=adapter, memory_manager=mock_memory)

    completed_results = []
    bridge.command_completed.connect(lambda res: completed_results.append(res))

    bridge.submit_command("check cpu")

    for _ in range(60):
        qapp.processEvents()
        if completed_results:
            break
        time.sleep(0.05)

    assert len(completed_results) == 1
    assert completed_results[0] == cpu_res

    # Give async memory worker a tick
    time.sleep(0.05)
    qapp.processEvents()

    assert mock_memory.add.call_count == 2
    user_call, asst_call = mock_memory.add.call_args_list
    assert user_call.kwargs["content"] == "check cpu"
    assert asst_call.kwargs["content"] == cpu_res.to_user_message()
    assert "SystemSkillResult(" not in asst_call.kwargs["content"]

    bridge.close()
    adapter.close()
    state_mgr.close()


def test_arc_reactor_core_headless_instantiation(qapp):
    """Verify ArcReactorCore instantiates, accepts states and amplitude, and updates."""
    widget = ArcReactorCore()
    assert widget._state == "IDLE"

    widget.set_state("LISTENING")
    assert widget._state == "LISTENING"
    assert widget._primary_color.name().lower() == JarvisTheme.STATE_LISTENING.lower()

    widget.set_amplitude(0.65)
    assert widget._amplitude == 0.65

    # Run animation tick
    widget._on_animation_tick()
    assert widget._smoothed_amplitude > 0.0

    # Ensure paint event executes offscreen without exceptions
    widget.resize(400, 400)
    widget.repaint()

    widget.close()


def test_waveform_visualizer_headless_instantiation(qapp):
    """Verify WaveformVisualizer instantiates, sets amplitude, and paints cleanly."""
    visualizer = WaveformVisualizer(num_bars=32)
    assert visualizer._num_bars == 32

    visualizer.set_state("SPEAKING", "Speaking output...")
    assert visualizer._state == "SPEAKING"
    assert visualizer._status_text == "Speaking output..."

    visualizer.set_amplitude(0.8)
    assert visualizer._target_amplitude == 0.8

    # Run animation tick
    visualizer._on_tick()
    assert visualizer._current_amplitude > 0.0

    visualizer.resize(500, 80)
    visualizer.repaint()

    visualizer.close()
