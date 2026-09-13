"""Phase 25 Tests — Audio-Reactive HUD and Real-Time Telemetry Propagation.

Verifies:
1. WaveformVisualizer: set_amplitude updates target amplitude and smoothly interpolates bars.
2. ArcReactorCore: set_amplitude modulates smoothed amplitude and audio boost cleanly.
3. CenterPanel: coordinates both ArcReactor and WaveformVisualizer with incoming amplitude.
4. UIBridge: drains PresentationAdapter snapshots and dispatches amplitude_updated signals.
5. Concurrency guard: duplicate/concurrent voice interactions are safely rejected.
6. Headless rendering: full HUD window and components render without display server.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.ui.bridge import UIBridge
from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.center_panel import CenterPanel
from app.ui.components.waveform import WaveformVisualizer
from app.ui.main_window import JarvisMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure an offscreen headless QApplication exists for all UI tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_waveform_amplitude_smoothing_and_interpolation(qapp):
    """Test WaveformVisualizer set_amplitude, clamping, and 60 FPS tick smoothing."""
    waveform = WaveformVisualizer(num_bars=32)

    # Initial state
    assert waveform._target_amplitude == 0.0
    assert waveform._current_amplitude == 0.0

    # Set active state and high amplitude
    waveform.set_state("LISTENING")
    waveform.set_amplitude(0.85)
    assert waveform._target_amplitude == 0.85

    # Simulate animation ticks
    prev_amp = waveform._current_amplitude
    for _ in range(5):
        waveform._on_tick()
        assert waveform._current_amplitude > prev_amp
        prev_amp = waveform._current_amplitude

    assert waveform._current_amplitude > 0.5
    # Center bars should rise higher than boundary bars
    mid_bar = waveform._bar_heights[len(waveform._bar_heights) // 2]
    edge_bar = waveform._bar_heights[0]
    assert mid_bar > edge_bar

    # Test clamping
    waveform.set_amplitude(1.5)
    assert waveform._target_amplitude == 1.0
    waveform.set_amplitude(-0.5)
    assert waveform._target_amplitude == 0.0

    # Smooth decay back to zero
    for _ in range(25):
        waveform._on_tick()
    assert waveform._current_amplitude < 0.05

    waveform._timer.stop()


def test_arc_reactor_amplitude_boost_and_render(qapp):
    """Test ArcReactorCore set_amplitude, smoothed tracking, and headless painting."""
    core = ArcReactorCore()
    core.resize(300, 300)
    assert core._amplitude == 0.0
    assert core._smoothed_amplitude == 0.0

    # Set listening state and loud amplitude
    core.set_state("LISTENING")
    core.set_amplitude(0.9)
    assert core._amplitude == 0.9

    # Advance animation ticks
    for _ in range(6):
        core._on_animation_tick()

    assert core._smoothed_amplitude > 0.4

    # Test paintEvent headless on an offscreen QPixmap buffer
    pixmap = QPixmap(300, 300)
    core.render(pixmap)
    assert not pixmap.isNull()

    # Reset amplitude to 0.0
    core.set_amplitude(0.0)
    for _ in range(25):
        core._on_animation_tick()
    assert core._smoothed_amplitude < 0.05

    core._anim_timer.stop()


def test_center_panel_amplitude_forwarding(qapp):
    """Test CenterPanel receives amplitude updates and forwards to both core and waveform."""
    panel = CenterPanel()
    panel.set_state("LISTENING")

    panel.set_amplitude(0.72)
    assert panel.waveform._target_amplitude == 0.72
    assert panel.arc_reactor._amplitude == 0.72

    panel.set_amplitude(0.0)
    assert panel.waveform._target_amplitude == 0.0
    assert panel.arc_reactor._amplitude == 0.0

    panel.arc_reactor._anim_timer.stop()
    panel.waveform._timer.stop()


@patch.object(UIBridge, "fetch_system_diagnostics", lambda self: None)
@patch.object(UIBridge, "fetch_ai_telemetry", lambda self: None)
def test_ui_bridge_amplitude_propagation(qapp):
    """Test full telemetry propagation: StateManager -> PresentationAdapter -> UIBridge -> Signal."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)
    bridge = UIBridge(presentation_adapter=adapter)

    received_amps: list[float] = []
    bridge.amplitude_updated.connect(lambda a: received_amps.append(a))

    # Trigger microphone amplitude update
    state_mgr.update_mic_amplitude(0.68)

    # Force event processing on bridge
    bridge._poll_backend_events()

    assert len(received_amps) >= 1
    assert any(abs(a - 0.68) < 0.01 for a in received_amps)

    # Trigger reset to 0.0
    state_mgr.update_mic_amplitude(0.0)
    bridge._poll_backend_events()

    assert any(abs(a - 0.0) < 0.01 for a in received_amps)

    bridge.close()
    state_mgr.close()


@patch.object(UIBridge, "fetch_system_diagnostics", lambda self: None)
@patch.object(UIBridge, "fetch_ai_telemetry", lambda self: None)
def test_voice_concurrency_guard(qapp):
    """Test that concurrent voice interactions are rejected safely."""
    state_mgr = AssistantStateManager()
    mock_engine = MagicMock()
    async def _dummy_listen(duration=None):
        import asyncio
        await asyncio.sleep(0.05)
        return "mock result"

    mock_engine.listen_once_async = _dummy_listen

    adapter = PresentationAdapter(state_manager=state_mgr, voice_engine=mock_engine)
    bridge = UIBridge(presentation_adapter=adapter)

    # First trigger should be accepted
    first = bridge.start_voice_interaction(duration=1.0)
    assert first is True
    assert bridge._voice_active is True

    # Immediate second trigger must be safely rejected by concurrency guard
    second = bridge.start_voice_interaction(duration=1.0)
    assert second is False

    bridge.close()
    state_mgr.close()


@patch.object(UIBridge, "fetch_system_diagnostics", lambda self: None)
@patch.object(UIBridge, "fetch_ai_telemetry", lambda self: None)
def test_main_window_audio_reactive_pipeline_headless(qapp):
    """Test full MainWindow HUD assembly and end-to-end amplitude reactivity."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)
    bridge = UIBridge(presentation_adapter=adapter)

    window = JarvisMainWindow(bridge=bridge)
    window.resize(1280, 720)
    window.show()

    # Verify center panel connection
    center_panel = window.center_panel
    assert center_panel is not None

    # Update state to LISTENING and send real amplitude
    state_mgr.transition_to(AssistantState.LISTENING, status_message="Listening...")
    state_mgr.update_mic_amplitude(0.77)
    bridge._poll_backend_events()

    assert center_panel.waveform._target_amplitude == 0.77
    assert center_panel.arc_reactor._amplitude == 0.77

    # Headless paint verification
    pixmap = QPixmap(1280, 720)
    window.render(pixmap)
    assert not pixmap.isNull()

    # Reset back to IDLE
    state_mgr.transition_to(AssistantState.IDLE, status_message="Ready")
    state_mgr.update_mic_amplitude(0.0)
    bridge._poll_backend_events()

    assert center_panel.waveform._target_amplitude == 0.0
    assert center_panel.arc_reactor._amplitude == 0.0

    window.close()
    bridge.close()
    state_mgr.close()
