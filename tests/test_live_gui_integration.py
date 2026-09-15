"""Integration Tests for Live Backend <-> PySide6 HUD Integration.

Phase 23.3:
- Verification of gui.py entry point and component wiring
- Headless offscreen initialization
- Asynchronous command dispatch through UIBridge -> PresentationAdapter -> Router
- Signal emission on command completion and failure
- State transition propagation from EventBus to HUD widgets
- Voice interaction delegation to VoiceConversationEngine without microphone lockup
- Graceful shutdown without hanging or blocking the event loop
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest.mock as mock
import pytest

from PySide6.QtWidgets import QApplication

import gui
from app.application import JarvisApplication
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.core.presentation import PresentationAdapter, create_presentation_adapter
from app.core.state import AssistantSnapshot, AssistantState
from app.core.state_manager import AssistantStateManager
from app.router.router import CommandRouter
from app.ui.bridge import UIBridge
from app.ui.main_window import JarvisMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a headless QApplication instance exists for offscreen testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_gui_module_import():
    """1. Verify gui module imports and exports setup_gui_components and main."""
    assert hasattr(gui, "setup_gui_components")
    assert hasattr(gui, "shutdown_gui")
    assert hasattr(gui, "main")
    assert hasattr(gui, "parse_gui_args")


def test_gui_args_parsing():
    """Verify argument parsing in gui.py."""
    args = gui.parse_gui_args(["--fullscreen", "--no-banner"])
    assert args.fullscreen is True
    assert args.no_banner is True


def test_gui_components_setup_and_wiring(qapp):
    """2 & 3 & 4. Verify setup_gui_components creates live backend, adapter, bridge, and window without run()."""
    # Create isolated container to prevent polluting global container in test
    container = ServiceContainer()
    backend = JarvisApplication(
        container=container,
        voice_mode=False,
        auto_discover_skills=False,
        print_ready=False,
    )
    assert not backend.is_running  # run() must NOT be called

    q_app, window, bridge, app_inst, adapter = gui.setup_gui_components(
        ["--no-banner"],
        app_instance=backend,
    )

    assert q_app is not None
    assert isinstance(window, JarvisMainWindow)
    assert isinstance(bridge, UIBridge)
    assert app_inst is backend
    assert isinstance(adapter, PresentationAdapter)
    assert window.bridge is bridge

    # Verify HUD has components assembled
    assert window.center_panel.arc_reactor is not None
    assert window.center_panel.waveform is not None

    gui.shutdown_gui(bridge=bridge, backend=backend, adapter=adapter)
    window.close()


def test_uibridge_async_command_dispatch_and_completion(qapp):
    """5 & 6. Verify command travels through UI -> UIBridge -> Adapter -> Router asynchronously."""
    mock_router = mock.MagicMock()

    async def _mock_route(cmd, source="gui"):
        await asyncio.sleep(0.02)
        return f"Echo: {cmd}"

    mock_router.route_async = mock.AsyncMock(side_effect=_mock_route)

    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    completed_results = []
    bridge.command_completed.connect(completed_results.append)

    # Dispatch command from bottom bar
    window.bottom_bar._input_edit.setText("test command")
    window.bottom_bar._on_send()

    # Process Qt events until completed (max 5 seconds)
    start_time = time.perf_counter()
    while not completed_results and (time.perf_counter() - start_time) < 5.0:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(completed_results) == 1
    assert completed_results[0] == "Echo: test command"
    mock_router.route_async.assert_called_once_with("test command", source="gui")

    # Verify conversation card received response
    assert any("Echo: test command" in str(getattr(w, "text", lambda: "")())
               for w in window.left_panel.conversation_card.findChildren(object))

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


def test_uibridge_command_failure_handling(qapp):
    """7. Verify backend failure reaches command_failed and triggers ERROR state."""
    mock_router = mock.MagicMock()

    async def _failing_route(cmd, source="gui"):
        await asyncio.sleep(0.01)
        raise RuntimeError("Service unavailable")

    mock_router.route_async = mock.AsyncMock(side_effect=_failing_route)

    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    failed_errors = []
    bridge.command_failed.connect(failed_errors.append)

    window._on_command_dispatched("bad command")

    start_time = time.perf_counter()
    while not failed_errors and (time.perf_counter() - start_time) < 5.0:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(failed_errors) == 1
    assert "Service unavailable" in failed_errors[0]
    assert window.center_panel.arc_reactor.state == "ERROR"

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


def test_assistant_state_transition_propagates_to_hud(qapp):
    """8. Verify state transitions from EventBus update ArcReactorCore and HUD pills."""
    event_bus = EventBus()
    state_mgr = AssistantStateManager(event_bus=event_bus, initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    assert window.center_panel.arc_reactor.state == "IDLE"

    # Simulate backend event publishing
    event_bus.publish("voice_engine.recording", payload={"duration": 3.0}, source="voice_engine")

    # Allow Qt polling timer to drain event
    start_time = time.perf_counter()
    while window.center_panel.arc_reactor.state != "LISTENING" and (time.perf_counter() - start_time) < 2.0:
        qapp.processEvents()
        time.sleep(0.01)

    assert window.center_panel.arc_reactor.state == "LISTENING"
    assert window.center_panel.state_pills._active_state == "LISTENING"

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


def test_voice_interaction_boundary_delegation(qapp):
    """9. Verify UIBridge.start_voice_interaction() delegates to VoiceConversationEngine without freezing."""
    mock_voice_engine = mock.MagicMock()

    async def _mock_listen_once(duration=None):
        await asyncio.sleep(0.02)
        from app.voice.models import VoiceConversationResult
        return VoiceConversationResult(
            session_id="test_session",
            text="what time is it",
            command="what time is it",
            response_text="It is 10:00 PM",
            success=True,
        )

    mock_voice_engine.listen_once_async = mock.AsyncMock(side_effect=_mock_listen_once)

    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr, voice_engine=mock_voice_engine)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    completed_results = []
    bridge.command_completed.connect(completed_results.append)

    # Trigger microphone from header
    window.header.voice_toggled.emit()

    start_time = time.perf_counter()
    while not completed_results and (time.perf_counter() - start_time) < 2.0:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(completed_results) == 1
    assert completed_results[0].text == "what time is it"
    mock_voice_engine.listen_once_async.assert_called_once()

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


def test_gui_shutdown_does_not_hang(qapp):
    """10. Verify shutdown_gui cleanly terminates bridge, backend, and state manager without hanging."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)

    mock_backend = mock.MagicMock()
    mock_backend.stop = mock.MagicMock()

    # Time shutdown execution
    start = time.perf_counter()
    gui.shutdown_gui(bridge=bridge, backend=mock_backend, adapter=adapter)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    mock_backend.stop.assert_called_once()
    assert not bridge._timer.isActive()
