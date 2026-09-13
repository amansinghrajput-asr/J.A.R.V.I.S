"""Unit and Integration Tests for PresentationAdapter and PresentationQueue.

Tests Phase 23.1:
- Bounded event queue, draining, and drop-oldest policy
- Text command submission via CommandRouter.route_async
- Voice interaction initiation via VoiceConversationEngine.listen_once_async
- Confirmation resolution delegation to SystemConfirmationManager
- Cancellation delegation to ExecutionController
- ServiceContainer factory wiring
- Microphone RMS telemetry integration
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest.mock as mock
import pytest

from app.core.container import ServiceContainer
from app.core.presentation import (
    PresentationAdapter,
    PresentationQueue,
    create_presentation_adapter,
)
from app.core.state import (
    AssistantSnapshot,
    AssistantState,
    PendingConfirmation,
    PresentationEvent,
)
from app.core.state_manager import AssistantStateManager
from app.voice.microphone import MicrophoneRecorder


# ------------------------------------------------------------------------------
# PresentationQueue Tests
# ------------------------------------------------------------------------------

def test_presentation_queue_bounded_and_drop_oldest():
    """Verify queue enforces maxsize and drops oldest on overflow."""
    q = PresentationQueue(maxsize=5)
    snap = AssistantSnapshot(state=AssistantState.IDLE)

    for i in range(10):
        evt = PresentationEvent(event_type=f"evt_{i}", snapshot=snap)
        q.put(evt)

    assert q.qsize() == 5
    assert q.dropped_count == 5

    # Should retain the latest 5 events (evt_5 through evt_9)
    drained = q.drain(max_items=10)
    assert len(drained) == 5
    assert drained[0].event_type == "evt_5"
    assert drained[-1].event_type == "evt_9"
    assert q.qsize() == 0


def test_presentation_queue_drain_batching():
    """Verify drain respects max_items and does not block."""
    q = PresentationQueue(maxsize=100)
    snap = AssistantSnapshot(state=AssistantState.IDLE)

    for i in range(10):
        q.put(PresentationEvent(event_type=f"evt_{i}", snapshot=snap))

    first_batch = q.drain(max_items=4)
    assert len(first_batch) == 4
    assert q.qsize() == 6

    second_batch = q.drain(max_items=10)
    assert len(second_batch) == 6
    assert q.qsize() == 0


# ------------------------------------------------------------------------------
# PresentationAdapter Core & Dispatch Tests
# ------------------------------------------------------------------------------

def test_presentation_adapter_submit_command_success():
    """Verify submit_command calls route_async on CommandRouter."""
    async def _run():
        state_mgr = AssistantStateManager()
        mock_router = mock.AsyncMock()
        mock_router.route_async.return_value = "Command executed successfully"

        adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)

        res = await adapter.submit_command("check weather")
        assert res == "Command executed successfully"
        mock_router.route_async.assert_awaited_once_with("check weather", source="gui")

        adapter.close()
        state_mgr.close()

    asyncio.run(_run())


def test_presentation_adapter_submit_command_validation():
    """Verify empty or invalid text commands are rejected."""
    async def _run():
        state_mgr = AssistantStateManager()
        adapter = PresentationAdapter(state_manager=state_mgr)

        with pytest.raises(ValueError, match="non-empty string"):
            await adapter.submit_command("   ")

        adapter.close()
        state_mgr.close()

    asyncio.run(_run())


def test_presentation_adapter_voice_interaction():
    """Verify start_voice_interaction calls listen_once_async on VoiceConversationEngine."""
    async def _run():
        state_mgr = AssistantStateManager()
        mock_engine = mock.AsyncMock()
        mock_engine.listen_once_async.return_value = mock.MagicMock(success=True, text="hello")

        adapter = PresentationAdapter(state_manager=state_mgr, voice_engine=mock_engine)

        res = await adapter.start_voice_interaction(duration=4.0)
        assert res.success is True
        mock_engine.listen_once_async.assert_awaited_once_with(duration=4.0)

        adapter.close()
        state_mgr.close()

    asyncio.run(_run())


def test_presentation_adapter_confirmation_bridge():
    """Verify resolve_confirmation delegates to SystemConfirmationManager without bypass."""
    state_mgr = AssistantStateManager()
    mock_conf_mgr = mock.MagicMock()
    mock_conf_mgr.resolve_confirmation.return_value = True

    adapter = PresentationAdapter(state_manager=state_mgr, confirmation_manager=mock_conf_mgr)

    # Approve
    ok = adapter.resolve_confirmation("conf_100", approved=True, decided_by="user1")
    assert ok is True
    mock_conf_mgr.resolve_confirmation.assert_called_once_with(
        "conf_100", approved=True, decided_by="user1", reason=""
    )
    assert adapter.get_snapshot().state == AssistantState.EXECUTING

    # Reject
    mock_conf_mgr.resolve_confirmation.return_value = True
    ok2 = adapter.resolve_confirmation("conf_101", approved=False, reason="Too risky")
    assert ok2 is True
    assert adapter.get_snapshot().state == AssistantState.IDLE

    # Not found / expired
    mock_conf_mgr.resolve_confirmation.return_value = False
    ok3 = adapter.resolve_confirmation("conf_expired", approved=True)
    assert ok3 is False

    adapter.close()
    state_mgr.close()


def test_presentation_adapter_cancellation_bridge():
    """Verify cancel_current_task delegates to ExecutionController.cancel."""
    state_mgr = AssistantStateManager()
    mock_controller = mock.MagicMock()

    adapter = PresentationAdapter(state_manager=state_mgr, execution_controller=mock_controller)

    # Put state manager in EXECUTING state
    state_mgr.transition_to(AssistantState.EXECUTING)

    success = adapter.cancel_current_task("User clicked cancel")
    assert success is True
    mock_controller.cancel.assert_called_once_with(reason="User clicked cancel")
    assert adapter.get_snapshot().state == AssistantState.IDLE

    adapter.close()
    state_mgr.close()


def test_presentation_adapter_event_stream_draining():
    """Verify state transitions flow into presentation queue and can be drained."""
    state_mgr = AssistantStateManager()
    adapter = PresentationAdapter(state_manager=state_mgr)

    state_mgr.transition_to(AssistantState.LISTENING)
    state_mgr.transition_to(AssistantState.TRANSCRIBING)
    state_mgr.transition_to(AssistantState.IDLE)

    events = adapter.drain_events(max_items=10)
    assert len(events) >= 3
    states = [e.snapshot.state for e in events]
    assert AssistantState.LISTENING in states
    assert AssistantState.TRANSCRIBING in states
    assert AssistantState.IDLE in states

    adapter.close()
    state_mgr.close()


def test_create_presentation_adapter_di_factory():
    """Verify create_presentation_adapter wires services from ServiceContainer and reuses singleton."""
    sc = ServiceContainer()
    adapter = create_presentation_adapter(sc)

    assert isinstance(adapter, PresentationAdapter)
    assert sc.exists("state_manager")
    assert sc.exists("presentation_adapter")
    assert sc.resolve("presentation_adapter") is adapter

    # Second call should return the exact same instance without adding duplicate listeners
    state_mgr = sc.resolve("state_manager")
    listener_count_before = len(state_mgr._listeners)
    adapter2 = create_presentation_adapter(sc)
    assert adapter2 is adapter
    assert len(state_mgr._listeners) == listener_count_before

    adapter.close()


# ------------------------------------------------------------------------------
# Microphone RMS Telemetry Tests
# ------------------------------------------------------------------------------

def test_microphone_rms_telemetry_callback():
    """Verify MicrophoneRecorder invokes optional amplitude_callback with normalized value."""
    received_amps: list[float] = []

    def on_amplitude(val: float):
        received_amps.append(val)

    # 16-bit PCM silent dummy audio callback
    def dummy_audio_source(num_frames: int) -> bytes:
        # Generates half-amplitude sine/constant wave for predictable non-zero RMS
        return b"\x00\x20" * num_frames

    recorder = MicrophoneRecorder(
        sample_rate=16000,
        record_duration=0.1,
        auto_register_in_container=False,
        audio_source_callback=dummy_audio_source,
        amplitude_callback=on_amplitude,
    )

    out_file = recorder.record(duration=0.1)
    assert out_file.exists()
    assert len(received_amps) > 0
    # Values should be between 0.0 and 1.0
    for amp in received_amps:
        assert 0.0 <= amp <= 1.0

    recorder.close()
    if out_file.exists():
        out_file.unlink()


def test_microphone_rms_telemetry_callback_failure_resilience():
    """Verify errors inside amplitude_callback do not disrupt recording."""
    def buggy_callback(val: float):
        raise RuntimeError("Buggy GUI amplitude handler")

    def dummy_audio_source(num_frames: int) -> bytes:
        return b"\x00\x00" * num_frames

    recorder = MicrophoneRecorder(
        sample_rate=16000,
        record_duration=0.1,
        auto_register_in_container=False,
        audio_source_callback=dummy_audio_source,
        amplitude_callback=buggy_callback,
    )

    # Should not raise exception
    out_file = recorder.record(duration=0.1)
    assert out_file.exists()

    recorder.close()
    if out_file.exists():
        out_file.unlink()
