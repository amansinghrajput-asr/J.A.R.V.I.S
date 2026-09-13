"""Unit and Property Tests for AssistantState and Immutable Snapshot Data Models.

Tests Phase 23.1 data models:
- AssistantState enumeration completeness
- PendingConfirmation validation and immutability
- AssistantSnapshot validation, bounds clamping, and deep immutability
- PresentationEvent payload immutability and dict conversion
"""

from __future__ import annotations

import dataclasses
import pytest

from app.core.state import (
    AssistantSnapshot,
    AssistantState,
    PendingConfirmation,
    PresentationEvent,
)


def test_assistant_state_enumeration_values():
    """Verify all 10 required operational states exist and have str values."""
    expected = {
        "INITIALIZING",
        "IDLE",
        "LISTENING",
        "TRANSCRIBING",
        "THINKING",
        "PLANNING",
        "EXECUTING",
        "SPEAKING",
        "AWAITING_CONFIRMATION",
        "ERROR",
    }
    actual = {s.value for s in AssistantState}
    assert actual == expected
    assert len(AssistantState) == 10
    assert AssistantState.IDLE == "IDLE"
    assert isinstance(AssistantState.IDLE.value, str)


def test_pending_confirmation_creation_and_immutability():
    """Verify PendingConfirmation validation and frozen immutability."""
    conf = PendingConfirmation(
        confirmation_id="conf_123",
        operation="reboot",
        target="system",
        risk_level="HIGH",
        description="Reboot the system",
    )
    assert conf.confirmation_id == "conf_123"
    assert conf.operation == "reboot"
    assert conf.target == "system"
    assert conf.risk_level == "HIGH"
    assert conf.description == "Reboot the system"
    assert conf.created_at > 0

    # Frozen dataclass mutation attempts must fail
    with pytest.raises(dataclasses.FrozenInstanceError):
        conf.operation = "shutdown"

    with pytest.raises(dataclasses.FrozenInstanceError):
        conf.confirmation_id = "other"


def test_pending_confirmation_validation():
    """Verify empty confirmation_id or operation is rejected."""
    with pytest.raises(ValueError, match="confirmation_id must be a non-empty string"):
        PendingConfirmation(confirmation_id="", operation="shutdown")

    with pytest.raises(ValueError, match="operation must be a non-empty string"):
        PendingConfirmation(confirmation_id="id_1", operation="   ")


def test_assistant_snapshot_creation_and_deep_immutability():
    """Verify AssistantSnapshot enforces deep immutability and clamps bounds."""
    conf = PendingConfirmation(confirmation_id="c_1", operation="format_drive")
    snap = AssistantSnapshot(
        state=AssistantState.EXECUTING,
        status_message="Executing disk scan",
        current_command="scan drive C:",
        last_response="Scan finished",
        execution_id="exec_99",
        task_id="task_1",
        plan_id="plan_1",
        task_progress=0.75,
        pending_confirmation=conf,
        transcript=["User: scan drive C:", "Assistant: Scanning..."],  # Passed as list
        mic_amplitude=0.45,
    )

    # Immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.state = AssistantState.IDLE

    # Deep immutability: transcript was converted to tuple
    assert isinstance(snap.transcript, tuple)
    assert len(snap.transcript) == 2
    assert snap.transcript[0] == "User: scan drive C:"

    # Progress and amplitude clamping
    snap_clamped = AssistantSnapshot(
        state=AssistantState.IDLE,
        task_progress=1.5,  # Exceeds 1.0
        mic_amplitude=-0.2,  # Below 0.0
    )
    assert snap_clamped.task_progress == 1.0
    assert snap_clamped.mic_amplitude == 0.0

    snap_clamped2 = AssistantSnapshot(
        state=AssistantState.IDLE,
        task_progress=-0.5,
        mic_amplitude=2.0,
    )
    assert snap_clamped2.task_progress == 0.0
    assert snap_clamped2.mic_amplitude == 1.0


def test_assistant_snapshot_type_safety():
    """Verify invalid state or pending_confirmation types raise TypeError."""
    with pytest.raises(TypeError, match="state must be an AssistantState instance"):
        AssistantSnapshot(state="INVALID_STRING_STATE")

    with pytest.raises(TypeError, match="pending_confirmation must be a PendingConfirmation"):
        AssistantSnapshot(state=AssistantState.IDLE, pending_confirmation={"id": "bad"})


def test_presentation_event_structure():
    """Verify PresentationEvent payload immutability and helper."""
    snap = AssistantSnapshot(state=AssistantState.IDLE)
    evt = PresentationEvent(
        event_type="test.event",
        snapshot=snap,
        payload={"key1": "val1", "key2": 42},
    )

    assert evt.event_type == "test.event"
    assert evt.snapshot.state == AssistantState.IDLE
    assert isinstance(evt.payload, tuple)
    assert evt.get_payload_dict() == {"key1": "val1", "key2": 42}

    # Frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        evt.event_type = "modified"
