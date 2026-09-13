"""Unit and Integration Tests for AssistantStateManager.

Tests Phase 23.1:
- State transitions and validation safeguards
- EventBus lifecycle mapping
- PlannerEventBus lifecycle mapping
- Confirmation lifecycle handling
- Task progress calculation
- ERROR recovery
- Thread-safety, listener isolation, and bounded history
"""

from __future__ import annotations

import threading
import time
import pytest

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlanStarted,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillFailed,
    SystemSkillStarted,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
)
from app.core.event_bus import Event, EventBus
from app.core.state import AssistantState, PresentationEvent
from app.core.state_manager import AssistantStateManager


def test_initial_state_and_snapshot():
    """Verify default initial state is IDLE and snapshot is returned."""
    manager = AssistantStateManager(initial_state=AssistantState.IDLE)
    snap = manager.get_snapshot()
    assert snap.state == AssistantState.IDLE
    assert snap.status_message == "Ready"
    assert len(manager.get_history()) == 1


def test_valid_state_transitions():
    """Verify standard happy-path transitions succeed."""
    manager = AssistantStateManager(initial_state=AssistantState.IDLE)

    # IDLE -> LISTENING
    s1 = manager.transition_to(AssistantState.LISTENING, status_message="Listening...")
    assert s1.state == AssistantState.LISTENING

    # LISTENING -> TRANSCRIBING
    s2 = manager.transition_to(AssistantState.TRANSCRIBING, status_message="Transcribing...")
    assert s2.state == AssistantState.TRANSCRIBING

    # TRANSCRIBING -> THINKING
    s3 = manager.transition_to(AssistantState.THINKING, status_message="Thinking...")
    assert s3.state == AssistantState.THINKING

    # THINKING -> PLANNING
    s4 = manager.transition_to(AssistantState.PLANNING, status_message="Planning...")
    assert s4.state == AssistantState.PLANNING

    # PLANNING -> EXECUTING
    s5 = manager.transition_to(AssistantState.EXECUTING, status_message="Executing...")
    assert s5.state == AssistantState.EXECUTING

    # EXECUTING -> SPEAKING
    s6 = manager.transition_to(AssistantState.SPEAKING, status_message="Speaking...")
    assert s6.state == AssistantState.SPEAKING

    # SPEAKING -> IDLE
    s7 = manager.transition_to(AssistantState.IDLE, status_message="Ready")
    assert s7.state == AssistantState.IDLE


def test_invalid_state_transition_is_rejected_safely():
    """Verify invalid transitions are rejected without crashing."""
    manager = AssistantStateManager(initial_state=AssistantState.INITIALIZING)

    # INITIALIZING -> EXECUTING is invalid
    snap = manager.transition_to(AssistantState.EXECUTING)
    # State should remain INITIALIZING
    assert snap.state == AssistantState.INITIALIZING
    assert manager.get_snapshot().state == AssistantState.INITIALIZING


def test_error_state_and_recovery():
    """Verify transition to ERROR and subsequent recovery to IDLE."""
    manager = AssistantStateManager(initial_state=AssistantState.EXECUTING)

    # EXECUTING -> ERROR
    snap_err = manager.transition_to(
        AssistantState.ERROR,
        status_message="Process failed",
        last_error="Crash in subsystem",
    )
    assert snap_err.state == AssistantState.ERROR
    assert snap_err.last_error == "Crash in subsystem"

    # Recover to IDLE
    snap_recovered = manager.recover_to_idle("Recovered cleanly")
    assert snap_recovered.state == AssistantState.IDLE
    assert snap_recovered.last_error is None
    assert snap_recovered.status_message == "Recovered cleanly"


def test_event_bus_mapping():
    """Verify system EventBus events transition state manager accurately."""
    eb = EventBus()
    manager = AssistantStateManager(event_bus=eb, initial_state=AssistantState.IDLE)

    # voice_engine.recording
    eb.publish("voice_engine.recording", {"duration": 5.0})
    assert manager.get_snapshot().state == AssistantState.LISTENING

    # voice_engine.transcribing
    eb.publish("voice_engine.transcribing", {})
    assert manager.get_snapshot().state == AssistantState.TRANSCRIBING

    # voice_engine.routing (using standard 'command' payload key)
    eb.publish("voice_engine.routing", {"command": "open browser"})
    snap = manager.get_snapshot()
    assert snap.state == AssistantState.THINKING
    assert snap.current_command == "open browser"
    assert any("open browser" in line for line in snap.transcript)

    # voice_engine.routing (also supports fallback 'text' key)
    eb.publish("voice_engine.routing", {"text": "what is the time?"})
    snap2 = manager.get_snapshot()
    assert snap2.state == AssistantState.THINKING
    assert snap2.current_command == "what is the time?"

    # voice_engine.speaking
    eb.publish("voice_engine.speaking", {"response_text": "The time is 1:40 PM."})
    snap_spk = manager.get_snapshot()
    assert snap_spk.state == AssistantState.SPEAKING
    assert snap_spk.last_response == "The time is 1:40 PM."

    # voice_engine.completed
    eb.publish("voice_engine.completed", {})
    assert manager.get_snapshot().state == AssistantState.IDLE

    # voice_engine.failed
    eb.publish("voice_engine.failed", {"error": "Microphone disconnected"})
    assert manager.get_snapshot().state == AssistantState.ERROR
    assert manager.get_snapshot().last_error == "Microphone disconnected"

    manager.close()


def test_planner_event_bus_mapping_and_progress():
    """Verify PlannerEventBus lifecycle events and task progress calculation."""
    peb = PlannerEventBus()
    manager = AssistantStateManager(planner_event_bus=peb, initial_state=AssistantState.IDLE)

    # PlanStarted (total 4 tasks)
    peb.publish(PlanStarted(plan_id="p_001", execution_id="ex_001", query="test query", task_count=4))
    snap_plan = manager.get_snapshot()
    assert snap_plan.state == AssistantState.PLANNING
    assert snap_plan.plan_id == "p_001"
    assert snap_plan.execution_id == "ex_001"
    assert snap_plan.task_progress == 0.0

    # Task 1 Started
    peb.publish(TaskStarted(plan_id="p_001", task_id="t_1", action="open_app", target="notepad"))
    assert manager.get_snapshot().state == AssistantState.EXECUTING
    assert manager.get_snapshot().task_id == "t_1"

    # Task 1 Completed
    peb.publish(TaskCompleted(plan_id="p_001", task_id="t_1", action="open_app", result="OK"))
    assert manager.get_snapshot().task_progress == 0.25

    # Task 2 Completed
    peb.publish(TaskCompleted(plan_id="p_001", task_id="t_2", action="type_text", result="OK"))
    assert manager.get_snapshot().task_progress == 0.50

    # SystemSkillConfirmationRequired
    peb.publish(
        SystemSkillConfirmationRequired(
            skill_name="system",
            operation="reboot",
            target="pc",
            confirmation_id="conf_999",
            risk_level="HIGH",
        )
    )
    snap_conf = manager.get_snapshot()
    assert snap_conf.state == AssistantState.AWAITING_CONFIRMATION
    assert snap_conf.pending_confirmation is not None
    assert snap_conf.pending_confirmation.confirmation_id == "conf_999"
    assert snap_conf.pending_confirmation.operation == "reboot"

    # PlanCompleted
    peb.publish(PlanCompleted(plan_id="p_001", success=True, completed_count=4))
    snap_done = manager.get_snapshot()
    assert snap_done.state == AssistantState.IDLE
    assert snap_done.pending_confirmation is None
    assert snap_done.task_progress == 1.0

    manager.close()


def test_listener_registration_and_exception_isolation():
    """Verify listener receives events and listener exceptions do not disrupt manager."""
    manager = AssistantStateManager(initial_state=AssistantState.IDLE)
    received: list[PresentationEvent] = []

    def healthy_listener(event: PresentationEvent):
        received.append(event)

    def failing_listener(event: PresentationEvent):
        raise RuntimeError("Buggy listener crashed")

    unsub1 = manager.add_listener(failing_listener)
    unsub2 = manager.add_listener(healthy_listener)

    # Transition
    snap = manager.transition_to(AssistantState.LISTENING, status_message="Listening...")
    assert snap.state == AssistantState.LISTENING

    # Healthy listener should still have received the event despite failing_listener
    assert len(received) == 1
    assert received[0].snapshot.state == AssistantState.LISTENING

    # Unsubscribe
    unsub2()
    manager.transition_to(AssistantState.IDLE)
    assert len(received) == 1  # No more events appended

    manager.close()


def test_telemetry_mic_amplitude():
    """Verify update_mic_amplitude publishes telemetry without altering operational state."""
    manager = AssistantStateManager(initial_state=AssistantState.IDLE)
    events: list[PresentationEvent] = []
    manager.add_listener(lambda e: events.append(e))

    manager.update_mic_amplitude(0.85)
    snap = manager.get_snapshot()
    assert snap.state == AssistantState.IDLE
    assert snap.mic_amplitude == 0.85
    assert len(events) == 1
    assert events[0].event_type == "telemetry.mic_amplitude"

    manager.close()


def test_bounded_snapshot_history():
    """Verify snapshot history remains bounded to max history_size."""
    manager = AssistantStateManager(history_size=10, initial_state=AssistantState.IDLE)
    for _ in range(25):
        manager.transition_to(AssistantState.LISTENING)
        manager.transition_to(AssistantState.IDLE)

    history = manager.get_history()
    assert len(history) <= 10
    manager.close()


def test_concurrent_state_transitions():
    """Verify concurrent transitions from multiple threads do not corrupt internal state."""
    manager = AssistantStateManager(initial_state=AssistantState.IDLE)
    errors = []

    def worker(worker_id: int):
        try:
            for _ in range(50):
                manager.transition_to(AssistantState.THINKING)
                manager.update_mic_amplitude(0.5)
                manager.transition_to(AssistantState.IDLE)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert manager.get_snapshot().state in (AssistantState.IDLE, AssistantState.THINKING)
    manager.close()
