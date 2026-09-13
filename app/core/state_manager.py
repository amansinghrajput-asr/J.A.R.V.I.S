"""Assistant State Manager for J.A.R.V.I.S. Phase 23.1.

Aggregates events from EventBus and PlannerEventBus, manages state transitions,
tracks execution and plan progress, preserves bounded snapshot history, and
notifies registered listeners with exception isolation. Headless and Qt-free.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence

from app.core.event_bus import Event, EventBus
from app.core.logger import get_logger
from app.core.state import (
    AssistantSnapshot,
    AssistantState,
    PendingConfirmation,
    PresentationEvent,
)

logger = get_logger("STATE_MANAGER")

# Maximum snapshots to keep in bounded history
DEFAULT_SNAPSHOT_HISTORY_SIZE = 100

# Allowed transitions map: SourceState -> Set of valid TargetStates.
# Note: ERROR can transition to IDLE or INITIALIZING for safe recovery.
# Any state can transition to ERROR.
ALLOWED_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.INITIALIZING: {AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.IDLE: {
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.PLANNING,
        AssistantState.EXECUTING,
        AssistantState.AWAITING_CONFIRMATION,
        AssistantState.ERROR,
        AssistantState.IDLE,
    },
    AssistantState.LISTENING: {
        AssistantState.TRANSCRIBING,
        AssistantState.IDLE,
        AssistantState.ERROR,
        AssistantState.LISTENING,
    },
    AssistantState.TRANSCRIBING: {
        AssistantState.THINKING,
        AssistantState.PLANNING,
        AssistantState.EXECUTING,
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.THINKING: {
        AssistantState.PLANNING,
        AssistantState.EXECUTING,
        AssistantState.SPEAKING,
        AssistantState.AWAITING_CONFIRMATION,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.PLANNING: {
        AssistantState.EXECUTING,
        AssistantState.AWAITING_CONFIRMATION,
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.EXECUTING: {
        AssistantState.EXECUTING,
        AssistantState.AWAITING_CONFIRMATION,
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,
        AssistantState.LISTENING,
        AssistantState.ERROR,
        AssistantState.SPEAKING,
    },
    AssistantState.AWAITING_CONFIRMATION: {
        AssistantState.EXECUTING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.ERROR: {
        AssistantState.IDLE,
        AssistantState.INITIALIZING,
        AssistantState.LISTENING,
    },
}


class AssistantStateManager:
    """Thread-safe state manager unifying assistant lifecycle and presentation events.

    Listens to EventBus and PlannerEventBus to maintain an authoritative, deeply
    immutable AssistantSnapshot. Never throws exceptions into backend publishers.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        planner_event_bus: Optional[Any] = None,
        *,
        history_size: int = DEFAULT_SNAPSHOT_HISTORY_SIZE,
        initial_state: AssistantState = AssistantState.IDLE,
    ) -> None:
        """Initialize AssistantStateManager.

        Args:
            event_bus: Optional system EventBus.
            planner_event_bus: Optional PlannerEventBus.
            history_size: Maximum number of recent snapshots retained.
            initial_state: Starting assistant state (defaults to IDLE).
        """
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._planner_event_bus = planner_event_bus
        self._history_size = max(10, history_size)

        # Internal mutable state tracking (protected by _lock)
        self._state = initial_state
        self._status_message = "Ready" if initial_state == AssistantState.IDLE else "Initializing"
        self._current_command: Optional[str] = None
        self._last_response: Optional[str] = None
        self._last_error: Optional[str] = None
        self._execution_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._plan_id: Optional[str] = None
        self._total_tasks: int = 0
        self._completed_tasks: int = 0
        self._last_completed_progress: float = 0.0
        self._pending_confirmation: Optional[PendingConfirmation] = None
        self._transcript: list[str] = []
        self._mic_amplitude: float = 0.0

        # Bounded snapshot history
        self._history: deque[AssistantSnapshot] = deque(maxlen=self._history_size)

        # Listeners: list of callables receiving (PresentationEvent)
        self._listeners: list[Callable[[PresentationEvent], Any]] = []

        # Subscriptions unsubscribe handles
        self._unsubscribers: list[Callable[[], Any]] = []

        # Store initial snapshot
        initial_snap = self._build_snapshot()
        self._history.append(initial_snap)

        # Setup event bus listeners if provided
        self._setup_event_subscriptions()

    # --------------------------------------------------------------------------
    # Snapshot & Listener API
    # --------------------------------------------------------------------------

    def get_snapshot(self) -> AssistantSnapshot:
        """Return the current deeply immutable AssistantSnapshot (thread-safe)."""
        with self._lock:
            return self._build_snapshot()

    def get_history(self) -> list[AssistantSnapshot]:
        """Return a copy of the bounded snapshot history."""
        with self._lock:
            return list(self._history)

    def add_listener(self, listener: Callable[[PresentationEvent], Any]) -> Callable[[], bool]:
        """Register a state/presentation event listener.

        Args:
            listener: Callable accepting a single PresentationEvent argument.

        Returns:
            Unsubscribe function returning True if removed.
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

        def unsubscribe() -> bool:
            return self.remove_listener(listener)

        return unsubscribe

    def remove_listener(self, listener: Callable[[PresentationEvent], Any]) -> bool:
        """Remove a registered listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                return True
            return False

    # --------------------------------------------------------------------------
    # Telemetry Updates
    # --------------------------------------------------------------------------

    def update_mic_amplitude(self, amplitude: float) -> None:
        """Update microphone amplitude telemetry safely without changing state machine state."""
        clamped = max(0.0, min(1.0, float(amplitude)))
        snap = None
        with self._lock:
            self._mic_amplitude = clamped
            snap = self._build_snapshot()

        event = PresentationEvent(
            event_type="telemetry.mic_amplitude",
            snapshot=snap,
            payload=(("amplitude", clamped),),
        )
        self._notify_listeners(event)

    # --------------------------------------------------------------------------
    # State Transition & Mutation Core
    # --------------------------------------------------------------------------

    def transition_to(
        self,
        new_state: AssistantState,
        *,
        status_message: Optional[str] = None,
        last_error: Optional[str] = None,
        event_type: Optional[str] = None,
        payload_data: Optional[dict[str, Any]] = None,
    ) -> AssistantSnapshot:
        """Attempt transition to a new state with deterministic validation.

        If the transition is invalid according to ALLOWED_TRANSITIONS, the transition
        is rejected safely (logged as warning, existing state preserved) and no exception
        is propagated to the caller.
        """
        snap = None
        with self._lock:
            if not self._is_valid_transition(self._state, new_state):
                logger.warning(
                    "Invalid state transition rejected: %s -> %s (current: %s)",
                    self._state.value,
                    new_state.value,
                    self._status_message,
                )
                return self._build_snapshot()

            self._state = new_state
            if status_message is not None:
                self._status_message = status_message
            if last_error is not None:
                self._last_error = last_error
            elif new_state != AssistantState.ERROR:
                # Clear error on successful transition away from ERROR
                self._last_error = None

            # Confirmation lifecycle cleanup
            if new_state != AssistantState.AWAITING_CONFIRMATION:
                self._pending_confirmation = None

            # Reset task tracking if returning to IDLE
            if new_state == AssistantState.IDLE:
                self._task_id = None
                if self._total_tasks > 0 and self._completed_tasks >= self._total_tasks:
                    self._last_completed_progress = 1.0
                else:
                    self._last_completed_progress = 0.0
                self._total_tasks = 0
                self._completed_tasks = 0
                self._mic_amplitude = 0.0

            snap = self._build_snapshot()
            self._history.append(snap)

        pres_event = PresentationEvent(
            event_type=event_type or f"state.{new_state.value.lower()}",
            snapshot=snap,
            payload=tuple((payload_data or {}).items()),
        )
        self._notify_listeners(pres_event)
        return snap

    def recover_to_idle(self, status_message: str = "Recovered to idle") -> AssistantSnapshot:
        """Safely force recovery from ERROR or any anomalous state back to IDLE."""
        return self.transition_to(
            AssistantState.IDLE,
            status_message=status_message,
            event_type="state.recovered",
        )

    def _is_valid_transition(self, current: AssistantState, target: AssistantState) -> bool:
        """Check if transition from current to target is permissible."""
        if current == target:
            return True
        allowed = ALLOWED_TRANSITIONS.get(current, set())
        return target in allowed

    def _build_snapshot(self) -> AssistantSnapshot:
        """Internal helper to construct an immutable snapshot from current internal state."""
        progress = self._last_completed_progress
        if self._total_tasks > 0:
            progress = max(0.0, min(1.0, float(self._completed_tasks) / float(self._total_tasks)))

        return AssistantSnapshot(
            state=self._state,
            status_message=self._status_message,
            current_command=self._current_command,
            last_response=self._last_response,
            last_error=self._last_error,
            execution_id=self._execution_id,
            task_id=self._task_id,
            plan_id=self._plan_id,
            task_progress=progress,
            pending_confirmation=self._pending_confirmation,
            transcript=tuple(self._transcript),
            mic_amplitude=self._mic_amplitude,
            timestamp=time.time(),
        )

    def _notify_listeners(self, event: PresentationEvent) -> None:
        """Dispatch presentation event to all listeners with exception isolation."""
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.warning("Listener %s raised unhandled exception: %s", listener, exc)

    # --------------------------------------------------------------------------
    # EventBus & PlannerEventBus Subscriptions
    # --------------------------------------------------------------------------

    def _setup_event_subscriptions(self) -> None:
        """Attach listeners to existing EventBus and PlannerEventBus."""
        if self._event_bus is not None:
            self._subscribe_event_bus()

        if self._planner_event_bus is not None:
            self._subscribe_planner_event_bus()

    def _subscribe_event_bus(self) -> None:
        """Subscribe to system EventBus lifecycle topics."""
        bus = self._event_bus
        if bus is None:
            return

        handlers: list[tuple[str, Callable[[Event], None]]] = [
            ("voice_engine.recording", self._handle_voice_recording),
            ("voice_engine.transcribing", self._handle_voice_transcribing),
            ("voice_engine.routing", self._handle_voice_routing),
            ("voice_engine.speaking", self._handle_voice_speaking),
            ("voice_engine.completed", self._handle_voice_completed),
            ("voice_engine.failed", self._handle_voice_failed),
            ("command.received", self._handle_command_received),
            ("command.completed", self._handle_command_completed),
            ("command.failed", self._handle_command_failed),
            ("application.ready", self._handle_app_ready),
        ]

        for topic, handler in handlers:
            try:
                unsub = bus.subscribe(topic, handler)
                if callable(unsub):
                    self._unsubscribers.append(unsub)
            except Exception as exc:
                logger.warning("Failed to subscribe to EventBus topic '%s': %s", topic, exc)

    def _subscribe_planner_event_bus(self) -> None:
        """Subscribe to PlannerEventBus typed events."""
        p_bus = self._planner_event_bus
        if p_bus is None:
            return

        try:
            from app.ai.planner.events import (
                PlanCancelled,
                PlanCompleted,
                PlanFailed,
                PlanStarted,
                SystemSkillCompleted,
                SystemSkillConfirmationRequired,
                SystemSkillFailed,
                SystemSkillStarted,
                TaskCompleted,
                TaskFailed,
                TaskStarted,
            )

            planner_handlers = [
                (PlanStarted, self._handle_plan_started),
                (PlanCompleted, self._handle_plan_completed),
                (PlanFailed, self._handle_plan_failed),
                (PlanCancelled, self._handle_plan_cancelled),
                (TaskStarted, self._handle_task_started),
                (TaskCompleted, self._handle_task_completed),
                (TaskFailed, self._handle_task_failed),
                (SystemSkillStarted, self._handle_system_skill_started),
                (SystemSkillCompleted, self._handle_system_skill_completed),
                (SystemSkillFailed, self._handle_system_skill_failed),
                (SystemSkillConfirmationRequired, self._handle_confirmation_required),
            ]

            for event_cls, handler in planner_handlers:
                try:
                    unsub = p_bus.subscribe(event_cls, handler)
                    if callable(unsub):
                        self._unsubscribers.append(unsub)
                except Exception as exc:
                    logger.warning("Failed to subscribe to PlannerEvent '%s': %s", event_cls, exc)
        except ImportError as exc:
            logger.warning("Planner events could not be imported: %s", exc)

    def close(self) -> None:
        """Unsubscribe all event listeners and release resources."""
        with self._lock:
            for unsub in self._unsubscribers:
                try:
                    unsub()
                except Exception:
                    pass
            self._unsubscribers.clear()
            self._listeners.clear()

    # --------------------------------------------------------------------------
    # EventBus Handlers
    # --------------------------------------------------------------------------

    def _handle_app_ready(self, event: Event) -> None:
        """Handle application.ready."""
        self.transition_to(AssistantState.IDLE, status_message="System ready")

    def _handle_voice_recording(self, event: Event) -> None:
        """Handle voice_engine.recording."""
        payload = event.payload or {}
        duration = payload.get("duration", 0.0)
        self.transition_to(
            AssistantState.LISTENING,
            status_message=f"Listening... ({duration:.1f}s)" if duration else "Listening...",
            payload_data=payload if isinstance(payload, dict) else {},
        )

    def _handle_voice_transcribing(self, event: Event) -> None:
        """Handle voice_engine.transcribing."""
        self.transition_to(
            AssistantState.TRANSCRIBING,
            status_message="Transcribing speech...",
        )

    def _handle_voice_routing(self, event: Event) -> None:
        """Handle voice_engine.routing."""
        payload = event.payload or {}
        command = payload.get("text") or payload.get("command") or ""
        with self._lock:
            if command:
                self._current_command = str(command)
                self._transcript.append(f"User: {command}")
        self.transition_to(
            AssistantState.THINKING,
            status_message=f"Processing command: {command}" if command else "Processing command...",
        )

    def _handle_voice_speaking(self, event: Event) -> None:
        """Handle voice_engine.speaking."""
        payload = event.payload or {}
        resp_text = payload.get("response_text", "")
        with self._lock:
            if resp_text:
                self._last_response = resp_text
                self._transcript.append(f"J.A.R.V.I.S: {resp_text}")
        self.transition_to(
            AssistantState.SPEAKING,
            status_message="Speaking response...",
        )

    def _handle_voice_completed(self, event: Event) -> None:
        """Handle voice_engine.completed."""
        self.transition_to(AssistantState.IDLE, status_message="Ready")

    def _handle_voice_failed(self, event: Event) -> None:
        """Handle voice_engine.failed."""
        payload = event.payload or {}
        err = payload.get("error", "Voice conversation error")
        self.transition_to(
            AssistantState.ERROR,
            status_message="Voice operation failed",
            last_error=str(err),
            payload_data=payload if isinstance(payload, dict) else {},
        )

    def _handle_command_received(self, event: Event) -> None:
        """Handle command.received."""
        payload = event.payload or {}
        cmd_text = payload.get("raw_command") or payload.get("command") or ""
        with self._lock:
            if cmd_text:
                self._current_command = str(cmd_text)
                self._transcript.append(f"Command: {cmd_text}")
        self.transition_to(
            AssistantState.THINKING,
            status_message=f"Evaluating: {cmd_text}" if cmd_text else "Evaluating command...",
        )

    def _handle_command_completed(self, event: Event) -> None:
        """Handle command.completed."""
        payload = event.payload or {}
        resp = payload.get("result") or payload.get("response") or "Completed"
        with self._lock:
            self._last_response = str(resp)
        self.transition_to(AssistantState.IDLE, status_message="Ready")

    def _handle_command_failed(self, event: Event) -> None:
        """Handle command.failed."""
        payload = event.payload or {}
        err = payload.get("error", "Command failed")
        self.transition_to(
            AssistantState.ERROR,
            status_message="Command failed",
            last_error=str(err),
        )

    # --------------------------------------------------------------------------
    # PlannerEventBus Handlers
    # --------------------------------------------------------------------------

    def _handle_plan_started(self, event: Any) -> None:
        """Handle PlanStarted."""
        with self._lock:
            self._plan_id = getattr(event, "plan_id", None)
            self._execution_id = getattr(event, "execution_id", None)
            self._total_tasks = getattr(event, "task_count", 0)
            self._completed_tasks = 0
        self.transition_to(
            AssistantState.PLANNING,
            status_message=f"Planning: {self._total_tasks} tasks scheduled",
            payload_data={"plan_id": self._plan_id, "task_count": self._total_tasks},
        )

    def _handle_plan_completed(self, event: Any) -> None:
        """Handle PlanCompleted."""
        with self._lock:
            self._completed_tasks = self._total_tasks
        self.transition_to(
            AssistantState.IDLE,
            status_message="Plan execution completed",
            payload_data={"plan_id": getattr(event, "plan_id", "")},
        )

    def _handle_plan_failed(self, event: Any) -> None:
        """Handle PlanFailed."""
        err = getattr(event, "error", "Plan execution failed")
        self.transition_to(
            AssistantState.ERROR,
            status_message="Plan failed",
            last_error=str(err),
            payload_data={"plan_id": getattr(event, "plan_id", "")},
        )

    def _handle_plan_cancelled(self, event: Any) -> None:
        """Handle PlanCancelled."""
        reason = getattr(event, "reason", "Plan cancelled")
        self.transition_to(
            AssistantState.IDLE,
            status_message=f"Plan cancelled: {reason}",
            payload_data={"plan_id": getattr(event, "plan_id", "")},
        )

    def _handle_task_started(self, event: Any) -> None:
        """Handle TaskStarted."""
        with self._lock:
            self._task_id = getattr(event, "task_id", None)
            action = getattr(event, "action", "")
            target = getattr(event, "target", "")
        self.transition_to(
            AssistantState.EXECUTING,
            status_message=f"Executing: {action} {target or ''}".strip(),
            payload_data={"task_id": self._task_id, "action": action},
        )

    def _handle_task_completed(self, event: Any) -> None:
        """Handle TaskCompleted."""
        with self._lock:
            self._completed_tasks += 1
            snap = self._build_snapshot()
            self._history.append(snap)

        pres_event = PresentationEvent(
            event_type="planner.task_completed",
            snapshot=snap,
            payload=(
                ("task_id", getattr(event, "task_id", "")),
                ("progress", snap.task_progress),
            ),
        )
        self._notify_listeners(pres_event)

    def _handle_task_failed(self, event: Any) -> None:
        """Handle TaskFailed."""
        err = getattr(event, "error", "Task execution failed")
        logger.warning("Planner task failed: %s", err)
        # Note: Task failure doesn't automatically move entire plan to ERROR
        # because planner has replanning/recovery capabilities.

    def _handle_system_skill_started(self, event: Any) -> None:
        """Handle SystemSkillStarted."""
        op = getattr(event, "operation", "")
        target = getattr(event, "target", "")
        self.transition_to(
            AssistantState.EXECUTING,
            status_message=f"System operation: {op} {target or ''}".strip(),
            payload_data={"operation": op, "target": target},
        )

    def _handle_system_skill_completed(self, event: Any) -> None:
        """Handle SystemSkillCompleted."""
        # Remain in EXECUTING or transition to IDLE if no plan active
        with self._lock:
            if not self._plan_id:
                self.transition_to(AssistantState.IDLE, status_message="Ready")

    def _handle_system_skill_failed(self, event: Any) -> None:
        """Handle SystemSkillFailed."""
        err = getattr(event, "error", "System skill failed")
        self.transition_to(
            AssistantState.ERROR,
            status_message="System skill failed",
            last_error=str(err),
            payload_data={"operation": getattr(event, "operation", "")},
        )

    def _handle_confirmation_required(self, event: Any) -> None:
        """Handle SystemSkillConfirmationRequired."""
        conf_id = getattr(event, "confirmation_id", "")
        op = getattr(event, "operation", "")
        target = getattr(event, "target", None)
        risk = getattr(event, "risk_level", "HIGH")

        conf = PendingConfirmation(
            confirmation_id=conf_id,
            operation=op,
            target=target,
            risk_level=risk,
            description=f"Action '{op}' requires confirmation",
        )
        with self._lock:
            self._pending_confirmation = conf

        self.transition_to(
            AssistantState.AWAITING_CONFIRMATION,
            status_message=f"Confirmation required: {op}",
            payload_data={"confirmation_id": conf_id, "operation": op, "risk_level": risk},
        )
