"""Presentation Bridge for PySide6 Desktop HUD in J.A.R.V.I.S. Phase 23.2.

Provides the decoupled QObject boundary connecting PresentationAdapter events
and snapshots to Qt signals on the GUI main thread. Never blocks the Qt event
loop and never accesses raw AI providers or planner internals directly.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState, PresentationEvent

logger = logging.getLogger("app.ui.bridge")

DEFAULT_POLL_INTERVAL_MS = 33  # ~30 Hz polling rate for smooth telemetry and state updates


class UIBridge(QObject):
    """Qt-native Presentation Bridge delivering AssistantSnapshots and telemetry to GUI.

    Signals:
        snapshot_updated(object): Emits latest AssistantSnapshot on change.
        state_changed(str): Emits new AssistantState value string on transition.
        amplitude_updated(float): Emits normalized microphone amplitude (0.0 - 1.0).
        event_dispatched(object): Emits each discrete PresentationEvent drained from backend.
        command_completed(object): Emits result of asynchronously submitted command.
        command_failed(str): Emits error string if command execution fails.
    """

    snapshot_updated = Signal(object)
    state_changed = Signal(str)
    amplitude_updated = Signal(float)
    event_dispatched = Signal(object)
    command_completed = Signal(object)
    command_failed = Signal(str)

    def __init__(
        self,
        presentation_adapter: PresentationAdapter,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize UIBridge with PresentationAdapter and start event polling.

        Args:
            presentation_adapter: Authoritative Phase 23.1 presentation boundary.
            poll_interval_ms: Timer tick rate in milliseconds for draining backend events.
            parent: Optional Qt parent QObject.
        """
        super().__init__(parent)
        self._adapter = presentation_adapter
        self._poll_interval_ms = max(10, poll_interval_ms)

        # Thread pool strictly for dispatching async coroutine submissions from Qt main thread
        self._async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="UIBridgeAsync")

        # Cache last seen values to avoid redundant signal emissions
        self._last_state: Optional[AssistantState] = None
        self._last_snapshot: Optional[AssistantSnapshot] = None
        self._last_amplitude: float = -1.0

        # High-frequency polling timer running safely on Qt main thread
        self._timer = QTimer(self)
        self._timer.setInterval(self._poll_interval_ms)
        self._timer.timeout.connect(self._poll_backend_events)
        self._timer.start()

        # Initial snapshot capture
        self._sync_current_snapshot()

    @property
    def adapter(self) -> PresentationAdapter:
        """Return attached PresentationAdapter."""
        return self._adapter

    @property
    def current_snapshot(self) -> AssistantSnapshot:
        """Return most recently fetched AssistantSnapshot."""
        if self._last_snapshot is None:
            self._sync_current_snapshot()
        return self._last_snapshot

    def _sync_current_snapshot(self) -> None:
        """Fetch current snapshot and emit signals if changed."""
        snap = self._adapter.get_snapshot()
        self._process_snapshot(snap)

    def _process_snapshot(self, snap: AssistantSnapshot) -> None:
        """Check state and amplitude diffs and emit corresponding Qt signals."""
        self._last_snapshot = snap

        # State change detection
        if snap.state != self._last_state:
            self._last_state = snap.state
            self.state_changed.emit(snap.state.value)

        # Amplitude telemetry change detection
        if abs(snap.mic_amplitude - self._last_amplitude) > 0.005:
            self._last_amplitude = snap.mic_amplitude
            self.amplitude_updated.emit(snap.mic_amplitude)

        self.snapshot_updated.emit(snap)

    def _poll_backend_events(self) -> None:
        """Timer callback draining bounded presentation queue non-blockingly."""
        try:
            events = self._adapter.drain_events(max_items=50)
            for event in events:
                self.event_dispatched.emit(event)
                # Event carries a point-in-time snapshot
                if event.snapshot:
                    self._process_snapshot(event.snapshot)
        except Exception as exc:
            logger.warning("Error draining events in UIBridge: %s", exc)

    # --------------------------------------------------------------------------
    # User Actions (Non-blocking Delegation to PresentationAdapter)
    # --------------------------------------------------------------------------

    def submit_command(self, text: str) -> None:
        """Submit a text command asynchronously without blocking the Qt event loop."""
        if not text or not text.strip():
            return

        def _worker() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._adapter.submit_command(text))
                loop.close()
                self.command_completed.emit(result)
            except Exception as exc:
                logger.error("Command failed via UIBridge: %s", exc)
                self.command_failed.emit(str(exc))

        self._async_executor.submit(_worker)

    def start_voice_interaction(self, duration: Optional[float] = None) -> None:
        """Start a voice interaction cycle asynchronously without blocking Qt."""
        def _worker() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(self._adapter.start_voice_interaction(duration=duration))
                loop.close()
                self.command_completed.emit(res)
            except Exception as exc:
                logger.error("Voice interaction failed via UIBridge: %s", exc)
                self.command_failed.emit(str(exc))

        self._async_executor.submit(_worker)

    def resolve_confirmation(
        self,
        confirmation_id: str,
        approved: bool,
        *,
        decided_by: str = "gui_operator",
        reason: str = "",
    ) -> bool:
        """Resolve a pending confirmation synchronously through PresentationAdapter."""
        return self._adapter.resolve_confirmation(
            confirmation_id,
            approved=approved,
            decided_by=decided_by,
            reason=reason,
        )

    def cancel_current_task(self, reason: Optional[str] = None) -> bool:
        """Cancel current task via PresentationAdapter."""
        return self._adapter.cancel_current_task(reason=reason or "Cancelled by user via UI")

    def close(self) -> None:
        """Stop polling timer and shutdown background executor."""
        if self._timer.isActive():
            self._timer.stop()
        self._async_executor.shutdown(wait=False)
