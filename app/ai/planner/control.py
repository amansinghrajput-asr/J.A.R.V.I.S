"""Execution Control Subsystem for J.A.R.V.I.S. Adaptive Planner.

Provides thread-safe and asyncio-compatible execution control primitives:
pause, resume, cancellation, and recovery abortion without busy-waiting.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, List, Optional

from app.ai.planner.events import (
    PlanCancelled,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    RecoveryAborted,
)

logger = logging.getLogger("app.ai.planner.control")


class ExecutionCancelledError(Exception):
    """Raised at execution checkpoints when a cancellation token is activated."""

    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason or "Execution cancelled."
        super().__init__(self.reason)


class ExecutionPausedError(Exception):
    """Raised when an operation cannot proceed due to suspended execution state."""

    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason or "Execution paused."
        super().__init__(self.reason)


class CancellationToken:
    """Thread-safe cancellation token for propagating cooperative abort signals."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cancelled = False
        self._reason: Optional[str] = None

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> Optional[str]:
        """Return reason for cancellation if provided."""
        with self._lock:
            return self._reason

    def cancel(self, reason: Optional[str] = None) -> None:
        """Signal cancellation."""
        with self._lock:
            self._cancelled = True
            if reason is not None:
                self._reason = reason

    def raise_if_cancelled(self) -> None:
        """Raise ExecutionCancelledError if cancellation has been requested."""
        if self.is_cancelled:
            raise ExecutionCancelledError(self.reason)


class ExecutionController:
    """Coordinates lifecycle execution control: pause, resume, cancel, and abort recovery.

    Guarantees thread-safe and asyncio-compatible synchronization without busy waiting,
    leveraging threading.Event and asyncio.Event primitives.
    """

    def __init__(
        self,
        event_bus: Optional[PlannerEventBus] = None,
        execution_id: str = "",
        plan_id: str = "",
    ) -> None:
        """Initialize ExecutionController.

        Args:
            event_bus: Optional PlannerEventBus to publish control events to.
            execution_id: Correlation ID for execution instances.
            plan_id: Identifier of the plan under control.
        """
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self.execution_id = execution_id
        self.plan_id = plan_id

        self._token = CancellationToken()
        self._is_paused = False
        self._pause_reason: Optional[str] = None
        self._recovery_aborted = False
        self._recovery_abort_reason: Optional[str] = None

        # Threading event: SET when running, CLEARED when paused
        self._thread_gate = threading.Event()
        self._thread_gate.set()

        # Asyncio event: lazy-initialized per active event loop
        self._async_gate: Optional[asyncio.Event] = None
        self._async_gate_loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks for observability hooks
        self._pause_callbacks: List[Callable[[str], Any]] = []
        self._resume_callbacks: List[Callable[[str], Any]] = []
        self._cancel_callbacks: List[Callable[[str], Any]] = []

    # --------------------------------------------------------------------------
    # Status Properties
    # --------------------------------------------------------------------------

    @property
    def cancellation_token(self) -> CancellationToken:
        """Return the attached CancellationToken."""
        return self._token

    @property
    def is_paused(self) -> bool:
        """Return True if execution is currently paused."""
        with self._lock:
            return self._is_paused

    @property
    def is_cancelled(self) -> bool:
        """Return True if execution has been cancelled."""
        return self._token.is_cancelled

    @property
    def cancellation_reason(self) -> Optional[str]:
        """Return cancellation reason if cancelled."""
        return self._token.reason

    @property
    def is_recovery_aborted(self) -> bool:
        """Return True if recovery replanning has been explicitly aborted."""
        with self._lock:
            return self._recovery_aborted

    @property
    def recovery_abort_reason(self) -> Optional[str]:
        """Return recovery abort reason if aborted."""
        with self._lock:
            return self._recovery_abort_reason

    # --------------------------------------------------------------------------
    # Control Actions
    # --------------------------------------------------------------------------

    def pause(self, reason: Optional[str] = None) -> None:
        """Pause execution without busy waiting.

        Blocks execution checkpoints until resume() or cancel() is invoked.
        """
        with self._lock:
            if self._is_paused:
                return
            self._is_paused = True
            self._pause_reason = reason or "Execution paused"
            self._thread_gate.clear()
            self._update_async_gate(running=False)

        logger.info("Execution paused: %s", self._pause_reason)

        if self._event_bus is not None:
            self._event_bus.publish(
                PlanPaused(
                    execution_id=self.execution_id,
                    plan_id=self.plan_id,
                    reason=self._pause_reason,
                )
            )

        for cb in list(self._pause_callbacks):
            try:
                cb(self._pause_reason)
            except Exception as exc:
                logger.warning("Error in pause callback: %s", exc)

    def resume(self, reason: Optional[str] = None) -> None:
        """Resume paused execution.

        Unblocks all threads and tasks waiting at execution checkpoints.
        """
        with self._lock:
            if not self._is_paused:
                return
            self._is_paused = False
            resume_reason = reason or "Execution resumed"
            self._pause_reason = None
            self._thread_gate.set()
            self._update_async_gate(running=True)

        logger.info("Execution resumed: %s", resume_reason)

        if self._event_bus is not None:
            self._event_bus.publish(
                PlanResumed(
                    execution_id=self.execution_id,
                    plan_id=self.plan_id,
                    reason=resume_reason,
                )
            )

        for cb in list(self._resume_callbacks):
            try:
                cb(resume_reason)
            except Exception as exc:
                logger.warning("Error in resume callback: %s", exc)

    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel execution.

        Unblocks paused threads so they can immediately exit cleanly via
        ExecutionCancelledError at the nearest checkpoint.
        """
        cancel_reason = reason or "Execution cancelled"
        with self._lock:
            self._token.cancel(cancel_reason)
            # Unblock gate so paused threads can wake up and detect cancellation
            self._thread_gate.set()
            self._update_async_gate(running=True)

        logger.info("Execution cancelled: %s", cancel_reason)

        if self._event_bus is not None:
            self._event_bus.publish(
                PlanCancelled(
                    execution_id=self.execution_id,
                    plan_id=self.plan_id,
                    reason=cancel_reason,
                )
            )

        for cb in list(self._cancel_callbacks):
            try:
                cb(cancel_reason)
            except Exception as exc:
                logger.warning("Error in cancel callback: %s", exc)

    def abort_recovery(self, reason: Optional[str] = None) -> None:
        """Explicitly abort recovery replanning loop without cancelling overall context."""
        abort_reason = reason or "Recovery aborted"
        with self._lock:
            self._recovery_aborted = True
            self._recovery_abort_reason = abort_reason

        logger.info("Recovery aborted: %s", abort_reason)

        if self._event_bus is not None:
            self._event_bus.publish(
                RecoveryAborted(
                    execution_id=self.execution_id,
                    reason=abort_reason,
                )
            )

    # --------------------------------------------------------------------------
    # Checkpoint Interceptors
    # --------------------------------------------------------------------------

    def check_checkpoint(self, checkpoint_name: str = "") -> None:
        """Synchronous execution checkpoint.

        1. Checks cancellation; raises ExecutionCancelledError if cancelled.
        2. If paused, synchronously blocks on threading.Event without busy waiting.
        3. Upon waking, checks cancellation again.

        Args:
            checkpoint_name: Diagnostic identifier (e.g. 'before_task_t1', 'between_waves').
        """
        self._token.raise_if_cancelled()

        # If paused, block on the thread gate
        if not self._thread_gate.is_set():
            logger.debug("Execution checkpoint '%s' paused; awaiting resume signal.", checkpoint_name)
            self._thread_gate.wait()

        # Re-check cancellation upon unblocking
        self._token.raise_if_cancelled()

    async def check_checkpoint_async(self, checkpoint_name: str = "") -> None:
        """Asynchronous execution checkpoint.

        1. Checks cancellation; raises ExecutionCancelledError if cancelled.
        2. If paused, awaits asyncio.Event without blocking event loop or busy waiting.
        3. Upon waking, checks cancellation again.

        Args:
            checkpoint_name: Diagnostic identifier.
        """
        self._token.raise_if_cancelled()

        gate = self._get_async_gate()
        if not gate.is_set():
            logger.debug("Async execution checkpoint '%s' paused; awaiting resume signal.", checkpoint_name)
            await gate.wait()

        self._token.raise_if_cancelled()

    # --------------------------------------------------------------------------
    # Internal Synchronization Helpers
    # --------------------------------------------------------------------------

    def _get_async_gate(self) -> asyncio.Event:
        """Retrieve or create an asyncio.Event bound to the current running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            if self._async_gate is None or self._async_gate_loop != loop:
                self._async_gate_loop = loop
                self._async_gate = asyncio.Event()
                if not self._is_paused:
                    self._async_gate.set()
                else:
                    self._async_gate.clear()
            return self._async_gate

    def _update_async_gate(self, running: bool) -> None:
        """Update the state of the active asyncio gate."""
        if self._async_gate is not None:
            if running:
                self._async_gate.set()
            else:
                self._async_gate.clear()

    # --------------------------------------------------------------------------
    # Hook Registration
    # --------------------------------------------------------------------------

    def add_pause_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback to invoke on pause."""
        with self._lock:
            self._pause_callbacks.append(callback)

    def add_resume_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback to invoke on resume."""
        with self._lock:
            self._resume_callbacks.append(callback)

    def add_cancel_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback to invoke on cancellation."""
        with self._lock:
            self._cancel_callbacks.append(callback)
