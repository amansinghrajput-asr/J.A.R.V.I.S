"""Presentation Adapter Subsystem for J.A.R.V.I.S. Phase 23.1.

Provides a unified, thread-safe, headless presentation boundary between the
backend services and future GUI / HUD clients (such as Phase 23.2 PySide6).
Maintains a bounded event queue with drop-oldest overflow policy, bridges
confirmation and cancellation calls to backend services without creating
competing thread pools, and exposes non-blocking asynchronous dispatch.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence

from app.core.container import ServiceContainer, container
from app.core.logger import get_logger
from app.core.state import (
    AssistantSnapshot,
    AssistantState,
    PendingConfirmation,
    PresentationEvent,
)
from app.core.state_manager import AssistantStateManager

logger = get_logger("PRESENTATION")

DEFAULT_QUEUE_MAX_SIZE = 1000


class PresentationQueue:
    """Thread-safe bounded event queue with drop-oldest overflow policy.

    Guarantees non-blocking enqueue for backend event emitters and non-blocking
    draining for GUI event loops.
    """

    def __init__(self, maxsize: int = DEFAULT_QUEUE_MAX_SIZE) -> None:
        """Initialize PresentationQueue.

        Args:
            maxsize: Maximum number of events retained before dropping oldest.
        """
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.RLock()
        self._queue: deque[PresentationEvent] = deque(maxlen=self._maxsize)
        self._dropped_count: int = 0

    @property
    def maxsize(self) -> int:
        """Maximum queue capacity."""
        return self._maxsize

    @property
    def dropped_count(self) -> int:
        """Total number of events dropped due to overflow."""
        with self._lock:
            return self._dropped_count

    def put(self, event: PresentationEvent) -> None:
        """Enqueue an event non-blockingly, dropping the oldest if capacity is reached."""
        with self._lock:
            if len(self._queue) >= self._maxsize:
                self._dropped_count += 1
            self._queue.append(event)

    def drain(self, max_items: int = 50) -> list[PresentationEvent]:
        """Drain up to max_items events from the queue non-blockingly."""
        items: list[PresentationEvent] = []
        with self._lock:
            count = min(max_items, len(self._queue))
            for _ in range(count):
                items.append(self._queue.popleft())
        return items

    def qsize(self) -> int:
        """Return current number of items in the queue."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear all events from queue."""
        with self._lock:
            self._queue.clear()


class PresentationAdapter:
    """Headless presentation boundary connecting backend services to presentation clients.

    Key guarantees:
    - Thread-safe snapshot inspection
    - Bounded event stream via PresentationQueue
    - Non-blocking asynchronous command & voice submission reusing existing pipelines
    - Safe delegation of confirmation approvals without security bypass
    - Safe delegation of task cancellations to ExecutionController
    - Zero Qt / PySide6 dependencies
    - Zero competing thread pools
    """

    def __init__(
        self,
        state_manager: AssistantStateManager,
        *,
        command_router: Optional[Any] = None,
        voice_engine: Optional[Any] = None,
        confirmation_manager: Optional[Any] = None,
        execution_controller: Optional[Any] = None,
        queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE,
        container_instance: Optional[ServiceContainer] = None,
    ) -> None:
        """Initialize PresentationAdapter.

        Args:
            state_manager: The authoritative AssistantStateManager instance.
            command_router: Optional CommandRouter for dispatching text commands.
            voice_engine: Optional VoiceConversationEngine for voice turns.
            confirmation_manager: Optional SystemConfirmationManager for approvals.
            execution_controller: Optional ExecutionController for cancellation.
            queue_max_size: Maximum size of the internal event queue.
            container_instance: Optional ServiceContainer for resolving audio_player/voice_engine.
        """
        self._lock = threading.RLock()
        self._state_manager = state_manager
        self._command_router = command_router
        self._voice_engine = voice_engine
        self._confirmation_manager = confirmation_manager
        self._execution_controller = execution_controller
        self._container = container_instance if container_instance is not None else container

        self._queue = PresentationQueue(maxsize=queue_max_size)

        # Connect state manager listener to enqueue events into presentation queue
        self._unsub_listener = self._state_manager.add_listener(self._handle_state_event)

    # --------------------------------------------------------------------------
    # Event Queue & Snapshot API
    # --------------------------------------------------------------------------

    def get_snapshot(self) -> AssistantSnapshot:
        """Return the current authoritative AssistantSnapshot."""
        return self._state_manager.get_snapshot()

    def drain_events(self, max_items: int = 50) -> list[PresentationEvent]:
        """Drain up to max_items events from the queue non-blockingly."""
        return self._queue.drain(max_items=max_items)

    def get_queue_size(self) -> int:
        """Return current number of queued presentation events."""
        return self._queue.qsize()

    def get_dropped_events_count(self) -> int:
        """Return count of dropped events due to buffer overflow."""
        return self._queue.dropped_count

    def _handle_state_event(self, event: PresentationEvent) -> None:
        """Callback invoked by StateManager on each state transition or telemetry event."""
        self._queue.put(event)

    # --------------------------------------------------------------------------
    # Command Dispatch (Text)
    # --------------------------------------------------------------------------

    async def submit_command(self, text: str) -> Any:
        """Submit a text command asynchronously to CommandRouter.

        Reuses CommandRouter.route_async() directly without creating a competing
        thread pool or executor.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Command text must be a non-empty string.")

        clean_text = text.strip()
        logger.info("PresentationAdapter submitting command: '%s'", clean_text)

        router = self._command_router
        if router is None:
            # Fallback error transition if router not configured
            self._state_manager.transition_to(
                AssistantState.ERROR,
                status_message="Command router not available",
                last_error="No CommandRouter registered",
            )
            raise RuntimeError("CommandRouter is not configured on PresentationAdapter.")

        try:
            return await router.route_async(clean_text, source="gui")
        except Exception as exc:
            logger.error("Command execution failed via PresentationAdapter: %s", exc)
            self._state_manager.transition_to(
                AssistantState.ERROR,
                status_message=f"Command failed: {exc}",
                last_error=str(exc),
            )
            raise

    # --------------------------------------------------------------------------
    # Voice Interaction
    # --------------------------------------------------------------------------

    async def start_voice_interaction(self, duration: Optional[float] = None) -> Any:
        """Trigger an asynchronous voice interaction cycle.

        Reuses VoiceConversationEngine.listen_once_async() directly.
        """
        engine = self._voice_engine
        if engine is None:
            self._state_manager.transition_to(
                AssistantState.ERROR,
                status_message="Voice engine not available",
                last_error="No VoiceConversationEngine registered",
            )
            raise RuntimeError("VoiceConversationEngine is not configured on PresentationAdapter.")

        logger.info("PresentationAdapter starting voice interaction...")
        try:
            return await engine.listen_once_async(duration=duration)
        except Exception as exc:
            logger.error("Voice interaction failed via PresentationAdapter: %s", exc)
            self._state_manager.transition_to(
                AssistantState.ERROR,
                status_message=f"Voice interaction failed: {exc}",
                last_error=str(exc),
            )
            raise

    # --------------------------------------------------------------------------
    # Confirmation Management Bridge
    # --------------------------------------------------------------------------

    def resolve_confirmation(
        self,
        confirmation_id: str,
        approved: bool,
        *,
        decided_by: str = "operator",
        reason: str = "",
    ) -> bool:
        """Resolve a pending confirmation request through SystemConfirmationManager.

        PresentationAdapter NEVER directly executes protected operations. It strictly
        delegates resolution through the confirmation manager.
        """
        if not confirmation_id:
            return False

        mgr = self._confirmation_manager
        if mgr is None:
            logger.warning("No SystemConfirmationManager available to resolve confirmation.")
            return False

        try:
            success = mgr.resolve_confirmation(
                confirmation_id,
                approved=approved,
                decided_by=decided_by,
                reason=reason,
            )
            if success:
                if approved:
                    self._state_manager.transition_to(
                        AssistantState.EXECUTING,
                        status_message=f"Confirmation approved for {confirmation_id}",
                    )
                else:
                    self._state_manager.transition_to(
                        AssistantState.IDLE,
                        status_message=f"Confirmation rejected for {confirmation_id}",
                    )
            return success
        except Exception as exc:
            logger.error("Error resolving confirmation %s: %s", confirmation_id, exc)
            return False

    # --------------------------------------------------------------------------
    # Cancellation & Interruption Bridge
    # --------------------------------------------------------------------------

    def interrupt_speech(self) -> bool:
        """Interrupt active audio playback / voice speech and return assistant to IDLE.

        Thread-safe, non-blocking, and never raises.
        """
        interrupted = False
        try:
            if self._container is not None and self._container.exists("audio_player"):
                player = self._container.resolve("audio_player")
                if hasattr(player, "interrupt"):
                    player.interrupt()
                    interrupted = True
            elif self._voice_engine is not None and hasattr(self._voice_engine, "interrupt"):
                self._voice_engine.interrupt()
                interrupted = True
            elif self._container is not None and self._container.exists("voice_engine"):
                v_engine = self._container.resolve("voice_engine")
                if hasattr(v_engine, "interrupt"):
                    v_engine.interrupt()
                    interrupted = True
        except Exception as exc:
            logger.debug("Error during interrupt_speech: %s", exc)

        if self._state_manager is not None:
            try:
                # Use get_snapshot() — AssistantStateManager has no .snapshot property
                snap = self._state_manager.get_snapshot() if hasattr(self._state_manager, "get_snapshot") else None
                curr_state = snap.state if snap is not None else getattr(self._state_manager, "_state", None)
                if curr_state in (AssistantState.SPEAKING, AssistantState.LISTENING, AssistantState.THINKING):
                    self._state_manager.transition_to(
                        AssistantState.IDLE,
                        status_message="Speech interrupted",
                    )
            except Exception as exc:
                logger.debug("Error transitioning state during interrupt_speech: %s", exc)

        return interrupted

    def cancel_current_task(self, reason: Optional[str] = None) -> bool:
        """Cancel current running task or plan via ExecutionController and stop active audio.

        Delegates safely to ExecutionController.cancel(). Never terminates processes
        directly.
        """
        self.interrupt_speech()

        ctrl = self._execution_controller
        if ctrl is None:
            logger.warning("No ExecutionController configured to perform cancellation.")
            # Still transition state manager safely back to IDLE
            self._state_manager.transition_to(
                AssistantState.IDLE,
                status_message=f"Cancelled: {reason or 'User requested'}",
            )
            return False

        try:
            ctrl.cancel(reason=reason or "Cancelled by user via presentation adapter")
            self._state_manager.transition_to(
                AssistantState.IDLE,
                status_message="Task cancelled by user",
            )
            return True
        except Exception as exc:
            logger.error("Error invoking cancel on ExecutionController: %s", exc)
            return False

    # --------------------------------------------------------------------------
    # Lifecycle Cleanup
    # --------------------------------------------------------------------------

    def close(self) -> None:
        """Unsubscribe listeners and clean up presentation queue."""
        with self._lock:
            try:
                self._unsub_listener()
            except Exception:
                pass
            self._queue.clear()


# ------------------------------------------------------------------------------
# Dependency Injection Factory
# ------------------------------------------------------------------------------

def create_presentation_adapter(
    container_instance: Optional[ServiceContainer] = None,
    *,
    queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE,
) -> PresentationAdapter:
    """Factory creating and configuring PresentationAdapter from ServiceContainer.

    Resolves dependencies defensively, registering state_manager and presentation_adapter
    in the container if not already present.
    """
    sc = container_instance if container_instance is not None else container

    # Fast return if presentation_adapter is already registered in container
    if sc.exists("presentation_adapter"):
        return sc.resolve("presentation_adapter")

    # 1. Resolve or create AssistantStateManager
    if sc.exists("state_manager"):
        state_mgr: AssistantStateManager = sc.resolve("state_manager")
    else:
        eb = sc.resolve("event_bus") if sc.exists("event_bus") else None
        peb = sc.resolve("planner_event_bus") if sc.exists("planner_event_bus") else None
        state_mgr = AssistantStateManager(event_bus=eb, planner_event_bus=peb)
        try:
            sc.register_singleton("state_manager", state_mgr)
        except Exception:
            pass

    # 2. Resolve optional components defensively
    cmd_router = sc.resolve("router") if sc.exists("router") else None
    if cmd_router is None and sc.exists("command_router"):
        cmd_router = sc.resolve("command_router")

    v_engine = sc.resolve("voice_engine") if sc.exists("voice_engine") else None
    conf_mgr = sc.resolve("system_confirmation_manager") if sc.exists("system_confirmation_manager") else None
    exec_ctrl = sc.resolve("execution_controller") if sc.exists("execution_controller") else None

    adapter = PresentationAdapter(
        state_manager=state_mgr,
        command_router=cmd_router,
        voice_engine=v_engine,
        confirmation_manager=conf_mgr,
        execution_controller=exec_ctrl,
        queue_max_size=queue_max_size,
    )

    try:
        if not sc.exists("presentation_adapter"):
            sc.register_singleton("presentation_adapter", adapter)
    except Exception:
        pass

    return adapter
