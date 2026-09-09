"""Execution Timeline Recorder for the J.A.R.V.I.S. Planner Subsystem.

Subscribes to PlannerEventBus, records planner events in deterministic order,
provides querying/filtering and analytical duration calculation, and exports
to JSON and Markdown format.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from app.ai.planner.events import (
    PlanCompleted,
    PlanFailed,
    PlannerEvent,
    PlannerEventBus,
    PlanStarted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
)

TEvent = TypeVar("TEvent", bound=PlannerEvent)


class ExecutionTimeline:
    """Records, filters, and analyzes planner events in chronological sequence."""

    def __init__(self, bus: Optional[PlannerEventBus] = None) -> None:
        """Initialize ExecutionTimeline, optionally attaching to an event bus.

        Args:
            bus: Optional PlannerEventBus to automatically subscribe to.
        """
        self._lock = threading.RLock()
        self._events: List[PlannerEvent] = []
        self._bus: Optional[PlannerEventBus] = bus
        self._unsubscribe_fn: Optional[Callable[[], bool]] = None

        if bus is not None:
            self.attach(bus)

    def attach(self, bus: PlannerEventBus) -> None:
        """Attach to a PlannerEventBus and start recording events."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                self.detach()
            self._bus = bus
            self._unsubscribe_fn = bus.subscribe(PlannerEvent, self.record)

    def detach(self) -> None:
        """Detach from the active PlannerEventBus."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                try:
                    self._unsubscribe_fn()
                except Exception:
                    pass
                self._unsubscribe_fn = None
            self._bus = None

    def record(self, event: PlannerEvent) -> None:
        """Record an event, maintaining deterministic ordering by (sequence_id, timestamp)."""
        with self._lock:
            # Optimize append for ordered events, binary insert if out of order
            if not self._events or (
                self._events[-1].sequence_id <= event.sequence_id
                and self._events[-1].timestamp <= event.timestamp
            ):
                self._events.append(event)
            else:
                self._events.append(event)
                self._events.sort(key=lambda e: (e.sequence_id, e.timestamp))

    def clear(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()

    @property
    def events(self) -> List[PlannerEvent]:
        """Return a shallow copy of recorded events."""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # --------------------------------------------------------------------------
    # Filtering APIs
    # --------------------------------------------------------------------------

    def filter_by_task(self, task_id: str) -> List[PlannerEvent]:
        """Filter recorded events associated with a specific task ID."""
        with self._lock:
            matched: List[PlannerEvent] = []
            for e in self._events:
                # Direct task_id attribute
                if getattr(e, "task_id", None) == task_id:
                    matched.append(e)
                # In recovery failure lists
                elif isinstance(e, RecoveryStarted):
                    if task_id in e.failed_task_ids or task_id in e.skipped_task_ids:
                        matched.append(e)
            return matched

    def filter_by_type(self, event_type: Type[TEvent]) -> List[TEvent]:
        """Filter recorded events by specific event type or subclass."""
        with self._lock:
            return [e for e in self._events if isinstance(e, event_type)]

    def filter_by_time(self, start: float, end: float) -> List[PlannerEvent]:
        """Filter events falling within a timestamp window [start, end]."""
        with self._lock:
            return [e for e in self._events if start <= e.timestamp <= end]

    # --------------------------------------------------------------------------
    # Analytics APIs
    # --------------------------------------------------------------------------

    def get_plan_duration(self) -> float:
        """Calculate total plan duration from PlanStarted to completion/failure, or full span."""
        with self._lock:
            if not self._events:
                return 0.0

            # Check explicit PlanCompleted
            for e in reversed(self._events):
                if isinstance(e, PlanCompleted):
                    return e.duration

            # Span from first to last event
            start_ts = self._events[0].timestamp
            end_ts = self._events[-1].timestamp
            return max(0.0, end_ts - start_ts)

    def get_task_durations(self) -> Dict[str, float]:
        """Return mapping of task_id to execution duration in seconds."""
        with self._lock:
            durations: Dict[str, float] = {}
            for e in self._events:
                if isinstance(e, (TaskCompleted, TaskFailed)):
                    durations[e.task_id] = e.duration
            return durations

    def get_task_retries(self, task_id: Optional[str] = None) -> List[TaskRetried]:
        """Return list of retry events, optionally filtered by task ID."""
        with self._lock:
            retries = [e for e in self._events if isinstance(e, TaskRetried)]
            if task_id is not None:
                return [r for r in retries if r.task_id == task_id]
            return retries

    def get_recovery_waves(self) -> List[Dict[str, Any]]:
        """Return structured timeline breakdown of recovery replanning waves."""
        with self._lock:
            waves: List[Dict[str, Any]] = []
            current_wave: Optional[Dict[str, Any]] = None

            for e in self._events:
                if isinstance(e, RecoveryStarted):
                    current_wave = {
                        "attempt": e.attempt,
                        "start_time": e.timestamp,
                        "failed_tasks": list(e.failed_task_ids),
                        "skipped_tasks": list(e.skipped_task_ids),
                        "status": "IN_PROGRESS",
                        "duration": 0.0,
                    }
                    waves.append(current_wave)
                elif isinstance(e, RecoveryCompleted) and current_wave is not None:
                    current_wave["duration"] = e.duration
                    current_wave["success"] = e.success
                    current_wave["status"] = "SUCCESS" if e.success else "FAILED"
                    current_wave["new_plan_id"] = e.new_plan_id
                    current_wave["task_count"] = e.task_count
                    current_wave = None
                elif isinstance(e, RecoveryFailed) and current_wave is not None:
                    current_wave["duration"] = e.duration
                    current_wave["status"] = "FAILED"
                    current_wave["reason"] = e.reason
                    current_wave = None

            return waves

    # --------------------------------------------------------------------------
    # Export APIs
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize timeline and events to a dictionary."""
        with self._lock:
            return {
                "total_events": len(self._events),
                "plan_duration": self.get_plan_duration(),
                "task_durations": self.get_task_durations(),
                "recovery_waves": self.get_recovery_waves(),
                "events": [e.to_dict() for e in self._events],
            }

    def to_json(self, indent: int = 2) -> str:
        """Export timeline to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export timeline as an easy-to-read Markdown document."""
        with self._lock:
            if not self._events:
                return "# Execution Timeline\n\n*No events recorded.*\n"

            first_ts = self._events[0].timestamp
            lines: List[str] = [
                "# Planner Execution Timeline",
                "",
                f"- **Total Events Recorded**: {len(self._events)}",
                f"- **Total Plan Duration**: {self.get_plan_duration():.3f}s",
                "",
                "| Seq | Relative Time | Event | Task ID / Detail | Duration | Status |",
                "| :---: | :---: | :--- | :--- | :---: | :---: |",
            ]

            for e in self._events:
                rel_time = f"+{(e.timestamp - first_ts):.3f}s"
                event_name = type(e).__name__
                detail = "-"
                duration_str = "-"
                status_icon = "ℹ️"

                if isinstance(e, PlanStarted):
                    detail = f"Query: `{e.query}` ({e.task_count} tasks)"
                    status_icon = "🚀"
                elif isinstance(e, PlanCompleted):
                    detail = f"{e.completed_count} completed, {e.failed_count} failed, {e.skipped_count} skipped"
                    duration_str = f"{e.duration:.3f}s"
                    status_icon = "✅" if e.success else "❌"
                elif isinstance(e, PlanFailed):
                    detail = f"Error: `{e.error}`"
                    status_icon = "💥"
                elif isinstance(e, TaskStarted):
                    detail = f"`{e.task_id}` ({e.action})"
                    status_icon = "⏳"
                elif isinstance(e, TaskCompleted):
                    detail = f"`{e.task_id}` ({e.action}) -> {e.result[:30]}"
                    duration_str = f"{e.duration:.3f}s"
                    status_icon = "✅"
                elif isinstance(e, TaskFailed):
                    detail = f"`{e.task_id}` ({e.action}) -> {e.error[:30]}"
                    duration_str = f"{e.duration:.3f}s"
                    status_icon = "❌"
                elif isinstance(e, TaskRetried):
                    detail = f"`{e.task_id}` attempt #{e.attempt} ({e.reason})"
                    status_icon = "🔄"
                elif isinstance(e, RecoveryStarted):
                    detail = f"Attempt #{e.attempt} for {len(e.failed_task_ids)} failed tasks"
                    status_icon = "🩹"
                elif isinstance(e, RecoveryCompleted):
                    detail = f"Attempt #{e.attempt} ({e.task_count} tasks generated)"
                    duration_str = f"{e.duration:.3f}s"
                    status_icon = "✅" if e.success else "⚠️"
                elif isinstance(e, RecoveryFailed):
                    detail = f"Attempt #{e.attempt} rejected: {e.reason}"
                    duration_str = f"{e.duration:.3f}s"
                    status_icon = "🛑"
                else:
                    detail = str(e.metadata) if e.metadata else "-"

                lines.append(
                    f"| {e.sequence_id} | {rel_time} | `{event_name}` | {detail} | {duration_str} | {status_icon} |"
                )

            return "\n".join(lines) + "\n"
