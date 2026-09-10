"""Offline Planner Replay Engine for J.A.R.V.I.S. Adaptive Planner.

Provides deterministic, step-by-step navigation, time-seeking, and lifecycle
state snapshot inspection across past executions entirely offline.
NEVER invokes LLMs, external APIs, skills, or network routers.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEvent,
    PlanPaused,
    PlanResumed,
    PlanStarted,
    RecoveryAborted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
    TaskTimeout,
)
from app.ai.planner.models import TaskStatus
from app.ai.planner.persistence import PersistedExecutionState
from app.ai.planner.timeline import ExecutionTimeline

# Event class registry for deserializing raw event dicts
EVENT_CLASS_MAP = {
    "PlanStarted": PlanStarted,
    "PlanCompleted": PlanCompleted,
    "PlanFailed": PlanFailed,
    "TaskStarted": TaskStarted,
    "TaskCompleted": TaskCompleted,
    "TaskFailed": TaskFailed,
    "TaskRetried": TaskRetried,
    "RecoveryStarted": RecoveryStarted,
    "RecoveryCompleted": RecoveryCompleted,
    "RecoveryFailed": RecoveryFailed,
    "PlanPaused": PlanPaused,
    "PlanResumed": PlanResumed,
    "PlanCancelled": PlanCancelled,
    "TaskTimeout": TaskTimeout,
    "RecoveryAborted": RecoveryAborted,
    "PlannerEvent": PlannerEvent,
}


@dataclass
class ReplaySnapshot:
    """State snapshot of execution lifecycle at a specific replay cursor point.

    Attributes:
        cursor: Zero-indexed position in event stream (-1 represents initial unstarted state).
        total_steps: Total number of recorded events in timeline.
        timestamp: Epoch timestamp of the event at current cursor (0.0 if cursor < 0).
        elapsed_duration: Elapsed time from initial event to current cursor event.
        current_execution_progress: Fraction of timeline completed [0.0, 1.0].
        current_event: Serialized representation of event at current cursor.
        task_lifecycle: Current TaskStatus for each known task ID.
        completed_tasks: Task IDs that have reached COMPLETED status.
        failed_tasks: Task IDs whose latest state is FAILED.
        skipped_tasks: Task IDs whose latest state is SKIPPED.
        pending_tasks: Task IDs that have been defined but not yet started.
        recovery_wave: Current active recovery attempt wave number.
        is_paused: Whether execution is currently paused at this step.
        is_cancelled: Whether execution has been cancelled at this step.
    """

    cursor: int
    total_steps: int
    timestamp: float = 0.0
    elapsed_duration: float = 0.0
    current_execution_progress: float = 0.0
    current_event: Optional[Dict[str, Any]] = None
    task_lifecycle: Dict[str, str] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    skipped_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    recovery_wave: int = 1
    is_paused: bool = False
    is_cancelled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "cursor": self.cursor,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
            "elapsed_duration": self.elapsed_duration,
            "current_execution_progress": self.current_execution_progress,
            "current_event": self.current_event,
            "task_lifecycle": dict(self.task_lifecycle),
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "skipped_tasks": list(self.skipped_tasks),
            "pending_tasks": list(self.pending_tasks),
            "recovery_wave": self.recovery_wave,
            "is_paused": self.is_paused,
            "is_cancelled": self.is_cancelled,
        }


class PlannerReplayEngine:
    """Offline deterministic replay engine for planner event timelines.

    Enables non-destructive step forward, step backward, index seek, and timestamp
    seeking through historical executions without invoking any LLMs or external services.
    """

    def __init__(
        self,
        source: Optional[Union[ExecutionTimeline, PersistedExecutionState, List[PlannerEvent], List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initialize PlannerReplayEngine.

        Args:
            source: Timeline, PersistedExecutionState, or list of PlannerEvents/dicts.
        """
        self._events: List[PlannerEvent] = []
        self._initial_known_tasks: Set[str] = set()
        self._cursor: int = -1
        self._snapshots_cache: List[ReplaySnapshot] = []

        if source is not None:
            self.load(source)

    def load(
        self,
        source: Union[ExecutionTimeline, PersistedExecutionState, List[PlannerEvent], List[Dict[str, Any]]],
    ) -> None:
        """Load an execution source into the replay engine and reset navigation."""
        raw_events: List[Any] = []
        self._initial_known_tasks.clear()

        if isinstance(source, ExecutionTimeline):
            raw_events = source.events
        elif isinstance(source, PersistedExecutionState):
            # Extract known tasks from persisted state
            if "task_map" in source.dag_state and isinstance(source.dag_state["task_map"], dict):
                self._initial_known_tasks.update(source.dag_state["task_map"].keys())
            self._initial_known_tasks.update(t.id for t in source.completed_tasks)
            self._initial_known_tasks.update(t.id for t in source.failed_tasks)
            self._initial_known_tasks.update(t.id for t in source.skipped_tasks)
            # Reconstruct events from records if timeline not attached
            raw_events = self._events_from_persisted_state(source)
        elif isinstance(source, list):
            raw_events = source
        else:
            raise TypeError(f"Unsupported source type for replay: {type(source).__name__}")

        # Normalize events
        self._events = []
        for item in raw_events:
            if isinstance(item, PlannerEvent):
                self._events.append(item)
            elif isinstance(item, dict):
                event_type_name = item.get("event_type", "PlannerEvent")
                cls = EVENT_CLASS_MAP.get(event_type_name, PlannerEvent)
                clean_dict = {k: v for k, v in item.items() if k != "event_type"}
                try:
                    self._events.append(cls(**clean_dict))
                except Exception:
                    self._events.append(PlannerEvent(metadata=dict(item)))

        # Deterministic sorting
        self._events.sort(key=lambda e: (e.sequence_id, e.timestamp))
        self._precompute_snapshots()
        self.reset()

    def _events_from_persisted_state(self, state: PersistedExecutionState) -> List[PlannerEvent]:
        """Synthesize chronological events from PersistedExecutionState memory records."""
        synth_events: List[PlannerEvent] = []
        plan_id = state.plan_id or state.execution_id

        # Plan started
        synth_events.append(
            PlanStarted(
                sequence_id=1,
                timestamp=state.created_at,
                execution_id=state.execution_id,
                plan_id=plan_id,
                query=state.query,
                task_count=len(state.completed_tasks) + len(state.failed_tasks) + len(state.skipped_tasks),
            )
        )

        seq = 2
        # Task events from memory records
        for record in state.memory.records:
            t_start = record.timestamp - record.duration
            synth_events.append(
                TaskStarted(
                    sequence_id=seq,
                    timestamp=t_start,
                    execution_id=state.execution_id,
                    plan_id=plan_id,
                    task_id=record.task_id,
                    action=record.action,
                    target=record.target,
                )
            )
            seq += 1
            if record.status == TaskStatus.COMPLETED:
                synth_events.append(
                    TaskCompleted(
                        sequence_id=seq,
                        timestamp=record.timestamp,
                        execution_id=state.execution_id,
                        plan_id=plan_id,
                        task_id=record.task_id,
                        action=record.action,
                        result=str(record.output or "OK"),
                        duration=record.duration,
                    )
                )
            elif record.status == TaskStatus.FAILED:
                synth_events.append(
                    TaskFailed(
                        sequence_id=seq,
                        timestamp=record.timestamp,
                        execution_id=state.execution_id,
                        plan_id=plan_id,
                        task_id=record.task_id,
                        action=record.action,
                        error=str(record.error or "Failed"),
                        duration=record.duration,
                    )
                )
            seq += 1

        # Plan completed
        synth_events.append(
            PlanCompleted(
                sequence_id=seq,
                timestamp=state.updated_at,
                execution_id=state.execution_id,
                plan_id=plan_id,
                success=len(state.failed_tasks) == 0 and len(state.skipped_tasks) == 0,
                completed_count=len(state.completed_tasks),
                failed_count=len(state.failed_tasks),
                skipped_count=len(state.skipped_tasks),
                duration=max(0.0, state.updated_at - state.created_at),
            )
        )
        return synth_events

    def _precompute_snapshots(self) -> None:
        """Precompute cumulative state snapshots for each event in the timeline."""
        self._snapshots_cache = []
        if not self._events:
            return

        total_steps = len(self._events)
        t_first = self._events[0].timestamp

        # Cumulative trackers
        known_tasks: Set[str] = set(self._initial_known_tasks)
        task_states: Dict[str, str] = {tid: TaskStatus.PENDING.value for tid in self._initial_known_tasks}
        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        skipped_tasks: Set[str] = set()
        current_wave: int = 1
        is_paused: bool = False
        is_cancelled: bool = False

        for idx, event in enumerate(self._events):
            # Process event lifecycle impacts
            if isinstance(event, PlanStarted):
                pass
            elif isinstance(event, TaskStarted):
                known_tasks.add(event.task_id)
                task_states[event.task_id] = TaskStatus.RUNNING.value
            elif isinstance(event, TaskCompleted):
                known_tasks.add(event.task_id)
                task_states[event.task_id] = TaskStatus.COMPLETED.value
                completed_tasks.add(event.task_id)
                failed_tasks.discard(event.task_id)
                skipped_tasks.discard(event.task_id)
            elif isinstance(event, TaskFailed):
                known_tasks.add(event.task_id)
                task_states[event.task_id] = TaskStatus.FAILED.value
                failed_tasks.add(event.task_id)
                completed_tasks.discard(event.task_id)
                skipped_tasks.discard(event.task_id)
            elif isinstance(event, RecoveryStarted):
                current_wave = event.attempt
                for tid in event.skipped_task_ids:
                    known_tasks.add(tid)
                    task_states[tid] = TaskStatus.SKIPPED.value
                    skipped_tasks.add(tid)
            elif isinstance(event, PlanPaused):
                is_paused = True
            elif isinstance(event, PlanResumed):
                is_paused = False
            elif isinstance(event, PlanCancelled):
                is_cancelled = True
            elif isinstance(event, TaskTimeout):
                if event.policy in ("SKIP", "ABORT"):
                    known_tasks.add(event.task_id)
                    task_states[event.task_id] = TaskStatus.SKIPPED.value
                    skipped_tasks.add(event.task_id)

            pending = [tid for tid in known_tasks if task_states.get(tid) == TaskStatus.PENDING.value]
            progress = (idx + 1) / total_steps
            elapsed = max(0.0, event.timestamp - t_first)

            snap = ReplaySnapshot(
                cursor=idx,
                total_steps=total_steps,
                timestamp=event.timestamp,
                elapsed_duration=elapsed,
                current_execution_progress=progress,
                current_event=event.to_dict(),
                task_lifecycle=dict(task_states),
                completed_tasks=sorted(completed_tasks),
                failed_tasks=sorted(failed_tasks),
                skipped_tasks=sorted(skipped_tasks),
                pending_tasks=sorted(pending),
                recovery_wave=current_wave,
                is_paused=is_paused,
                is_cancelled=is_cancelled,
            )
            self._snapshots_cache.append(snap)

    # --------------------------------------------------------------------------
    # Navigation APIs
    # --------------------------------------------------------------------------

    def step(self) -> Optional[Dict[str, Any]]:
        """Advance cursor one step forward and return the new snapshot, or None if at end."""
        if not self._events:
            return None
        if self._cursor < len(self._events) - 1:
            self._cursor += 1
            return self.get_snapshot()
        return None

    def step_back(self) -> Optional[Dict[str, Any]]:
        """Move cursor one step backward and return the new snapshot, or None if at beginning."""
        if not self._events:
            return None
        if self._cursor > 0:
            self._cursor -= 1
            return self.get_snapshot()
        if self._cursor == 0:
            self._cursor = -1
            return self.get_snapshot()
        return None

    def seek(self, index: int) -> Dict[str, Any]:
        """Jump cursor directly to index position [-1, total_steps - 1]."""
        total = len(self._events)
        if total == 0:
            self._cursor = -1
            return self.get_snapshot()
        self._cursor = max(-1, min(index, total - 1))
        return self.get_snapshot()

    def seek_time(self, timestamp: float) -> Dict[str, Any]:
        """Jump to the event closest to or immediately preceding the given epoch timestamp."""
        if not self._events:
            return self.get_snapshot()

        timestamps = [e.timestamp for e in self._events]
        # Binary search for rightmost timestamp <= target
        idx = bisect.bisect_right(timestamps, timestamp) - 1
        return self.seek(idx)

    def reset(self) -> Dict[str, Any]:
        """Reset cursor to before the initial step."""
        self._cursor = -1
        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        """Return snapshot dictionary reflecting execution state at current cursor position."""
        if not self._events or self._cursor < 0:
            # Initial pre-execution snapshot
            return ReplaySnapshot(
                cursor=self._cursor,
                total_steps=len(self._events),
                timestamp=self._events[0].timestamp if self._events else 0.0,
                elapsed_duration=0.0,
                current_execution_progress=0.0,
                current_event=None,
                task_lifecycle={tid: TaskStatus.PENDING.value for tid in self._initial_known_tasks},
                completed_tasks=[],
                failed_tasks=[],
                skipped_tasks=[],
                pending_tasks=sorted(self._initial_known_tasks),
                recovery_wave=1,
                is_paused=False,
                is_cancelled=False,
            ).to_dict()

        return self._snapshots_cache[self._cursor].to_dict()

    @property
    def total_steps(self) -> int:
        """Total number of events loaded in engine."""
        return len(self._events)

    @property
    def current_cursor(self) -> int:
        """Current zero-based event cursor index (-1 if before first step)."""
        return self._cursor

    @property
    def is_at_start(self) -> bool:
        """True if cursor is at or before first step."""
        return self._cursor <= 0

    @property
    def is_at_end(self) -> bool:
        """True if cursor has reached final event."""
        return self._cursor >= len(self._events) - 1
