"""Visual Temporal Context & Event History Engine for J.A.R.V.I.S (Phase 27.15).

Provides a bounded, session-scoped temporal event layer that derives and records
meaningful visual transitions across sequential observations without duplicating
underlying tracking identity logic or capturing screens.

Visual Temporal Pipeline:
ScreenObservation
    ↓
existing perception engines
    ↓
VisualTrackingResult
    ↓
VisualTemporalEngine
    ↓
VisualTemporalEvent
    ↓
bounded in-memory history (VisualTemporalLog)
    ↓
semantic temporal queries
    ↓
voice-safe VisionSkills result

Lifecycle Semantics:
- NEW        → APPEARED
- TRACKED    → no event (steady-state silence)
- MOVED      → MOVED
- UPDATED    → UPDATED
- MISSING    → no DISAPPEARED yet (grace window preserved)
- REAPPEARED → REAPPEARED
- TERMINATED → DISAPPEARED
- STATE_CHANGED: emitted when prior & current ControlVisualState are both known and materially different
- CONTAINER_CHANGED: emitted when prior & current UIContainerType are both known and materially different
- WINDOW_CHANGED: emitted when top-level active window or process context changes
- UNCERTAIN: emitted when evidence conflicts or cannot be safely determined

Invariants:
1. Purely read-only in-memory processing: ZERO mouse/keyboard automation,
   screen capture, OCR, VLM inference, or OS process manipulation.
2. Zero persistent raw screenshots or pixel bytes in events, logs, or telemetry.
3. Strict session-scoped, ephemeral in-memory registry: No SQLite, JSON, disk storage,
   or long-term memory persistence.
4. Thread-safe operations via threading.RLock.
5. Deterministic deduplication without universal timer debounces.
6. Queries operate strictly on recorded events without mutating history.
"""

from __future__ import annotations

from collections import deque
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

from app.core.logger import get_logger
from app.vision.models import (
    ControlVisualState,
    Point,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    VisualElementTrack,
    VisualEventType,
    VisualTemporalEvent,
    VisualTemporalHistoryResult,
    VisualTrackStatus,
    VisualTrackingResult,
    WindowBounds,
)

logger = get_logger("VISION.TEMPORAL")

# --------------------------------------------------------------------------
# Default Configurable Temporal Constants
# --------------------------------------------------------------------------

DEFAULT_MAX_EVENTS: int = 100
DEFAULT_TTL_SECONDS: float = 300.0


def _sanitize_text(text: Optional[str]) -> str:
    """Sanitize string by stripping sensitive markers, passwords, and excessive length."""
    if not text:
        return ""
    cleaned = str(text).strip()
    if re.search(r"(?:password|passwd|secret|token|api[_-]?key|credit|pin)", cleaned, re.IGNORECASE):
        return "[REDACTED]"
    if len(cleaned) > 100:
        cleaned = cleaned[:100] + "..."
    return cleaned


# --------------------------------------------------------------------------
# VisualTemporalLog
# --------------------------------------------------------------------------


class VisualTemporalLog:
    """Thread-safe, bounded, session-scoped FIFO event log with TTL pruning.

    Enforces:
    - Maximum capacity eviction (bounded FIFO).
    - TTL-based time pruning on insertion, queries, or explicit maintenance.
    - Pure in-memory storage (no SQLite, no disk, no vector store).
    """

    def __init__(
        self,
        max_events: int = DEFAULT_MAX_EVENTS,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Initialize temporal log with bounded capacity and TTL."""
        self._max_events = max(1, int(max_events))
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._events: deque[VisualTemporalEvent] = deque()
        self._lock = threading.RLock()

    @property
    def max_events(self) -> int:
        """Maximum event capacity before FIFO eviction."""
        return self._max_events

    @property
    def ttl_seconds(self) -> float:
        """Event time-to-live in seconds."""
        return self._ttl_seconds

    def append(self, event: VisualTemporalEvent, current_time: Optional[float] = None) -> None:
        """Append an event to the log, pruning expired events and enforcing capacity."""
        if not isinstance(event, VisualTemporalEvent):
            raise TypeError(f"Expected VisualTemporalEvent, got {type(event).__name__}")

        with self._lock:
            # 1. Prune expired entries
            now = current_time if current_time is not None else time.time()
            self.prune(current_time=now)

            # 2. Append event
            self._events.append(event)

            # 3. FIFO capacity enforcement
            while len(self._events) > self._max_events:
                self._events.popleft()

    def prune(self, current_time: Optional[float] = None) -> int:
        """Prune events older than ttl_seconds. Returns number of pruned events."""
        now = current_time if current_time is not None else time.time()
        cutoff = now - self._ttl_seconds
        with self._lock:
            initial_count = len(self._events)
            while self._events and self._events[0].timestamp < cutoff:
                self._events.popleft()
            return initial_count - len(self._events)

    def get_events(
        self,
        current_time: Optional[float] = None,
    ) -> List[VisualTemporalEvent]:
        """Return all valid (non-expired) events currently in the log without mutating history."""
        with self._lock:
            now = current_time if current_time is not None else time.time()
            cutoff = now - self._ttl_seconds
            return [e for e in self._events if e.timestamp >= cutoff]

    def clear(self) -> None:
        """Clear all events from the log."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """Return count of stored events."""
        with self._lock:
            return len(self._events)


# --------------------------------------------------------------------------
# VisualTemporalEngine
# --------------------------------------------------------------------------


class VisualTemporalEngine:
    """Engine responsible for deriving and querying semantic visual events across observations.

    Consumes existing perception & tracking outputs (VisualTrackingResult, UIScene, ScreenObservation)
    and converts track lifecycle changes into discrete, immutable VisualTemporalEvents.
    """

    def __init__(
        self,
        max_events: int = DEFAULT_MAX_EVENTS,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Initialize the temporal engine."""
        self._log = VisualTemporalLog(max_events=max_events, ttl_seconds=ttl_seconds)
        self._lock = threading.RLock()

        # Ephemeral track state caches for detecting transitions
        self._track_states: Dict[str, ControlVisualState] = {}
        self._track_containers: Dict[str, UIContainerType] = {}
        self._track_bounds: Dict[str, WindowBounds] = {}
        self._last_observation_id: Optional[str] = None
        self._last_window_title: Optional[str] = None
        self._last_process_name: Optional[str] = None

        # Deduplication tracker: maps deduplication signature to last emission info
        # Signature format: (track_id or canonical_name, event_type) -> (transition_key, timestamp)
        self._dedup_history: Dict[Tuple[str, VisualEventType], Tuple[Any, float]] = {}

    @property
    def log(self) -> VisualTemporalLog:
        """Direct access to the underlying VisualTemporalLog."""
        return self._log

    def reset(self) -> None:
        """Reset all temporal history and transition state."""
        with self._lock:
            self._log.clear()
            self._track_states.clear()
            self._track_containers.clear()
            self._track_bounds.clear()
            self._last_observation_id = None
            self._last_window_title = None
            self._last_process_name = None
            self._dedup_history.clear()

    # -----------------------------------------------------------------------
    # Event Derivation
    # -----------------------------------------------------------------------

    def derive_events_from_tracking(
        self,
        tracking_result: VisualTrackingResult,
        scene: Optional[UIScene] = None,
        observation: Optional[ScreenObservation] = None,
        current_time: Optional[float] = None,
    ) -> List[VisualTemporalEvent]:
        """Derive discrete semantic events from structured perception and tracking evidence.

        Args:
            tracking_result: Current VisualTrackingResult output from VisualTrackingEngine.
            scene: Optional parsed UIScene with containers and affordances.
            observation: Optional current ScreenObservation context.
            current_time: Optional deterministic timestamp injection (defaults to now).

        Returns:
            List of freshly derived VisualTemporalEvents appended to the temporal log.
        """
        if tracking_result is None:
            return []

        now = current_time if current_time is not None else time.time()
        obs_id = tracking_result.observation_id or ""
        win_title = ""
        proc_name = ""

        # Security check: fail closed on sensitive, blocked, unauthorized, or invalid observations
        if observation is not None:
            is_blocked = (
                getattr(observation, "is_sensitive", False)
                or bool(observation.metadata.get("is_sensitive", False))
                or bool(observation.metadata.get("blocked", False))
                or (observation.authorization is not None and not observation.authorization.is_allowed)
                or not getattr(observation, "is_valid", True)
            )
            if is_blocked:
                logger.warning("VisualTemporalEngine: sensitive or blocked observation skipped for event derivation.")
                with self._lock:
                    # Sensitive window transition isolation: invalidate previous window context
                    # so subsequent safe observations cannot claim a direct transition from the pre-sensitive window
                    self._last_window_title = None
                    self._last_process_name = None
                return []
            win_title = _sanitize_text(str(observation.metadata.get("window_title") or ""))
            proc_name = _sanitize_text(str(observation.metadata.get("process_name") or "").lower())

        with self._lock:
            derived_events: List[VisualTemporalEvent] = []

            # --------------------------------------------------------------
            # 1. WINDOW_CHANGED Derivation
            # --------------------------------------------------------------
            if win_title or proc_name:
                has_prior_win = self._last_window_title is not None or self._last_process_name is not None
                if has_prior_win:
                    win_changed = (
                        (self._last_window_title is not None and win_title and win_title != self._last_window_title)
                        or (self._last_process_name is not None and proc_name and proc_name != self._last_process_name)
                    )
                    if win_changed:
                        evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                        summary = f"Active window switched to '{win_title or proc_name}'."
                        evt = VisualTemporalEvent(
                            event_id=evt_id,
                            event_type=VisualEventType.WINDOW_CHANGED,
                            timestamp=now,
                            observation_id=obs_id,
                            canonical_name=win_title or proc_name,
                            element_type=UIElementType.CONTAINER,
                            window_title=win_title,
                            process_name=proc_name,
                            confidence=1.0,
                            summary=summary,
                            metadata={
                                "prior_window_title": self._last_window_title,
                                "prior_process_name": self._last_process_name,
                                "current_window_title": win_title,
                                "current_process_name": proc_name,
                            },
                        )
                        if self._should_emit(evt, transition_key=(win_title, proc_name), now=now):
                            derived_events.append(evt)

                self._last_window_title = win_title
                self._last_process_name = proc_name

            # Build affordance / detected_state lookup from scene if available
            scene_states: Dict[str, ControlVisualState] = {}
            if scene is not None:
                for aff in getattr(scene, "affordances", ()):
                    if aff.element_name and aff.detected_state != ControlVisualState.UNCERTAIN:
                        scene_states[aff.element_name.strip().lower()] = aff.detected_state

            # --------------------------------------------------------------
            # 2. NEW Tracks -> APPEARED
            # --------------------------------------------------------------
            for track in tracking_result.new_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"'{sanitized_name}' appeared on screen."
                curr_state = self._extract_track_state(track, scene_states)
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.APPEARED,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    current_bounds=track.last_known_bounds,
                    current_state=curr_state,
                    current_container=track.container_type,
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"history_count": track.history_count},
                )
                if self._should_emit(evt, transition_key=track.track_id, now=now):
                    derived_events.append(evt)
                    self._update_track_cache(track, curr_state)

            # --------------------------------------------------------------
            # 3. MOVED Tracks -> MOVED
            # --------------------------------------------------------------
            for track in tracking_result.moved_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                prior_b = self._track_bounds.get(track.track_id)
                curr_b = track.last_known_bounds
                disp = None
                if prior_b and curr_b:
                    disp = Point(x=curr_b.center.x - prior_b.center.x, y=curr_b.center.y - prior_b.center.y)
                elif "window_displacement" in track.metadata:
                    wd = track.metadata["window_displacement"]
                    disp = Point(x=wd.get("dx", 0), y=wd.get("dy", 0))

                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"'{sanitized_name}' moved."
                curr_state = self._extract_track_state(track, scene_states)
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.MOVED,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    prior_bounds=prior_b,
                    current_bounds=curr_b,
                    displacement=disp,
                    prior_state=self._track_states.get(track.track_id),
                    current_state=curr_state,
                    prior_container=self._track_containers.get(track.track_id),
                    current_container=track.container_type,
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"displacement": disp.to_dict() if disp else None},
                )
                trans_key = (
                    curr_b.left if curr_b else 0,
                    curr_b.top if curr_b else 0,
                    disp.x if disp else 0,
                    disp.y if disp else 0,
                )
                if self._should_emit(evt, transition_key=trans_key, now=now):
                    derived_events.append(evt)
                    self._update_track_cache(track, curr_state)

            # --------------------------------------------------------------
            # 4. UPDATED Tracks -> UPDATED
            # --------------------------------------------------------------
            for track in tracking_result.updated_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"'{sanitized_name}' updated."
                curr_state = self._extract_track_state(track, scene_states)
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.UPDATED,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    prior_bounds=self._track_bounds.get(track.track_id),
                    current_bounds=track.last_known_bounds,
                    prior_state=self._track_states.get(track.track_id),
                    current_state=curr_state,
                    prior_container=self._track_containers.get(track.track_id),
                    current_container=track.container_type,
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"text_content": track.metadata.get("text_content")},
                )
                trans_key = track.metadata.get("text_content")
                if self._should_emit(evt, transition_key=trans_key, now=now):
                    derived_events.append(evt)
                    self._update_track_cache(track, curr_state)

            # --------------------------------------------------------------
            # 5. REAPPEARED Tracks -> REAPPEARED
            # --------------------------------------------------------------
            for track in tracking_result.reappeared_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"'{sanitized_name}' reappeared."
                curr_state = self._extract_track_state(track, scene_states)
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.REAPPEARED,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    prior_bounds=self._track_bounds.get(track.track_id),
                    current_bounds=track.last_known_bounds,
                    prior_state=self._track_states.get(track.track_id),
                    current_state=curr_state,
                    prior_container=self._track_containers.get(track.track_id),
                    current_container=track.container_type,
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"missing_count": track.missing_count},
                )
                if self._should_emit(evt, transition_key=track.track_id, now=now):
                    derived_events.append(evt)
                    self._update_track_cache(track, curr_state)

            # --------------------------------------------------------------
            # 6. TERMINATED Tracks -> DISAPPEARED
            # (Note: MISSING tracks produce NO DISAPPEARED event!)
            # --------------------------------------------------------------
            for track in tracking_result.terminated_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"'{sanitized_name}' disappeared."
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.DISAPPEARED,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    prior_bounds=track.last_known_bounds or self._track_bounds.get(track.track_id),
                    prior_state=self._track_states.get(track.track_id),
                    prior_container=self._track_containers.get(track.track_id),
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"missing_count": track.missing_count},
                )
                if self._should_emit(evt, transition_key=track.track_id, now=now):
                    derived_events.append(evt)

            # --------------------------------------------------------------
            # 7. UNCERTAIN Tracks -> UNCERTAIN
            # --------------------------------------------------------------
            for track in tracking_result.uncertain_tracks:
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                summary = f"Ambiguous visual state for '{sanitized_name}'."
                evt = VisualTemporalEvent(
                    event_id=evt_id,
                    event_type=VisualEventType.UNCERTAIN,
                    timestamp=now,
                    observation_id=obs_id,
                    canonical_name=sanitized_name,
                    element_type=track.element_type,
                    track_id=track.track_id,
                    current_bounds=track.last_known_bounds,
                    window_title=track.window_title or win_title,
                    process_name=track.process_name or proc_name,
                    confidence=track.confidence,
                    summary=summary,
                    metadata={"reason": track.metadata.get("reason", "ambiguous_match")},
                )
                if self._should_emit(evt, transition_key=track.metadata.get("reason"), now=now):
                    derived_events.append(evt)

            # --------------------------------------------------------------
            # 8. Active Tracks: Check for STATE_CHANGED and CONTAINER_CHANGED
            # --------------------------------------------------------------
            for track in tracking_result.active_tracks:
                tid = track.track_id
                sanitized_name = _sanitize_text(track.canonical_name)
                if not sanitized_name:
                    continue

                curr_state = self._extract_track_state(track, scene_states)
                prior_state = self._track_states.get(tid)

                # STATE_CHANGED: Emit only when prior & current are both available and materially different
                if (
                    prior_state is not None
                    and curr_state is not None
                    and prior_state != ControlVisualState.UNCERTAIN
                    and curr_state != ControlVisualState.UNCERTAIN
                    and prior_state != curr_state
                ):
                    evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                    p_val = prior_state.value if hasattr(prior_state, "value") else str(prior_state)
                    c_val = curr_state.value if hasattr(curr_state, "value") else str(curr_state)
                    summary = f"'{sanitized_name}' state changed from {p_val} to {c_val}."
                    evt = VisualTemporalEvent(
                        event_id=evt_id,
                        event_type=VisualEventType.STATE_CHANGED,
                        timestamp=now,
                        observation_id=obs_id,
                        canonical_name=sanitized_name,
                        element_type=track.element_type,
                        track_id=tid,
                        current_bounds=track.last_known_bounds,
                        prior_state=prior_state,
                        current_state=curr_state,
                        prior_container=self._track_containers.get(tid),
                        current_container=track.container_type,
                        window_title=track.window_title or win_title,
                        process_name=track.process_name or proc_name,
                        confidence=track.confidence,
                        summary=summary,
                        metadata={"prior_state": p_val, "current_state": c_val},
                    )
                    trans_key = (p_val, c_val)
                    if self._should_emit(evt, transition_key=trans_key, now=now):
                        derived_events.append(evt)
                        self._track_states[tid] = curr_state

                # CONTAINER_CHANGED: Emit only when prior & current container are both available and materially different
                prior_cont = self._track_containers.get(tid)
                curr_cont = track.container_type
                if (
                    prior_cont is not None
                    and curr_cont is not None
                    and prior_cont != curr_cont
                ):
                    evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                    p_cval = prior_cont.value if hasattr(prior_cont, "value") else str(prior_cont)
                    c_cval = curr_cont.value if hasattr(curr_cont, "value") else str(curr_cont)
                    summary = f"'{sanitized_name}' moved from {p_cval} to {c_cval}."
                    evt = VisualTemporalEvent(
                        event_id=evt_id,
                        event_type=VisualEventType.CONTAINER_CHANGED,
                        timestamp=now,
                        observation_id=obs_id,
                        canonical_name=sanitized_name,
                        element_type=track.element_type,
                        track_id=tid,
                        current_bounds=track.last_known_bounds,
                        prior_state=self._track_states.get(tid),
                        current_state=curr_state,
                        prior_container=prior_cont,
                        current_container=curr_cont,
                        window_title=track.window_title or win_title,
                        process_name=track.process_name or proc_name,
                        confidence=track.confidence,
                        summary=summary,
                        metadata={"prior_container": p_cval, "current_container": c_cval},
                    )
                    trans_key = (p_cval, c_cval)
                    if self._should_emit(evt, transition_key=trans_key, now=now):
                        derived_events.append(evt)
                        self._track_containers[tid] = curr_cont

                # Update caches for steady state
                self._update_track_cache(track, curr_state)

            # --------------------------------------------------------------
            # Append derived events to log
            # --------------------------------------------------------------
            for evt in derived_events:
                self._log.append(evt, current_time=now)

            self._last_observation_id = obs_id
            return derived_events

    # -----------------------------------------------------------------------
    # Deduplication & State Helpers
    # -----------------------------------------------------------------------

    def _should_emit(
        self,
        event: VisualTemporalEvent,
        transition_key: Any,
        now: float,
    ) -> bool:
        """Deterministic deduplication check.

        Suppresses duplicate emissions if the identical semantic transition with no materially
        new evidence was already emitted recently. Legitimate rapid different transitions
        (different transition_key or different event_type) remain visible and are emitted.
        """
        entity_key = event.track_id or event.canonical_name
        sig = (entity_key, event.event_type)

        last_info = self._dedup_history.get(sig)
        if last_info is not None:
            last_trans_key, last_ts = last_info
            # If transition key is identical and within TTL, suppress duplicate
            if last_trans_key == transition_key:
                return False

        # Record this emission
        self._dedup_history[sig] = (transition_key, now)
        return True

    def _extract_track_state(
        self,
        track: VisualElementTrack,
        scene_states: Dict[str, ControlVisualState],
    ) -> Optional[ControlVisualState]:
        """Extract ControlVisualState from track metadata, scene affordance, or direct attribute."""
        # Check track metadata
        raw = track.metadata.get("detected_state") or track.metadata.get("state")
        if isinstance(raw, ControlVisualState):
            return raw
        if isinstance(raw, str):
            try:
                return ControlVisualState(raw.lower())
            except ValueError:
                pass

        # Check scene lookup by canonical name
        name_key = track.canonical_name.strip().lower()
        if name_key in scene_states:
            return scene_states[name_key]

        return None

    def _update_track_cache(
        self,
        track: VisualElementTrack,
        curr_state: Optional[ControlVisualState],
    ) -> None:
        """Update cached bounds, container, and state for a track."""
        tid = track.track_id
        if track.last_known_bounds:
            self._track_bounds[tid] = track.last_known_bounds
        if track.container_type:
            self._track_containers[tid] = track.container_type
        if curr_state:
            self._track_states[tid] = curr_state

    # -----------------------------------------------------------------------
    # Temporal Queries
    # -----------------------------------------------------------------------

    def get_recent_events(
        self,
        limit: int = 10,
        event_type: Optional[Union[VisualEventType, str]] = None,
        window_filter: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> VisualTemporalHistoryResult:
        """Retrieve recent events matching optional type and window filters.

        Args:
            limit: Maximum events to return (default 10).
            event_type: Optional VisualEventType filter.
            window_filter: Optional window title or process name substring filter.
            current_time: Optional deterministic timestamp for TTL check.

        Returns:
            VisualTemporalHistoryResult containing matched events and voice-safe summary.
        """
        req_limit = max(1, int(limit))
        filter_type = None
        if event_type:
            filter_type = event_type.value if isinstance(event_type, VisualEventType) else str(event_type).upper()

        win_norm = window_filter.strip().lower() if window_filter else None

        with self._lock:
            all_events = self._log.get_events(current_time=current_time)

            matched: List[VisualTemporalEvent] = []
            for evt in reversed(all_events):  # Most recent first
                if filter_type:
                    evt_type_str = evt.event_type.value if isinstance(evt.event_type, VisualEventType) else str(evt.event_type)
                    if evt_type_str.upper() != filter_type:
                        continue
                if win_norm:
                    w_title = (evt.window_title or "").lower()
                    p_name = (evt.process_name or "").lower()
                    if win_norm not in w_title and win_norm not in p_name:
                        continue

                matched.append(evt)
                if len(matched) >= req_limit:
                    break

            summary = self._generate_history_summary(matched, filter_label=filter_type)
            return VisualTemporalHistoryResult(
                query_id=f"qry_{uuid.uuid4().hex[:8]}",
                events=tuple(matched),
                total_count=len(matched),
                window_title=window_filter,
                summary=summary,
                metadata={"limit": req_limit, "filter_type": filter_type},
            )

    def get_element_history(
        self,
        target: str,
        track_id: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> VisualTemporalHistoryResult:
        """Retrieve full event trajectory for a specific named element or track_id.

        Args:
            target: Canonical label or text of the target element.
            track_id: Optional specific VisualElementTrack track_id.
            current_time: Optional deterministic timestamp for TTL check.

        Returns:
            VisualTemporalHistoryResult containing chronological event history for the element.
        """
        t_norm = target.strip().lower() if target else ""
        tid_norm = track_id.strip() if track_id else None

        with self._lock:
            all_events = self._log.get_events(current_time=current_time)

            matched: List[VisualTemporalEvent] = []
            for evt in all_events:
                if tid_norm and evt.track_id != tid_norm:
                    continue
                if t_norm and t_norm not in evt.canonical_name.lower():
                    continue
                matched.append(evt)

            # Summarize trajectory
            if not matched:
                summary = f"No visual events recorded for '{target}'."
            elif len(matched) == 1:
                summary = matched[0].summary
            else:
                event_types = [e.event_type.value.lower() for e in matched]
                summary = f"'{target}' underwent {len(matched)} events: {', '.join(event_types)}."

            return VisualTemporalHistoryResult(
                query_id=f"qry_{uuid.uuid4().hex[:8]}",
                events=tuple(matched),
                total_count=len(matched),
                summary=summary,
                metadata={"target": target, "track_id": track_id},
            )

    def get_disappeared_elements(
        self,
        window_filter: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> VisualTemporalHistoryResult:
        """Retrieve elements that have confirmed disappeared (TERMINATED) in recent history."""
        return self.get_recent_events(
            limit=50,
            event_type=VisualEventType.DISAPPEARED,
            window_filter=window_filter,
            current_time=current_time,
        )

    def get_state_changes(
        self,
        target: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> VisualTemporalHistoryResult:
        """Retrieve state change events, optionally filtered by target element name."""
        res = self.get_recent_events(
            limit=50,
            event_type=VisualEventType.STATE_CHANGED,
            current_time=current_time,
        )
        if not target or not target.strip():
            return res

        t_norm = target.strip().lower()
        filtered = tuple(e for e in res.events if t_norm in e.canonical_name.lower())
        summary = self._generate_history_summary(list(filtered), filter_label="STATE_CHANGED", target=target)
        return VisualTemporalHistoryResult(
            query_id=f"qry_{uuid.uuid4().hex[:8]}",
            events=filtered,
            total_count=len(filtered),
            summary=summary,
            metadata={"target": target},
        )

    # -----------------------------------------------------------------------
    # Summary Generation
    # -----------------------------------------------------------------------

    def _generate_history_summary(
        self,
        events: List[VisualTemporalEvent],
        filter_label: Optional[str] = None,
        target: Optional[str] = None,
    ) -> str:
        """Construct voice-safe natural-language summary of temporal history."""
        if not events:
            if target:
                return f"No recent visual changes detected for '{target}'."
            return "No recent visual events detected."

        if len(events) == 1:
            return events[0].summary or "One recent visual event detected."

        # Multiple events
        if target:
            return f"{len(events)} visual events detected for '{target}'."

        if filter_label == "STATE_CHANGED":
            return f"{len(events)} control state changes detected."

        if filter_label == "DISAPPEARED":
            names = ", ".join(f"'{e.canonical_name}'" for e in events[:2])
            if len(events) > 2:
                return f"{len(events)} elements disappeared, including {names}."
            return f"{names} disappeared."

        return f"{len(events)} recent visual changes detected."


__all__ = [
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_TTL_SECONDS",
    "VisualTemporalEngine",
    "VisualTemporalLog",
]
