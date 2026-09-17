"""Cross-Observation Visual Identity and Tracking Engine for J.A.R.V.I.S (Phase 27.14).

Provides deterministic-first, privacy-safe temporal tracking of visual UI elements
across sequential observations (T0, T1, ..., Tn) to answer questions such as:
- "Did the button move?"
- "Where did the submit button move?"
- "Is the element still there?"
- "Is this the same control as before?"
- "What elements are currently being tracked?"

Safety and Architectural Invariants Enforced:
1. Purely read-only visual perception and tracking: ZERO mouse/keyboard automation,
   clicks, typing, focus changes, process mutation, or desktop state manipulation.
2. Zero persistent raw screenshots or pixel bytes in tracks, results, logs, or telemetry.
3. Strict session-scoped, ephemeral in-memory registry: No SQLite, JSON, disk storage,
   or long-term memory persistence.
4. Deterministic-first matching: Functions 100% on CPU without requiring VLM inference.
5. Conservative identity association: Conflicting or ambiguous evidence produces UNCERTAIN;
   no arbitrary or forced identity assignment when evidence is insufficient.
6. Window translation compensation: Window movement does NOT classify child controls as MOVED
   unless they moved relative to their containing window.
7. Window/process isolation: Tracks never cross-associate elements between different
   applications or unrelated window lifecycles.
8. Bounded registry: Enforces max_tracks and max_missing_frames with automatic eviction.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

from app.core.logger import get_logger
from app.vision.models import (
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    VisualElementTrack,
    VisualTrackStatus,
    VisualTrackingResult,
    WindowBounds,
)

logger = get_logger("VISION.TRACKING")

# --------------------------------------------------------------------------
# Default Configurable Tracking Thresholds & Constants
# --------------------------------------------------------------------------

DEFAULT_MAX_TRACKS: int = 100
DEFAULT_MAX_MISSING_FRAMES: int = 3
DEFAULT_MIN_MATCH_SCORE: float = 0.45
DEFAULT_AMBIGUITY_THRESHOLD: float = 0.05
DEFAULT_MOVE_DISTANCE_THRESHOLD: float = 15.0
DEFAULT_RESIZE_RATIO_THRESHOLD: float = 0.20


def _compute_iou(b1: WindowBounds, b2: WindowBounds) -> float:
    """Compute spatial Intersection-over-Union between two WindowBounds."""
    if b1.is_empty or b2.is_empty:
        return 0.0

    inter_left = max(b1.left, b2.left)
    inter_top = max(b1.top, b2.top)
    inter_right = min(b1.right, b2.right)
    inter_bottom = min(b1.bottom, b2.bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    union_area = b1.area + b2.area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _compute_text_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Compute normalized text similarity ratio between two strings."""
    t1 = (s1 or "").strip().lower()
    t2 = (s2 or "").strip().lower()
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0
    return SequenceMatcher(None, t1, t2).ratio()


def _sanitize_text(text: Optional[str], is_password: bool = False) -> str:
    """Sanitize label or text, redacting potential passwords or credential values."""
    if not text:
        return ""
    if is_password or "password" in text.lower() or "secret" in text.lower():
        # Mask if it looks like an actual secret rather than a label
        if text.lower() not in ("password", "password:", "enter password", "secret", "confirm password"):
            return "[REDACTED]"
    return text.strip()


class VisualTrackingEngine:
    """Production Cross-Observation Visual Identity and Tracking Engine for J.A.R.V.I.S."""

    def __init__(
        self,
        *,
        max_tracks: int = DEFAULT_MAX_TRACKS,
        max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
        min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
        ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
        move_distance_threshold: float = DEFAULT_MOVE_DISTANCE_THRESHOLD,
        resize_ratio_threshold: float = DEFAULT_RESIZE_RATIO_THRESHOLD,
        weights: Optional[Dict[str, float]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the VisualTrackingEngine with configurable thresholds."""
        self._max_tracks = max(10, int(max_tracks))
        self._max_missing_frames = max(1, int(max_missing_frames))
        self._min_match_score = float(min_match_score)
        self._ambiguity_threshold = float(ambiguity_threshold)
        self._move_distance_threshold = float(move_distance_threshold)
        self._resize_ratio_threshold = float(resize_ratio_threshold)
        self._logger = logger_instance or logger

        # Configurable scoring weights
        # Default: 0.35 text, 0.30 IOU, 0.20 centroid distance, 0.10 type, 0.05 container
        w = weights or {}
        self._w_text = float(w.get("text", 0.35))
        self._w_iou = float(w.get("iou", 0.30))
        self._w_centroid = float(w.get("centroid", 0.20))
        self._w_type = float(w.get("type", 0.10))
        self._w_container = float(w.get("container", 0.05))

        # In-memory session registry: track_id -> VisualElementTrack
        self._tracks: Dict[str, VisualElementTrack] = {}
        # Track previous active window bounds per process/window context for translation compensation
        self._last_window_bounds: Dict[str, WindowBounds] = {}
        self._last_observation_id: str = ""
        self._lock = threading.RLock()

    # -----------------------------------------------------------------------
    # Configuration Properties
    # -----------------------------------------------------------------------

    @property
    def max_tracks(self) -> int:
        """Maximum tracks retained in memory registry."""
        return self._max_tracks

    @property
    def max_missing_frames(self) -> int:
        """Consecutive missing frames before a track transitions to TERMINATED."""
        return self._max_missing_frames

    @property
    def min_match_score(self) -> float:
        """Minimum composite matching score for positive element association."""
        return self._min_match_score

    @property
    def ambiguity_threshold(self) -> float:
        """Difference threshold between top candidates below which association is UNCERTAIN."""
        return self._ambiguity_threshold

    @property
    def move_distance_threshold(self) -> float:
        """Pixel translation distance relative to window before classifying as MOVED."""
        return self._move_distance_threshold

    # -----------------------------------------------------------------------
    # Registry Management Interface
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the entire in-memory tracking registry and cached window bounds."""
        with self._lock:
            self._tracks.clear()
            self._last_window_bounds.clear()
            self._last_observation_id = ""
            self._logger.debug("Visual tracking registry reset.")

    def get_track(self, track_id: str) -> Optional[VisualElementTrack]:
        """Retrieve a specific track by its track_id, if present."""
        with self._lock:
            return self._tracks.get(track_id)

    def get_active_tracks(self, target_filter: Optional[str] = None) -> List[VisualElementTrack]:
        """Return all active (non-terminated) tracks, optionally filtered by name."""
        with self._lock:
            active = [
                t for t in self._tracks.values()
                if t.status != VisualTrackStatus.TERMINATED
            ]
            if target_filter and target_filter.strip():
                tf = target_filter.strip().lower()
                active = [
                    t for t in active
                    if tf in t.canonical_name.lower() or tf in str(t.metadata.get("text_content", "")).lower()
                ]
            return active

    def prune_tracks(self, max_tracks: Optional[int] = None) -> int:
        """Prune tracks if registry exceeds capacity. Returns count of pruned tracks."""
        limit = max_tracks or self._max_tracks
        with self._lock:
            if len(self._tracks) <= limit:
                return 0

            # Prioritized eviction:
            # 1. TERMINATED tracks
            # 2. Oldest MISSING tracks (highest missing_count)
            # 3. Oldest by history_count or last_observation_id
            evicted = 0
            to_remove: List[str] = []

            # Phase 1: Evict TERMINATED
            for tid, t in list(self._tracks.items()):
                if t.status == VisualTrackStatus.TERMINATED:
                    to_remove.append(tid)
                    if len(self._tracks) - len(to_remove) <= limit:
                        break

            # Phase 2: Evict oldest MISSING if still over capacity
            if len(self._tracks) - len(to_remove) > limit:
                missing = [
                    (tid, t) for tid, t in self._tracks.items()
                    if tid not in to_remove and t.status == VisualTrackStatus.MISSING
                ]
                missing.sort(key=lambda x: x[1].missing_count, reverse=True)
                for tid, _ in missing:
                    to_remove.append(tid)
                    if len(self._tracks) - len(to_remove) <= limit:
                        break

            # Phase 3: Evict least recently seen
            if len(self._tracks) - len(to_remove) > limit:
                remaining = [
                    (tid, t) for tid, t in self._tracks.items()
                    if tid not in to_remove
                ]
                remaining.sort(key=lambda x: x[1].history_count)
                for tid, _ in remaining:
                    to_remove.append(tid)
                    if len(self._tracks) - len(to_remove) <= limit:
                        break

            for tid in to_remove:
                del self._tracks[tid]
                evicted += 1

            return evicted

    # -----------------------------------------------------------------------
    # Main Tracking Interface
    # -----------------------------------------------------------------------

    def track_observation(
        self,
        observation: ScreenObservation,
        scene: Optional[UIScene] = None,
        target_filter: Optional[str] = None,
    ) -> VisualTrackingResult:
        """Associate elements in the given observation with existing tracks and update registry."""
        start_time = time.time()
        tracking_id = str(uuid.uuid4())

        # Validation: Check observation
        if observation is None or observation.is_empty or not observation.is_valid:
            return self._handle_invalid_observation(observation, tracking_id)

        obs_id = observation.observation_id
        win_title = str(observation.metadata.get("window_title") or "").strip()
        proc_name = str(observation.metadata.get("process_name") or "").strip().lower()
        curr_win_bounds = observation.bounds

        # Window Context Identifier (e.g. process_name + title or HWND)
        hwnd = observation.metadata.get("hwnd")
        win_key = f"{proc_name}:{hwnd or win_title}"

        with self._lock:
            # --------------------------------------------------------------
            # Step 1: Compute Window Translation Compensation
            # --------------------------------------------------------------
            prev_win_bounds = self._last_window_bounds.get(win_key)
            win_dx = 0
            win_dy = 0
            if prev_win_bounds and curr_win_bounds and not prev_win_bounds.is_empty and not curr_win_bounds.is_empty:
                win_dx = curr_win_bounds.left - prev_win_bounds.left
                win_dy = curr_win_bounds.top - prev_win_bounds.top

            if curr_win_bounds and not curr_win_bounds.is_empty:
                self._last_window_bounds[win_key] = curr_win_bounds

            # --------------------------------------------------------------
            # Step 2: Extract Candidate UIElements from observation/scene
            # --------------------------------------------------------------
            curr_elements, container_map = self._extract_elements_and_containers(
                observation, scene
            )

            # Apply target filter if specified
            if target_filter and target_filter.strip():
                tf_clean = target_filter.strip().lower()
                curr_elements = [
                    el for el in curr_elements
                    if tf_clean in el.name.lower() or tf_clean in str(el.text_content or "").lower()
                ]

            # --------------------------------------------------------------
            # Step 3: Identify Candidate Active Tracks in the same window context
            # --------------------------------------------------------------
            # Only match tracks that belong to this window/process context
            candidate_tracks: List[VisualElementTrack] = []
            other_tracks: List[VisualElementTrack] = []

            for t in self._tracks.values():
                if t.status == VisualTrackStatus.TERMINATED:
                    continue

                # Process/window isolation:
                # If track has a process_name and current observation has a process_name, they must match
                is_same_context = True
                if t.process_name and proc_name and t.process_name != proc_name:
                    is_same_context = False
                elif t.window_title and win_title and t.window_title != win_title and not proc_name:
                    is_same_context = False

                if is_same_context:
                    candidate_tracks.append(t)
                else:
                    other_tracks.append(t)

            # --------------------------------------------------------------
            # Step 4: Deterministic Multi-Factor Association
            # --------------------------------------------------------------
            (
                matched_pairs,
                unmatched_track_ids,
                unmatched_element_indices,
                uncertain_matches,
            ) = self._associate_elements_with_tracks(
                candidate_tracks=candidate_tracks,
                current_elements=curr_elements,
                container_map=container_map,
                win_dx=win_dx,
                win_dy=win_dy,
            )

            # --------------------------------------------------------------
            # Step 5: Update Lifecycle States & Create Tracks
            # --------------------------------------------------------------
            new_tracks: List[VisualElementTrack] = []
            moved_tracks: List[VisualElementTrack] = []
            updated_tracks: List[VisualElementTrack] = []
            missing_tracks: List[VisualElementTrack] = []
            reappeared_tracks: List[VisualElementTrack] = []
            uncertain_tracks: List[VisualElementTrack] = []
            terminated_tracks: List[VisualElementTrack] = []
            tracked_tracks: List[VisualElementTrack] = []

            # 5a. Process Matched Pairs
            for track, el, score, is_rel_moved, is_updated in matched_pairs:
                prev_status = track.status
                if prev_status == VisualTrackStatus.MISSING:
                    new_status = VisualTrackStatus.REAPPEARED
                elif is_rel_moved:
                    new_status = VisualTrackStatus.MOVED
                elif is_updated:
                    new_status = VisualTrackStatus.UPDATED
                else:
                    new_status = VisualTrackStatus.TRACKED

                c_type = container_map.get(id(el)) or track.container_type

                updated_track = VisualElementTrack(
                    track_id=track.track_id,
                    element_type=el.element_type if el.element_type != UIElementType.UNKNOWN else track.element_type,
                    canonical_name=track.canonical_name or el.name,
                    first_observation_id=track.first_observation_id,
                    last_observation_id=obs_id,
                    last_known_bounds=el.bounds,
                    last_known_center=el.center,
                    confidence=max(0.0, min(1.0, float(score))),
                    status=new_status,
                    history_count=track.history_count + 1,
                    missing_count=0,
                    container_type=c_type,
                    window_title=win_title or track.window_title,
                    process_name=proc_name or track.process_name,
                    metadata={
                        "last_matched_score": round(score, 3),
                        "text_content": _sanitize_text(el.text_content or el.name),
                        "window_displacement": {"dx": win_dx, "dy": win_dy},
                    },
                )
                self._tracks[track.track_id] = updated_track

                if new_status == VisualTrackStatus.REAPPEARED:
                    reappeared_tracks.append(updated_track)
                elif new_status == VisualTrackStatus.MOVED:
                    moved_tracks.append(updated_track)
                elif new_status == VisualTrackStatus.UPDATED:
                    updated_tracks.append(updated_track)
                else:
                    tracked_tracks.append(updated_track)

            # 5b. Process Uncertain Matches
            for track, el, conf in uncertain_matches:
                unc_track = VisualElementTrack(
                    track_id=track.track_id,
                    element_type=track.element_type,
                    canonical_name=track.canonical_name,
                    first_observation_id=track.first_observation_id,
                    last_observation_id=obs_id,
                    last_known_bounds=track.last_known_bounds,
                    last_known_center=track.last_known_center,
                    confidence=max(0.0, min(1.0, float(conf))),
                    status=VisualTrackStatus.UNCERTAIN,
                    history_count=track.history_count,
                    missing_count=track.missing_count,
                    container_type=track.container_type,
                    window_title=win_title or track.window_title,
                    process_name=proc_name or track.process_name,
                    metadata={
                        "reason": "ambiguous_match_candidates",
                        "text_content": _sanitize_text(el.text_content or el.name),
                    },
                )
                self._tracks[track.track_id] = unc_track
                uncertain_tracks.append(unc_track)

            # 5c. Process Unmatched Existing Tracks -> MISSING or TERMINATED
            for tid in unmatched_track_ids:
                old_track = self._tracks[tid]
                new_missing_count = old_track.missing_count + 1
                if new_missing_count > self._max_missing_frames:
                    term_track = VisualElementTrack(
                        track_id=old_track.track_id,
                        element_type=old_track.element_type,
                        canonical_name=old_track.canonical_name,
                        first_observation_id=old_track.first_observation_id,
                        last_observation_id=old_track.last_observation_id,
                        last_known_bounds=old_track.last_known_bounds,
                        last_known_center=old_track.last_known_center,
                        confidence=max(0.0, old_track.confidence - 0.2),
                        status=VisualTrackStatus.TERMINATED,
                        history_count=old_track.history_count,
                        missing_count=new_missing_count,
                        container_type=old_track.container_type,
                        window_title=old_track.window_title,
                        process_name=old_track.process_name,
                        metadata=dict(old_track.metadata),
                    )
                    self._tracks[tid] = term_track
                    terminated_tracks.append(term_track)
                else:
                    miss_track = VisualElementTrack(
                        track_id=old_track.track_id,
                        element_type=old_track.element_type,
                        canonical_name=old_track.canonical_name,
                        first_observation_id=old_track.first_observation_id,
                        last_observation_id=old_track.last_observation_id,
                        last_known_bounds=old_track.last_known_bounds,
                        last_known_center=old_track.last_known_center,
                        confidence=max(0.0, old_track.confidence - 0.1),
                        status=VisualTrackStatus.MISSING,
                        history_count=old_track.history_count,
                        missing_count=new_missing_count,
                        container_type=old_track.container_type,
                        window_title=old_track.window_title,
                        process_name=old_track.process_name,
                        metadata=dict(old_track.metadata),
                    )
                    self._tracks[tid] = miss_track
                    missing_tracks.append(miss_track)

            # 5d. Process Unmatched Current Elements -> NEW Tracks
            for idx in unmatched_element_indices:
                el = curr_elements[idx]
                c_type = container_map.get(id(el))
                new_tid = f"trk_{uuid.uuid4().hex[:8]}"
                sanitized_text = _sanitize_text(el.text_content or el.name)
                fresh_track = VisualElementTrack(
                    track_id=new_tid,
                    element_type=el.element_type,
                    canonical_name=el.name,
                    first_observation_id=obs_id,
                    last_observation_id=obs_id,
                    last_known_bounds=el.bounds,
                    last_known_center=el.center,
                    confidence=float(el.confidence),
                    status=VisualTrackStatus.NEW,
                    history_count=1,
                    missing_count=0,
                    container_type=c_type,
                    window_title=win_title,
                    process_name=proc_name,
                    metadata={
                        "text_content": sanitized_text,
                    },
                )
                self._tracks[new_tid] = fresh_track
                new_tracks.append(fresh_track)

            # --------------------------------------------------------------
            # Step 6: Prune Registry if Exceeding Bound
            # --------------------------------------------------------------
            pruned_count = self.prune_tracks(self._max_tracks)

            # Gather active non-terminated tracks
            active_tracks = tuple(
                t for t in self._tracks.values()
                if t.status != VisualTrackStatus.TERMINATED
            )

            # Step 7: Natural-Language Summary
            summary = self._generate_summary(
                new_count=len(new_tracks),
                moved_count=len(moved_tracks),
                updated_count=len(updated_tracks),
                missing_count=len(missing_tracks),
                reappeared_count=len(reappeared_tracks),
                uncertain_count=len(uncertain_tracks),
                active_count=len(active_tracks),
                target_filter=target_filter,
                moved_tracks=moved_tracks,
                missing_tracks=missing_tracks,
                reappeared_tracks=reappeared_tracks,
            )

            duration_ms = (time.time() - start_time) * 1000.0

            result = VisualTrackingResult(
                tracking_id=tracking_id,
                observation_id=obs_id,
                active_tracks=active_tracks,
                new_tracks=tuple(new_tracks),
                moved_tracks=tuple(moved_tracks),
                updated_tracks=tuple(updated_tracks),
                missing_tracks=tuple(missing_tracks),
                reappeared_tracks=tuple(reappeared_tracks),
                uncertain_tracks=tuple(uncertain_tracks),
                terminated_tracks=tuple(terminated_tracks),
                confidence=1.0 if not uncertain_tracks else 0.75,
                summary=summary,
                metadata={
                    "duration_ms": round(duration_ms, 2),
                    "window_dx": win_dx,
                    "window_dy": win_dy,
                    "pruned_count": pruned_count,
                    "total_registered_tracks": len(self._tracks),
                },
            )

            self._last_observation_id = obs_id
            return result

    # -----------------------------------------------------------------------
    # Multi-Factor Association Implementation
    # -----------------------------------------------------------------------

    def _associate_elements_with_tracks(
        self,
        candidate_tracks: List[VisualElementTrack],
        current_elements: List[UIElement],
        container_map: Dict[int, Optional[UIContainerType]],
        win_dx: int,
        win_dy: int,
    ) -> Tuple[
        List[Tuple[VisualElementTrack, UIElement, float, bool, bool]],
        Set[str],
        Set[int],
        List[Tuple[VisualElementTrack, UIElement, float]],
    ]:
        """Perform deterministic one-to-one bipartite association.

        Returns:
            (matched_pairs, unmatched_track_ids, unmatched_element_indices, uncertain_matches)
        """
        matched_pairs: List[Tuple[VisualElementTrack, UIElement, float, bool, bool]] = []
        uncertain_matches: List[Tuple[VisualElementTrack, UIElement, float]] = []

        if not candidate_tracks and not current_elements:
            return matched_pairs, set(), set(), uncertain_matches

        if not candidate_tracks:
            return matched_pairs, set(), set(range(len(current_elements))), uncertain_matches

        if not current_elements:
            return (
                matched_pairs,
                {t.track_id for t in candidate_tracks},
                set(),
                uncertain_matches,
            )

        # Build candidate score matrix
        # For each candidate track: compute translated prior bounds
        # translated bounds = prior bounds shifted by (win_dx, win_dy)
        scored_candidates: List[Tuple[int, int, float, float, float, float, bool, bool]] = []
        track_candidates: Dict[int, List[Tuple[int, float]]] = {
            t_idx: [] for t_idx in range(len(candidate_tracks))
        }

        for t_idx, track in enumerate(candidate_tracks):
            trans_bounds = WindowBounds(
                left=track.last_known_bounds.left + win_dx,
                top=track.last_known_bounds.top + win_dy,
                right=track.last_known_bounds.right + win_dx,
                bottom=track.last_known_bounds.bottom + win_dy,
            )
            trans_center = trans_bounds.center

            for el_idx, el in enumerate(current_elements):
                # 1. Text Similarity (Strong signal)
                text_sim = _compute_text_similarity(
                    track.canonical_name,
                    el.name or el.text_content,
                )

                # 2. Translated IOU (Moderate signal)
                iou = _compute_iou(trans_bounds, el.bounds)

                # 3. Translated Centroid Proximity (Moderate signal)
                dist = math.hypot(el.center.x - trans_center.x, el.center.y - trans_center.y)
                centroid_sim = max(0.0, 1.0 - (dist / 500.0))

                # 4. Type Compatibility (Strong signal)
                if track.element_type == el.element_type:
                    type_score = 1.0
                elif (
                    track.element_type == UIElementType.UNKNOWN
                    or el.element_type == UIElementType.UNKNOWN
                ):
                    type_score = 0.5
                else:
                    type_score = 0.0

                # 5. Container Match (Moderate signal)
                el_container = container_map.get(id(el))
                if track.container_type is not None and el_container is not None:
                    container_score = 1.0 if track.container_type == el_container else 0.0
                else:
                    container_score = 0.5

                # Composite score
                composite_score = (
                    (self._w_text * text_sim)
                    + (self._w_iou * iou)
                    + (self._w_centroid * centroid_sim)
                    + (self._w_type * type_score)
                    + (self._w_container * container_score)
                )

                # Relative movement check (internal displacement beyond window translation)
                topleft_dist = math.hypot(el.bounds.left - trans_bounds.left, el.bounds.top - trans_bounds.top)
                is_rel_moved = dist > self._move_distance_threshold and topleft_dist > self._move_distance_threshold

                # Property update check (e.g. text slightly altered or in-place resize)
                is_updated = (
                    0.5 <= text_sim < 1.0
                    or (
                        track.last_known_bounds.width > 0
                        and abs(el.bounds.width - track.last_known_bounds.width) / track.last_known_bounds.width > self._resize_ratio_threshold
                    )
                    or (
                        track.last_known_bounds.height > 0
                        and abs(el.bounds.height - track.last_known_bounds.height) / track.last_known_bounds.height > self._resize_ratio_threshold
                    )
                )

                if composite_score >= self._min_match_score:
                    scored_candidates.append(
                        (
                            t_idx,
                            el_idx,
                            composite_score,
                            text_sim,
                            iou,
                            dist,
                            is_rel_moved,
                            is_updated,
                        )
                    )
                    track_candidates[t_idx].append((el_idx, composite_score))

        # Check for ambiguity: if a track has 2 or more candidates that are near-tied
        ambiguous_track_indices: Set[int] = set()
        for t_idx, cands in track_candidates.items():
            if len(cands) >= 2:
                cands.sort(key=lambda x: x[1], reverse=True)
                top1 = cands[0]
                top2 = cands[1]
                if abs(top1[1] - top2[1]) < self._ambiguity_threshold:
                    ambiguous_track_indices.add(t_idx)

        # Sort all potential matches globally by composite score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        assigned_tracks: Set[int] = set()
        assigned_elements: Set[int] = set()

        for (
            t_idx,
            el_idx,
            score,
            text_sim,
            iou,
            dist,
            is_rel_moved,
            is_updated,
        ) in scored_candidates:
            if t_idx in assigned_tracks or el_idx in assigned_elements:
                continue

            track = candidate_tracks[t_idx]
            el = current_elements[el_idx]

            if t_idx in ambiguous_track_indices:
                # Mark as UNCERTAIN rather than forcing an arbitrary choice
                uncertain_matches.append((track, el, score))
                assigned_tracks.add(t_idx)
                assigned_elements.add(el_idx)
                continue

            matched_pairs.append((track, el, score, is_rel_moved, is_updated))
            assigned_tracks.add(t_idx)
            assigned_elements.add(el_idx)

        unmatched_track_ids = {
            candidate_tracks[i].track_id
            for i in range(len(candidate_tracks))
            if i not in assigned_tracks
        }
        unmatched_element_indices = {
            i for i in range(len(current_elements)) if i not in assigned_elements
        }

        return (
            matched_pairs,
            unmatched_track_ids,
            unmatched_element_indices,
            uncertain_matches,
        )

    # -----------------------------------------------------------------------
    # Element & Container Extraction
    # -----------------------------------------------------------------------

    def _extract_elements_and_containers(
        self,
        observation: ScreenObservation,
        scene: Optional[UIScene],
    ) -> Tuple[List[UIElement], Dict[int, Optional[UIContainerType]]]:
        """Extract UIElements and build element -> container_type mapping."""
        elements: List[UIElement] = []
        container_map: Dict[int, Optional[UIContainerType]] = {}

        if scene is not None:
            # Prefer rich structured scene
            if scene.interactive_elements:
                elements.extend(scene.interactive_elements)
            for container in scene.containers:
                for el in container.elements:
                    container_map[id(el)] = container.container_type
                    if el not in elements:
                        elements.append(el)

        if not elements:
            # Fallback to observation metadata elements
            raw = observation.metadata.get("elements", [])
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, UIElement):
                        elements.append(item)
                    elif isinstance(item, dict):
                        b = item.get("bounds", {})
                        wb = WindowBounds(
                            left=b.get("left", 0),
                            top=b.get("top", 0),
                            right=b.get("right", 0),
                            bottom=b.get("bottom", 0),
                        )
                        c = item.get("center", {})
                        pt = Point(x=c.get("x", wb.center.x), y=c.get("y", wb.center.y))
                        etype_str = item.get("element_type", "unknown")
                        try:
                            etype = UIElementType(etype_str)
                        except Exception:
                            etype = UIElementType.UNKNOWN
                        elements.append(
                            UIElement(
                                name=item.get("name", "element"),
                                element_type=etype,
                                bounds=wb,
                                center=pt,
                                confidence=float(item.get("confidence", 1.0)),
                                text_content=item.get("text_content"),
                                metadata=dict(item.get("metadata", {})),
                            )
                        )

        if not elements and observation.ocr_result:
            # Fallback to OCR text blocks
            for blk in observation.ocr_result.blocks:
                t = blk.text.strip()
                if not t or not blk.bounds:
                    continue
                t_lower = t.lower()
                if any(k in t_lower for k in ("ok", "cancel", "submit", "apply", "save", "button", "yes", "no")):
                    etype = UIElementType.BUTTON
                elif any(k in t_lower for k in ("search", "username", "password", "input")):
                    etype = UIElementType.INPUT
                else:
                    etype = UIElementType.TEXT
                elements.append(
                    UIElement(
                        name=t,
                        element_type=etype,
                        bounds=blk.bounds,
                        center=blk.bounds.center,
                        confidence=blk.confidence,
                        source=GroundingSource.OCR_EXACT,
                        text_content=t,
                    )
                )

        return elements, container_map

    # -----------------------------------------------------------------------
    # Summary Generation
    # -----------------------------------------------------------------------

    def _generate_summary(
        self,
        new_count: int,
        moved_count: int,
        updated_count: int,
        missing_count: int,
        reappeared_count: int,
        uncertain_count: int,
        active_count: int,
        target_filter: Optional[str],
        moved_tracks: List[VisualElementTrack],
        missing_tracks: List[VisualElementTrack],
        reappeared_tracks: List[VisualElementTrack],
    ) -> str:
        """Construct a voice-safe natural-language tracking summary."""
        if target_filter and target_filter.strip():
            tf = target_filter.strip()
            if reappeared_tracks:
                return f"{tf} reappeared on the screen."
            if moved_tracks:
                return f"{tf} moved."
            if missing_tracks:
                return f"{tf} disappeared."
            if active_count > 0:
                return f"{tf} is still there."
            return f"{tf} is not on the screen."

        parts: List[str] = []
        if moved_count > 0:
            names = ", ".join(f"'{t.canonical_name}'" for t in moved_tracks[:2])
            parts.append(f"{names} moved")
        if reappeared_count > 0:
            names = ", ".join(f"'{t.canonical_name}'" for t in reappeared_tracks[:2])
            parts.append(f"{names} reappeared")
        if missing_count > 0:
            names = ", ".join(f"'{t.canonical_name}'" for t in missing_tracks[:2])
            parts.append(f"{names} disappeared")
        if uncertain_count > 0:
            parts.append(f"{uncertain_count} ambiguous track{'s' if uncertain_count > 1 else ''}")

        if parts:
            return ". ".join(parts) + "."

        if active_count > 0:
            return f"Tracking {active_count} visual element{'s' if active_count > 1 else ''}."
        return "No visual elements are currently being tracked."

    # -----------------------------------------------------------------------
    # Error Handling & Edge Cases
    # -----------------------------------------------------------------------

    def _handle_invalid_observation(
        self,
        observation: Optional[ScreenObservation],
        tracking_id: str,
    ) -> VisualTrackingResult:
        """Handle missing, expired, or invalid observation safely."""
        with self._lock:
            # Mark all existing tracks as MISSING or TERMINATED
            missing_tracks: List[VisualElementTrack] = []
            terminated_tracks: List[VisualElementTrack] = []
            for tid, track in list(self._tracks.items()):
                if track.status == VisualTrackStatus.TERMINATED:
                    continue
                new_missing = track.missing_count + 1
                new_status = (
                    VisualTrackStatus.TERMINATED
                    if new_missing > self._max_missing_frames
                    else VisualTrackStatus.MISSING
                )
                updated = VisualElementTrack(
                    track_id=track.track_id,
                    element_type=track.element_type,
                    canonical_name=track.canonical_name,
                    first_observation_id=track.first_observation_id,
                    last_observation_id=track.last_observation_id,
                    last_known_bounds=track.last_known_bounds,
                    last_known_center=track.last_known_center,
                    confidence=max(0.0, track.confidence - 0.2),
                    status=new_status,
                    history_count=track.history_count,
                    missing_count=new_missing,
                    container_type=track.container_type,
                    window_title=track.window_title,
                    process_name=track.process_name,
                    metadata=dict(track.metadata),
                )
                self._tracks[tid] = updated
                if new_status == VisualTrackStatus.MISSING:
                    missing_tracks.append(updated)
                elif new_status == VisualTrackStatus.TERMINATED:
                    terminated_tracks.append(updated)

            active_tracks = tuple(
                t for t in self._tracks.values()
                if t.status != VisualTrackStatus.TERMINATED
            )

        return VisualTrackingResult(
            tracking_id=tracking_id,
            observation_id=observation.observation_id if observation else "",
            active_tracks=active_tracks,
            missing_tracks=tuple(missing_tracks),
            terminated_tracks=tuple(terminated_tracks),
            confidence=0.0,
            summary="Cannot track elements from an invalid or empty screen observation.",
            metadata={"error": "invalid_or_empty_observation"},
        )

    # -----------------------------------------------------------------------
    # Benchmark / Telemetry Support
    # -----------------------------------------------------------------------

    def benchmark_matching(
        self,
        elements_t0: Sequence[UIElement],
        elements_t1: Sequence[UIElement],
    ) -> Dict[str, Any]:
        """Benchmark association performance across structured element collections."""
        start = time.perf_counter()
        t0_tracks = [
            VisualElementTrack(
                track_id=f"bm_{i}",
                element_type=el.element_type,
                canonical_name=el.name,
                first_observation_id="t0",
                last_observation_id="t0",
                last_known_bounds=el.bounds,
                last_known_center=el.center,
                confidence=el.confidence,
                status=VisualTrackStatus.TRACKED,
            )
            for i, el in enumerate(elements_t0)
        ]

        matched, unmatched_t, unmatched_el, uncertain = self._associate_elements_with_tracks(
            candidate_tracks=t0_tracks,
            current_elements=list(elements_t1),
            container_map={},
            win_dx=0,
            win_dy=0,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0

        return {
            "duration_ms": round(duration_ms, 3),
            "t0_count": len(elements_t0),
            "t1_count": len(elements_t1),
            "matched_count": len(matched),
            "unmatched_tracks": len(unmatched_t),
            "unmatched_elements": len(unmatched_el),
            "uncertain_count": len(uncertain),
            "ambiguity_rate": len(uncertain) / max(1, len(matched) + len(uncertain)),
        }


__all__ = [
    "DEFAULT_AMBIGUITY_THRESHOLD",
    "DEFAULT_MAX_MISSING_FRAMES",
    "DEFAULT_MAX_TRACKS",
    "DEFAULT_MIN_MATCH_SCORE",
    "DEFAULT_MOVE_DISTANCE_THRESHOLD",
    "DEFAULT_RESIZE_RATIO_THRESHOLD",
    "VisualTrackingEngine",
]
