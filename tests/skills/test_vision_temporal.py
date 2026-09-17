"""Phase 27.15 Visual Temporal Context & Event History Tests.

Comprehensive deterministic unit and integration test suite covering:
1. APPEARED on NEW track
2. MOVED event
3. UPDATED event
4. MISSING does NOT immediately emit DISAPPEARED
5. TERMINATED emits DISAPPEARED
6. REAPPEARED event
7. STATE_CHANGED
8. CONTAINER_CHANGED
9. WINDOW_CHANGED
10. UNCERTAIN transition
11. Steady-state silence
12. Semantic duplicate suppression
13. Legitimate rapid different transitions preserved
14. max_events FIFO eviction
15. TTL pruning
16. Recent-event query
17. Event-type filtering
18. Element-history query
19. track_id filtering
20. Disappeared-element query
21. State-change query
22. Window/process isolation
23. reset() clears history
24. No raw pixel persistence
25. No background service/thread/timer
26. Sensitive-window protection
27. Password/secret content never retained
28. Intent routing
29. Voice-safe formatting
30. Backward compatibility with tracking/delta/verification
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest import mock
import uuid

import pytest

from app.ai.intent_router import IntentRouter, IntentType
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.vision_skills import VisionSkills
from app.vision.models import (
    CaptureAuthorization,
    CaptureCategory,
    CaptureDecision,
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    Point,
    ScreenCapture,
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
from app.vision.security import SecureVisionManager, VisionSecurityPolicy
from app.vision.temporal import (
    DEFAULT_MAX_EVENTS,
    DEFAULT_TTL_SECONDS,
    VisualTemporalEngine,
    VisualTemporalLog,
)
from app.vision.tracking import VisualTrackingEngine


# ---------------------------------------------------------------------------
# Test Helpers & Synthetic Fixtures
# ---------------------------------------------------------------------------


def make_track(
    track_id: str,
    canonical_name: str,
    status: VisualTrackStatus = VisualTrackStatus.NEW,
    element_type: UIElementType = UIElementType.BUTTON,
    bounds: Optional[WindowBounds] = None,
    container_type: Optional[UIContainerType] = None,
    window_title: str = "TestApp - Main",
    process_name: str = "testapp.exe",
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> VisualElementTrack:
    """Create a synthetic VisualElementTrack for testing."""
    b = bounds or WindowBounds(left=100, top=100, right=200, bottom=140)
    return VisualElementTrack(
        track_id=track_id,
        element_type=element_type,
        canonical_name=canonical_name,
        first_observation_id="obs_0",
        last_observation_id="obs_1",
        last_known_bounds=b,
        last_known_center=b.center,
        confidence=confidence,
        status=status,
        container_type=container_type,
        window_title=window_title,
        process_name=process_name,
        metadata=dict(metadata or {}),
    )


def make_tracking_result(
    observation_id: str = "obs_1",
    active_tracks: Optional[List[VisualElementTrack]] = None,
    new_tracks: Optional[List[VisualElementTrack]] = None,
    moved_tracks: Optional[List[VisualElementTrack]] = None,
    updated_tracks: Optional[List[VisualElementTrack]] = None,
    missing_tracks: Optional[List[VisualElementTrack]] = None,
    reappeared_tracks: Optional[List[VisualElementTrack]] = None,
    uncertain_tracks: Optional[List[VisualElementTrack]] = None,
    terminated_tracks: Optional[List[VisualElementTrack]] = None,
    confidence: float = 1.0,
) -> VisualTrackingResult:
    """Create a synthetic VisualTrackingResult for testing."""
    return VisualTrackingResult(
        tracking_id=f"trk_res_{uuid.uuid4().hex[:8]}",
        observation_id=observation_id,
        active_tracks=tuple(active_tracks or []),
        new_tracks=tuple(new_tracks or []),
        moved_tracks=tuple(moved_tracks or []),
        updated_tracks=tuple(updated_tracks or []),
        missing_tracks=tuple(missing_tracks or []),
        reappeared_tracks=tuple(reappeared_tracks or []),
        uncertain_tracks=tuple(uncertain_tracks or []),
        terminated_tracks=tuple(terminated_tracks or []),
        confidence=confidence,
    )


def make_capture(
    width: int = 800,
    height: int = 600,
    bounds: Optional[WindowBounds] = None,
) -> ScreenCapture:
    """Create a synthetic ScreenCapture frame."""
    data = b"\x80\x80\x80\xff" * (width * height)
    return ScreenCapture(
        raw_data=data,
        width=width,
        height=height,
        pixel_format="RGBA",
        bounds=bounds or WindowBounds(0, 0, width, height),
    )


def make_observation(
    window_title: str = "TestApp - Main",
    process_name: str = "testapp.exe",
    is_sensitive: bool = False,
) -> ScreenObservation:
    """Create a lightweight synthetic ScreenObservation."""
    wb = WindowBounds(0, 0, 800, 600)
    auth = (
        CaptureAuthorization(
            decision=CaptureDecision.BLOCK,
            category=CaptureCategory.SENSITIVE_APPLICATION,
            reason="Blocked sensitive app",
        )
        if is_sensitive
        else CaptureAuthorization(
            decision=CaptureDecision.ALLOW,
            category=CaptureCategory.SAFE,
            reason="Safe window",
        )
    )
    cap = None if is_sensitive else make_capture(width=wb.width, height=wb.height, bounds=wb)
    return ScreenObservation(
        id=f"obs_{uuid.uuid4().hex[:8]}",
        capture=cap,
        authorization=auth,
        source="window",
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "is_sensitive": is_sensitive,
        },
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVisualTemporalContext:
    """Test suite verifying Visual Temporal Context & Event History (Phase 27.15)."""

    def test_01_appeared_on_new_track(self) -> None:
        """1. A new track in new_tracks confidently derives APPEARED event."""
        engine = VisualTemporalEngine()
        track = make_track("trk_1", "Submit Button", status=VisualTrackStatus.NEW)
        res = make_tracking_result(new_tracks=[track], active_tracks=[track])

        events = engine.derive_events_from_tracking(res, current_time=1000.0)

        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == VisualEventType.APPEARED
        assert evt.canonical_name == "Submit Button"
        assert evt.track_id == "trk_1"
        assert evt.timestamp == 1000.0
        assert "appeared" in evt.summary.lower()

    def test_02_moved_event(self) -> None:
        """2. Track classified MOVED derives MOVED event with bounds and displacement."""
        engine = VisualTemporalEngine()
        # Seed initial position
        t0 = make_track("trk_1", "Dialog Box", bounds=WindowBounds(100, 100, 300, 200))
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Move Dialog Box
        t1 = make_track("trk_1", "Dialog Box", status=VisualTrackStatus.MOVED, bounds=WindowBounds(150, 120, 350, 220))
        res1 = make_tracking_result(moved_tracks=[t1], active_tracks=[t1])
        events = engine.derive_events_from_tracking(res1, current_time=1001.0)

        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == VisualEventType.MOVED
        assert evt.prior_bounds == WindowBounds(100, 100, 300, 200)
        assert evt.current_bounds == WindowBounds(150, 120, 350, 220)
        assert evt.displacement == Point(50, 20)
        assert "moved" in evt.summary.lower()

    def test_03_updated_event(self) -> None:
        """3. Track classified UPDATED derives UPDATED event."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Status Label")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        t1 = make_track("trk_1", "Status Label", status=VisualTrackStatus.UPDATED, metadata={"text_content": "Ready"})
        events = engine.derive_events_from_tracking(make_tracking_result(updated_tracks=[t1], active_tracks=[t1]), current_time=1001.0)

        assert len(events) == 1
        assert events[0].event_type == VisualEventType.UPDATED
        assert events[0].canonical_name == "Status Label"
        assert "updated" in events[0].summary.lower()

    def test_04_missing_does_not_immediately_emit_disappeared(self) -> None:
        """4. Entering MISSING within grace window must NOT emit DISAPPEARED."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Cancel Button")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Track is temporarily missing (e.g. occluded)
        t_missing = make_track("trk_1", "Cancel Button", status=VisualTrackStatus.MISSING)
        events = engine.derive_events_from_tracking(make_tracking_result(missing_tracks=[t_missing]), current_time=1001.0)

        # Critical invariant: MISSING does not produce DISAPPEARED
        assert len(events) == 0
        disappeared_events = [e for e in engine.log.get_events(current_time=1001.0) if e.event_type == VisualEventType.DISAPPEARED]
        assert len(disappeared_events) == 0

    def test_05_terminated_emits_disappeared(self) -> None:
        """5. Track reaching TERMINATED emits DISAPPEARED."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Dismiss Button")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        t_term = make_track("trk_1", "Dismiss Button", status=VisualTrackStatus.TERMINATED)
        events = engine.derive_events_from_tracking(make_tracking_result(terminated_tracks=[t_term]), current_time=1001.0)

        assert len(events) == 1
        assert events[0].event_type == VisualEventType.DISAPPEARED
        assert events[0].canonical_name == "Dismiss Button"
        assert "disappeared" in events[0].summary.lower()

    def test_06_reappeared_event(self) -> None:
        """6. Track in reappeared_tracks derives REAPPEARED event."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Popup Banner")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        t_reapp = make_track("trk_1", "Popup Banner", status=VisualTrackStatus.REAPPEARED)
        events = engine.derive_events_from_tracking(make_tracking_result(reappeared_tracks=[t_reapp], active_tracks=[t_reapp]), current_time=1001.0)

        assert len(events) == 1
        assert events[0].event_type == VisualEventType.REAPPEARED
        assert events[0].canonical_name == "Popup Banner"
        assert "reappeared" in events[0].summary.lower()

    def test_07_state_changed(self) -> None:
        """7. STATE_CHANGED emitted when prior and current ControlVisualState differ materially."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Checkout Button", metadata={"detected_state": "enabled"})
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Button disabled on submission
        t1 = make_track("trk_1", "Checkout Button", status=VisualTrackStatus.TRACKED, metadata={"detected_state": "disabled"})
        events = engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t1]), current_time=1001.0)

        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == VisualEventType.STATE_CHANGED
        assert evt.prior_state == ControlVisualState.ENABLED
        assert evt.current_state == ControlVisualState.DISABLED
        assert "changed from enabled to disabled" in evt.summary.lower()

    def test_08_container_changed(self) -> None:
        """8. CONTAINER_CHANGED emitted when container materially changes."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Search Tool", container_type=UIContainerType.TOOLBAR)
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Tool docked into sidebar
        t1 = make_track("trk_1", "Search Tool", status=VisualTrackStatus.TRACKED, container_type=UIContainerType.SIDEBAR)
        events = engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t1]), current_time=1001.0)

        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == VisualEventType.CONTAINER_CHANGED
        assert evt.prior_container == UIContainerType.TOOLBAR
        assert evt.current_container == UIContainerType.SIDEBAR
        assert "sidebar" in evt.summary.lower()

    def test_09_window_changed(self) -> None:
        """9. WINDOW_CHANGED emitted when active window or process context changes."""
        engine = VisualTemporalEngine()
        obs0 = make_observation(window_title="Document 1 - Word", process_name="winword.exe")
        engine.derive_events_from_tracking(make_tracking_result(), observation=obs0, current_time=1000.0)

        obs1 = make_observation(window_title="Inbox - Outlook", process_name="outlook.exe")
        events = engine.derive_events_from_tracking(make_tracking_result(), observation=obs1, current_time=1001.0)

        assert any(e.event_type == VisualEventType.WINDOW_CHANGED for e in events)
        win_evt = [e for e in events if e.event_type == VisualEventType.WINDOW_CHANGED][0]
        assert "Inbox - Outlook" in win_evt.summary

    def test_10_uncertain_transition(self) -> None:
        """10. Ambiguous match in uncertain_tracks emits UNCERTAIN event."""
        engine = VisualTemporalEngine()
        t_unc = make_track("trk_ambig", "Duplicate Button", status=VisualTrackStatus.UNCERTAIN, confidence=0.4)
        events = engine.derive_events_from_tracking(make_tracking_result(uncertain_tracks=[t_unc]), current_time=1000.0)

        assert len(events) == 1
        assert events[0].event_type == VisualEventType.UNCERTAIN
        assert events[0].confidence == 0.4

    def test_11_steady_state_silence(self) -> None:
        """11. Steady-state observations with no changes produce no events."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Static Label")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Subsequent observation: same element, TRACKED, same state/position
        t1 = make_track("trk_1", "Static Label", status=VisualTrackStatus.TRACKED)
        events = engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t1]), current_time=1001.0)

        assert len(events) == 0

    def test_12_semantic_duplicate_suppression(self) -> None:
        """12. Identical semantic transition without new evidence is deduplicated."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Apply Button")
        events1 = engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)
        assert len(events1) == 1

        # Re-sending identical APPEARED evidence immediately without state change
        events2 = engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.1)
        assert len(events2) == 0

    def test_13_legitimate_rapid_different_transitions_preserved(self) -> None:
        """13. Rapid legitimate different transitions are preserved without universal debounce."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Toggle Switch", metadata={"detected_state": "enabled"})
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        # Quick click: enabled -> disabled (within 100ms)
        t1 = make_track("trk_1", "Toggle Switch", status=VisualTrackStatus.TRACKED, metadata={"detected_state": "disabled"})
        events1 = engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t1]), current_time=1000.1)
        assert len(events1) == 1
        assert events1[0].event_type == VisualEventType.STATE_CHANGED

        # Quick click again: disabled -> enabled (within another 100ms)
        t2 = make_track("trk_1", "Toggle Switch", status=VisualTrackStatus.TRACKED, metadata={"detected_state": "enabled"})
        events2 = engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t2]), current_time=1000.2)
        assert len(events2) == 1
        assert events2[0].event_type == VisualEventType.STATE_CHANGED

    def test_14_max_events_fifo_eviction(self) -> None:
        """14. Log enforces bounded capacity and FIFO eviction."""
        log = VisualTemporalLog(max_events=3, ttl_seconds=300.0)
        for i in range(5):
            evt = VisualTemporalEvent(
                event_id=f"evt_{i}",
                event_type=VisualEventType.APPEARED,
                timestamp=1000.0 + i,
                observation_id=f"obs_{i}",
                canonical_name=f"Button_{i}",
            )
            log.append(evt, current_time=1000.0 + i)

        assert len(log) == 3
        evts = log.get_events(current_time=1005.0)
        assert [e.canonical_name for e in evts] == ["Button_2", "Button_3", "Button_4"]

    def test_15_ttl_pruning(self) -> None:
        """15. Events older than ttl_seconds are pruned on query/maintenance."""
        log = VisualTemporalLog(max_events=100, ttl_seconds=10.0)
        e_old = VisualTemporalEvent(
            event_id="evt_old",
            event_type=VisualEventType.APPEARED,
            timestamp=1000.0,
            observation_id="obs_0",
            canonical_name="OldButton",
        )
        e_new = VisualTemporalEvent(
            event_id="evt_new",
            event_type=VisualEventType.APPEARED,
            timestamp=1005.0,
            observation_id="obs_1",
            canonical_name="NewButton",
        )
        log.append(e_old, current_time=1000.0)
        log.append(e_new, current_time=1005.0)

        # At timestamp 1012.0 (cutoff is 1002.0), old event should be pruned
        pruned = log.prune(current_time=1012.0)
        assert pruned == 1
        active = log.get_events(current_time=1012.0)
        assert len(active) == 1
        assert active[0].canonical_name == "NewButton"

    def test_16_recent_event_query(self) -> None:
        """16. get_recent_events returns recent events up to limit in reverse chronological order."""
        engine = VisualTemporalEngine()
        for i in range(5):
            t = make_track(f"trk_{i}", f"Item_{i}")
            engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t]), current_time=1000.0 + i)

        res = engine.get_recent_events(limit=3, current_time=1010.0)
        assert res.total_count == 3
        assert [e.canonical_name for e in res.events] == ["Item_4", "Item_3", "Item_2"]
        assert "recent visual changes detected" in res.summary

    def test_17_event_type_filtering(self) -> None:
        """17. get_recent_events filters by VisualEventType."""
        engine = VisualTemporalEngine()
        t1 = make_track("trk_1", "Btn1")
        t2 = make_track("trk_2", "Btn2")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t1]), current_time=1000.0)
        engine.derive_events_from_tracking(make_tracking_result(moved_tracks=[t2]), current_time=1001.0)

        res = engine.get_recent_events(event_type=VisualEventType.MOVED, current_time=1002.0)
        assert res.total_count == 1
        assert res.events[0].canonical_name == "Btn2"
        assert res.events[0].event_type == VisualEventType.MOVED

    def test_18_element_history_query(self) -> None:
        """18. get_element_history retrieves full trajectory for target element."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_sub", "Submit Button", bounds=WindowBounds(100, 100, 200, 140))
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0]), current_time=1000.0)

        t1 = make_track("trk_sub", "Submit Button", status=VisualTrackStatus.MOVED, bounds=WindowBounds(150, 100, 250, 140))
        engine.derive_events_from_tracking(make_tracking_result(moved_tracks=[t1]), current_time=1001.0)

        t2 = make_track("trk_sub", "Submit Button", status=VisualTrackStatus.TERMINATED)
        engine.derive_events_from_tracking(make_tracking_result(terminated_tracks=[t2]), current_time=1002.0)

        res = engine.get_element_history(target="Submit Button", current_time=1003.0)
        assert res.total_count == 3
        types = [e.event_type for e in res.events]
        assert types == [VisualEventType.APPEARED, VisualEventType.MOVED, VisualEventType.DISAPPEARED]
        assert "underwent 3 events" in res.summary

    def test_19_track_id_filtering(self) -> None:
        """19. get_element_history supports exact track_id targeting."""
        engine = VisualTemporalEngine()
        t1 = make_track("trk_alpha", "Save", bounds=WindowBounds(10, 10, 50, 30))
        t2 = make_track("trk_beta", "Save", bounds=WindowBounds(100, 100, 150, 130))
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t1, t2]), current_time=1000.0)

        res = engine.get_element_history(target="Save", track_id="trk_beta", current_time=1001.0)
        assert res.total_count == 1
        assert res.events[0].track_id == "trk_beta"

    def test_20_disappeared_element_query(self) -> None:
        """20. get_disappeared_elements returns only DISAPPEARED events."""
        engine = VisualTemporalEngine()
        t_live = make_track("trk_1", "Live Control")
        t_gone = make_track("trk_2", "Banner Warning", status=VisualTrackStatus.TERMINATED)
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t_live], terminated_tracks=[t_gone]), current_time=1000.0)

        res = engine.get_disappeared_elements(current_time=1001.0)
        assert res.total_count == 1
        assert res.events[0].canonical_name == "Banner Warning"
        assert res.events[0].event_type == VisualEventType.DISAPPEARED

    def test_21_state_change_query(self) -> None:
        """21. get_state_changes retrieves only STATE_CHANGED events."""
        engine = VisualTemporalEngine()
        t0 = make_track("trk_1", "Mute Toggle", metadata={"detected_state": "unchecked"})
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t0], active_tracks=[t0]), current_time=1000.0)

        t1 = make_track("trk_1", "Mute Toggle", status=VisualTrackStatus.TRACKED, metadata={"detected_state": "checked"})
        engine.derive_events_from_tracking(make_tracking_result(active_tracks=[t1]), current_time=1001.0)

        res = engine.get_state_changes(target="Mute Toggle", current_time=1002.0)
        assert res.total_count == 1
        assert res.events[0].event_type == VisualEventType.STATE_CHANGED
        assert res.events[0].prior_state == ControlVisualState.UNCHECKED
        assert res.events[0].current_state == ControlVisualState.CHECKED

    def test_22_window_process_isolation(self) -> None:
        """22. Events isolate and filter by window context correctly."""
        engine = VisualTemporalEngine()
        obs_word = make_observation(window_title="Document - Word", process_name="winword.exe")
        t_word = make_track("trk_w", "Font Size", window_title="Document - Word", process_name="winword.exe")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t_word]), observation=obs_word, current_time=1000.0)

        obs_calc = make_observation(window_title="Calculator", process_name="calc.exe")
        t_calc = make_track("trk_c", "Plus Button", window_title="Calculator", process_name="calc.exe")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t_calc]), observation=obs_calc, current_time=1001.0)

        res_calc = engine.get_recent_events(window_filter="Calculator", current_time=1002.0)
        assert res_calc.total_count == 2
        assert all(e.window_title == "Calculator" for e in res_calc.events)
        assert not any("Word" in (e.window_title or "") for e in res_calc.events)

    def test_23_reset_clears_history(self) -> None:
        """23. reset() completely purges events and cached state."""
        engine = VisualTemporalEngine()
        t = make_track("trk_1", "Temp Button")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t]), current_time=1000.0)
        assert len(engine.log) > 0

        engine.reset()
        assert len(engine.log) == 0
        assert engine.get_recent_events().total_count == 0

    def test_24_no_raw_pixel_persistence(self) -> None:
        """24. Invariant: zero raw screenshots, image bytes, or base64 in models or results."""
        evt = VisualTemporalEvent(
            event_id="evt_test",
            event_type=VisualEventType.APPEARED,
            timestamp=1000.0,
            observation_id="obs_test",
            canonical_name="TestElement",
            metadata={"source": "grounding"},
        )
        d = evt.to_dict()
        for forbidden in ("raw_data", "image", "bytes", "base64", "pixels"):
            assert forbidden not in d
            assert forbidden not in d["metadata"]

        res = VisualTemporalHistoryResult(query_id="qry_1", events=(evt,))
        rd = res.to_dict()
        for forbidden in ("raw_data", "image", "bytes", "base64", "pixels"):
            assert forbidden not in rd

    def test_25_no_background_service_thread_or_timer(self) -> None:
        """25. Invariant: VisualTemporalEngine creates no daemon threads or background polling."""
        initial_threads = threading.active_count()
        engine = VisualTemporalEngine()
        t = make_track("trk_1", "Test")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t]))
        final_threads = threading.active_count()
        assert final_threads == initial_threads

    def test_26_sensitive_window_protection(self) -> None:
        """26. Sensitive observations are blocked from event derivation."""
        engine = VisualTemporalEngine()
        obs_sens = make_observation(window_title="Password Manager", is_sensitive=True)
        t = make_track("trk_sec", "Vault Password Entry")
        events = engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t]), observation=obs_sens)

        assert len(events) == 0
        assert len(engine.log) == 0

    def test_26b_blocked_observation_without_sensitive_flag(self) -> None:
        """26b. Blocked/unauthorized observations without is_sensitive=True produce zero events."""
        engine = VisualTemporalEngine()
        auth_blocked = CaptureAuthorization(
            decision=CaptureDecision.BLOCK,
            category=CaptureCategory.SENSITIVE_APPLICATION,
            reason="Blocked by security rule",
        )
        obs_blocked = ScreenObservation(
            id="obs_blk",
            capture=None,
            authorization=auth_blocked,
            source="window",
            metadata={"blocked": True, "window_title": "Bitwarden Vault", "process_name": "bitwarden.exe"},
        )
        track = make_track("trk_pwd", "Master Password Field")
        events = engine.derive_events_from_tracking(
            make_tracking_result(new_tracks=[track]),
            observation=obs_blocked,
            current_time=1000.0,
        )

        assert len(events) == 0
        assert len(engine.log) == 0
        # Verify no protected window title or process name leaked into log
        res = engine.get_recent_events(current_time=1001.0)
        assert res.total_count == 0
        assert "bitwarden" not in str(res.to_dict()).lower()

    def test_26c_sensitive_window_transition_isolation(self) -> None:
        """26c. Normal App A -> Protected App -> Normal App B does NOT claim App A -> App B."""
        engine = VisualTemporalEngine()

        # Step 1: Normal App A
        obs_a = make_observation(window_title="Editor - Code", process_name="code.exe")
        t_a = make_track("trk_a", "Line Number", window_title="Editor - Code", process_name="code.exe")
        engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t_a]), observation=obs_a, current_time=1000.0)

        # Step 2: Sensitive/Blocked Window (e.g. KeePass / 1Password)
        auth_blocked = CaptureAuthorization(
            decision=CaptureDecision.BLOCK,
            category=CaptureCategory.CREDENTIAL_INTERFACE,
            reason="Credential window protected",
        )
        obs_prot = ScreenObservation(
            id="obs_prot",
            capture=None,
            authorization=auth_blocked,
            source="window",
            metadata={"blocked": True, "window_title": "KeePass - Database.kdbx", "process_name": "keepass.exe"},
        )
        events_prot = engine.derive_events_from_tracking(
            make_tracking_result(new_tracks=[make_track("trk_kp", "Password")]),
            observation=obs_prot,
            current_time=1001.0,
        )
        assert len(events_prot) == 0

        # Step 3: Normal App B
        obs_b = make_observation(window_title="Browser - Research", process_name="browser.exe")
        t_b = make_track("trk_b", "Search Box", window_title="Browser - Research", process_name="browser.exe")
        events_b = engine.derive_events_from_tracking(
            make_tracking_result(new_tracks=[t_b]),
            observation=obs_b,
            current_time=1002.0,
        )

        # Verify: App B does NOT claim a direct transition from App A!
        win_events = [e for e in events_b if e.event_type == VisualEventType.WINDOW_CHANGED]
        assert len(win_events) == 0, "Must not claim a direct window transition across a protected window!"

        # Verify no KeePass information leaked anywhere
        all_events = engine.log.get_events(current_time=1003.0)
        for evt in all_events:
            assert "keepass" not in (evt.window_title or "").lower()
            assert "keepass" not in (evt.process_name or "").lower()
            assert "keepass" not in evt.summary.lower()

        # Step 4: Subsequent normal transition App B -> App C works normally
        obs_c = make_observation(window_title="Terminal - Shell", process_name="cmd.exe")
        events_c = engine.derive_events_from_tracking(
            make_tracking_result(),
            observation=obs_c,
            current_time=1004.0,
        )
        win_events_c = [e for e in events_c if e.event_type == VisualEventType.WINDOW_CHANGED]
        assert len(win_events_c) == 1
        assert "Browser - Research" in str(win_events_c[0].metadata.get("prior_window_title"))
        assert "Terminal - Shell" in str(win_events_c[0].metadata.get("current_window_title"))

    def test_27_password_secret_content_never_retained(self) -> None:
        """27. Invariant: password or secret content is sanitized and never retained in events."""
        engine = VisualTemporalEngine()
        t = make_track("trk_sec", "Secret Password Token Input")
        events = engine.derive_events_from_tracking(make_tracking_result(new_tracks=[t]))

        assert len(events) == 1
        assert events[0].canonical_name == "[REDACTED]"
        assert "password" not in events[0].canonical_name.lower()

    def test_28_intent_routing(self) -> None:
        """28. IntentRouter correctly routes retrospective visual queries to IntentType.VISION."""
        router = IntentRouter()
        queries = [
            "what happened recently",
            "what just changed on my screen",
            "show recent visual events",
            "what happened to the submit button",
            "did any buttons change state",
            "what elements disappeared",
        ]
        for q in queries:
            result = router.classify(q)
            assert result.intent == IntentType.VISION, f"Failed for query: {q}"

    def test_29_voice_safe_formatting(self) -> None:
        """29. BaseSystemSkill produces concise voice-safe messages for temporal events."""
        # Test empty
        empty_res = SystemSkillResult(
            success=True,
            operation="get_recent_events",
            data={"events": [], "summary": "No recent visual events detected."},
        )
        assert "No recent visual events detected." in empty_res.message

        # Test single event
        single_res = SystemSkillResult(
            success=True,
            operation="get_recent_events",
            data={"events": [{}], "summary": "The Submit button moved."},
        )
        assert "The Submit button moved." in single_res.message

        # Test multiple events
        multi_res = SystemSkillResult(
            success=True,
            operation="query_event_history",
            data={"events": [{}, {}, {}], "summary": "Three recent visual changes detected."},
        )
        assert "Three recent visual changes detected." in multi_res.message

    def test_30_backward_compatibility_with_tracking_delta_verification(self) -> None:
        """30. VisionSkills integration with get_recent_events and query_event_history."""
        skill = VisionSkills()
        assert skill.can_handle("what happened recently")
        assert skill.can_handle("what just changed on my screen")
        assert skill.can_handle("show recent visual events")
        assert skill.can_handle("what happened to the submit button")
        assert skill.can_handle("did any buttons change state")

        # Parse command test
        op, tgt, params, _ = skill.parse_command("what happened recently")
        assert op == "get_recent_events"

        op2, tgt2, params2, _ = skill.parse_command("what happened to the submit button")
        assert op2 == "query_event_history"
        assert params2.get("target") == "submit button"

        op3, tgt3, params3, _ = skill.parse_command("did any buttons change state")
        assert op3 == "query_event_history"
        assert params3.get("event_type") == "STATE_CHANGED"
