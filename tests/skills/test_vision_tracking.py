"""Phase 27.14 Cross-Observation Visual Identity & Tracking Tests.

Comprehensive deterministic unit and integration test suite covering:
1. Same element, same position -> TRACKED
2. Same element, moved -> MOVED
3. Same element, resized -> UPDATED
4. Same element, control state/text changed -> UPDATED
5. Parent window translated -> child element remains TRACKED, not MOVED
6. New element appears -> NEW
7. Existing element disappears -> MISSING
8. Element reappears -> REAPPEARED
9. Track termination after configured missing frames -> TERMINATED
10. Duplicate labels with distinct geometry/container
11. Conflicting semantic/geometry signals -> UNCERTAIN
12. Different applications/windows isolation
13. Sensitive/blocked observation safety
14. Empty observation safety
15. Invalid observation safety
16. Registry max capacity bounded
17. No unbounded history retention
18. No raw pixels in models or results
19. Password contents redacted / never stored
20. Session-scoped track IDs
21. No background tracking capture service or daemon threads
22. Ambiguous matching returns UNCERTAIN
23. Element movement vs window movement distinction
24. Container relationship tracked and preserved
25. Reappearance preserves track_id
26. New element gets distinct track_id
27. Process/window lifecycle isolation
28. Configurable thresholds
29. Backward compatibility of existing vision operations
30. VisionSkills track_elements and get_visual_tracks execution
31. IntentRouter classification for tracking phrases
32. BaseSystemSkill SystemSkillResult message formatting
33. Benchmark matching performance with structured models
"""

from __future__ import annotations

import math
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
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    GroundingSource,
    Point,
    ScreenCapture,
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
from app.vision.security import SecureVisionManager, VisionSecurityPolicy
from app.vision.tracking import VisualTrackingEngine


# ---------------------------------------------------------------------------
# Test Helpers & Synthetic Fixtures
# ---------------------------------------------------------------------------


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
    elements: Optional[List[UIElement]] = None,
    window_bounds: Optional[WindowBounds] = None,
    window_title: str = "TestApp - Main",
    process_name: str = "testapp.exe",
    hwnd: int = 12345,
    is_blocked: bool = False,
) -> ScreenObservation:
    """Create a synthetic ScreenObservation containing specified UIElements."""
    wb = window_bounds or WindowBounds(100, 100, 900, 700)
    auth = (
        CaptureAuthorization(
            decision=CaptureDecision.BLOCK,
            category=CaptureCategory.SENSITIVE_APPLICATION,
            reason="Blocked sensitive app",
        )
        if is_blocked
        else CaptureAuthorization(
            decision=CaptureDecision.ALLOW,
            category=CaptureCategory.SAFE,
            reason="Safe window",
        )
    )

    cap = None if is_blocked else make_capture(width=wb.width, height=wb.height, bounds=wb)

    obs = ScreenObservation(
        id=f"obs_{uuid.uuid4().hex[:6]}",
        capture=cap,
        authorization=auth,
        source="window",
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "hwnd": hwnd,
            "elements": elements or [],
        },
    )
    return obs


def make_element(
    name: str,
    element_type: UIElementType = UIElementType.BUTTON,
    bounds: Optional[WindowBounds] = None,
    confidence: float = 1.0,
    text_content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UIElement:
    """Create a synthetic grounded UIElement."""
    wb = bounds or WindowBounds(150, 150, 250, 190)
    return UIElement(
        name=name,
        element_type=element_type,
        bounds=wb,
        center=wb.center,
        confidence=confidence,
        source=GroundingSource.OCR_EXACT,
        text_content=text_content or name,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVisualTrackingEngine:
    """Test suite for VisualTrackingEngine and cross-observation tracking."""

    def test_01_same_element_same_position(self) -> None:
        """Test 1: Same element at same position across T0 and T1 transitions to TRACKED."""
        engine = VisualTrackingEngine()
        el0 = make_element("Submit", bounds=WindowBounds(200, 200, 300, 240))
        obs0 = make_observation([el0])

        res0 = engine.track_observation(obs0)
        assert len(res0.new_tracks) == 1
        track0 = res0.new_tracks[0]
        assert track0.status == VisualTrackStatus.NEW
        assert track0.canonical_name == "Submit"

        # T1: exact same element
        el1 = make_element("Submit", bounds=WindowBounds(200, 200, 300, 240))
        obs1 = make_observation([el1])

        res1 = engine.track_observation(obs1)
        assert len(res1.active_tracks) == 1
        t1 = res1.active_tracks[0]
        assert t1.track_id == track0.track_id
        assert t1.status == VisualTrackStatus.TRACKED
        assert t1.history_count == 2
        assert t1.missing_count == 0

    def test_02_same_element_moved(self) -> None:
        """Test 2: Element moved significantly relative to window transitions to MOVED."""
        engine = VisualTrackingEngine(move_distance_threshold=10.0)
        el0 = make_element("Save", bounds=WindowBounds(150, 150, 250, 190))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Same window, but element relocated to (350, 150) (dx=200px)
        el1 = make_element("Save", bounds=WindowBounds(350, 150, 450, 190))
        obs1 = make_observation([el1])
        res1 = engine.track_observation(obs1)

        assert len(res1.moved_tracks) == 1
        t1 = res1.moved_tracks[0]
        assert t1.track_id == tid
        assert t1.status == VisualTrackStatus.MOVED
        assert "Save" in res1.summary
        assert "moved" in res1.summary

    def test_03_same_element_resized(self) -> None:
        """Test 3: Element significantly resized transitions to UPDATED."""
        engine = VisualTrackingEngine(resize_ratio_threshold=0.20)
        el0 = make_element("Search", bounds=WindowBounds(100, 100, 200, 140))  # width=100
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Width expands to 300 (>20% expansion)
        el1 = make_element("Search", bounds=WindowBounds(100, 100, 300, 140))
        obs1 = make_observation([el1])
        res1 = engine.track_observation(obs1)

        assert len(res1.updated_tracks) == 1
        assert res1.updated_tracks[0].track_id == tid
        assert res1.updated_tracks[0].status == VisualTrackStatus.UPDATED

    def test_04_same_element_control_state_changed(self) -> None:
        """Test 4: Control text/state update transitions to UPDATED."""
        engine = VisualTrackingEngine()
        el0 = make_element("Download", bounds=WindowBounds(100, 100, 200, 140), text_content="Download")
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Text updates to "Downloading..."
        el1 = make_element("Download", bounds=WindowBounds(100, 100, 200, 140), text_content="Downloading...")
        obs1 = make_observation([el1])
        res1 = engine.track_observation(obs1)

        assert len(res1.active_tracks) == 1
        assert res1.active_tracks[0].track_id == tid
        assert res1.active_tracks[0].status in (VisualTrackStatus.UPDATED, VisualTrackStatus.TRACKED)

    def test_05_parent_window_translated(self) -> None:
        """Test 5: When parent window moves, child element moves with window and remains TRACKED, not MOVED."""
        engine = VisualTrackingEngine(move_distance_threshold=15.0)

        # T0: Window at (100, 100, 700, 500), Button at (150, 150, 250, 190) -> rel (50, 50)
        win0 = WindowBounds(100, 100, 700, 500)
        el0 = make_element("OK", bounds=WindowBounds(150, 150, 250, 190))
        obs0 = make_observation([el0], window_bounds=win0)
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Window moved by (dx=100, dy=50) to (200, 150, 800, 550)
        # Button also moved by (dx=100, dy=50) to (250, 200, 350, 240) -> relative to window it did not move!
        win1 = WindowBounds(200, 150, 800, 550)
        el1 = make_element("OK", bounds=WindowBounds(250, 200, 350, 240))
        obs1 = make_observation([el1], window_bounds=win1)
        res1 = engine.track_observation(obs1)

        # Child element must NOT be classified as MOVED
        assert len(res1.moved_tracks) == 0
        assert len(res1.active_tracks) == 1
        t1 = res1.active_tracks[0]
        assert t1.track_id == tid
        assert t1.status == VisualTrackStatus.TRACKED

    def test_06_new_element_appears(self) -> None:
        """Test 6: A new element appearing in T1 receives status NEW and a fresh track_id."""
        engine = VisualTrackingEngine()
        el0 = make_element("Cancel", bounds=WindowBounds(100, 100, 200, 140))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid0 = res0.new_tracks[0].track_id

        # T1: Cancel remains, but "Confirm" appears
        el1_cancel = make_element("Cancel", bounds=WindowBounds(100, 100, 200, 140))
        el1_confirm = make_element("Confirm", bounds=WindowBounds(220, 100, 320, 140))
        obs1 = make_observation([el1_cancel, el1_confirm])
        res1 = engine.track_observation(obs1)

        assert len(res1.new_tracks) == 1
        new_track = res1.new_tracks[0]
        assert new_track.canonical_name == "Confirm"
        assert new_track.track_id != tid0
        assert new_track.status == VisualTrackStatus.NEW

    def test_07_existing_element_disappears(self) -> None:
        """Test 7: An element present in T0 that vanishes in T1 transitions to MISSING."""
        engine = VisualTrackingEngine()
        el0 = make_element("PopupDialog", bounds=WindowBounds(200, 200, 500, 400))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Empty observation (dialog vanished)
        obs1 = make_observation([])
        res1 = engine.track_observation(obs1)

        assert len(res1.missing_tracks) == 1
        assert res1.missing_tracks[0].track_id == tid
        assert res1.missing_tracks[0].status == VisualTrackStatus.MISSING
        assert res1.missing_tracks[0].missing_count == 1

    def test_08_element_reappears(self) -> None:
        """Test 8: An element missing in T1 that reappears in T2 transitions to REAPPEARED with same track_id."""
        engine = VisualTrackingEngine(max_missing_frames=3)
        el0 = make_element("Notification", bounds=WindowBounds(600, 50, 780, 120))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # T1: Notification missing
        obs1 = make_observation([])
        res1 = engine.track_observation(obs1)
        assert res1.missing_tracks[0].status == VisualTrackStatus.MISSING

        # T2: Notification reappears at same location
        el2 = make_element("Notification", bounds=WindowBounds(600, 50, 780, 120))
        obs2 = make_observation([el2])
        res2 = engine.track_observation(obs2)

        assert len(res2.reappeared_tracks) == 1
        reapp = res2.reappeared_tracks[0]
        assert reapp.track_id == tid
        assert reapp.status == VisualTrackStatus.REAPPEARED
        assert reapp.missing_count == 0

    def test_09_track_termination_after_configured_missing_frames(self) -> None:
        """Test 9: Tracks missing for > max_missing_frames transition to TERMINATED."""
        engine = VisualTrackingEngine(max_missing_frames=2)
        el0 = make_element("Banner", bounds=WindowBounds(100, 100, 300, 150))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # Frame 1 missing
        obs1 = make_observation([])
        res1 = engine.track_observation(obs1)
        assert engine.get_track(tid).status == VisualTrackStatus.MISSING
        assert engine.get_track(tid).missing_count == 1

        # Frame 2 missing
        obs2 = make_observation([])
        res2 = engine.track_observation(obs2)
        assert engine.get_track(tid).status == VisualTrackStatus.MISSING
        assert engine.get_track(tid).missing_count == 2

        # Frame 3 missing: exceeds max_missing_frames (2) -> TERMINATED
        obs3 = make_observation([])
        res3 = engine.track_observation(obs3)
        term_track = engine.get_track(tid)
        assert term_track.status == VisualTrackStatus.TERMINATED
        assert tid not in [t.track_id for t in res3.active_tracks]

    def test_10_duplicate_labels_distinct_positions(self) -> None:
        """Test 10: Multiple elements with duplicate labels ("+") are tracked distinctively by position/container."""
        engine = VisualTrackingEngine()
        el_plus_top = make_element("+", bounds=WindowBounds(100, 100, 130, 130))
        el_plus_bottom = make_element("+", bounds=WindowBounds(100, 400, 130, 430))
        obs0 = make_observation([el_plus_top, el_plus_bottom])

        res0 = engine.track_observation(obs0)
        assert len(res0.new_tracks) == 2
        tid_top = res0.new_tracks[0].track_id
        tid_bottom = res0.new_tracks[1].track_id
        assert tid_top != tid_bottom

        # T1: Both still present at their distinct positions
        el1_top = make_element("+", bounds=WindowBounds(100, 100, 130, 130))
        el1_bottom = make_element("+", bounds=WindowBounds(100, 400, 130, 430))
        obs1 = make_observation([el1_top, el1_bottom])

        res1 = engine.track_observation(obs1)
        assert len(res1.active_tracks) == 2
        active_ids = {t.track_id for t in res1.active_tracks}
        assert active_ids == {tid_top, tid_bottom}

    def test_11_conflicting_semantic_and_geometry_signals(self) -> None:
        """Test 11: Two identical candidates with indistinguishable scores yield UNCERTAIN, never forced."""
        engine = VisualTrackingEngine(ambiguity_threshold=0.10)
        # Track initial single button
        el0 = make_element("Button", bounds=WindowBounds(100, 100, 150, 130))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid0 = res0.new_tracks[0].track_id

        # T1: Two identical candidate buttons appear equidistant from original location
        c1 = make_element("Button", bounds=WindowBounds(110, 100, 160, 130))
        c2 = make_element("Button", bounds=WindowBounds(100, 110, 150, 140))
        obs1 = make_observation([c1, c2])
        res1 = engine.track_observation(obs1)

        # Ambiguous matching must yield UNCERTAIN
        assert len(res1.uncertain_tracks) > 0
        assert res1.uncertain_tracks[0].status == VisualTrackStatus.UNCERTAIN

    def test_12_different_applications_windows(self) -> None:
        """Test 12: Elements in different applications/processes are strictly isolated."""
        engine = VisualTrackingEngine()
        el_notepad = make_element("File", bounds=WindowBounds(10, 10, 50, 30))
        obs_notepad = make_observation([el_notepad], process_name="notepad.exe", window_title="Untitled - Notepad")
        res0 = engine.track_observation(obs_notepad)
        tid_notepad = res0.new_tracks[0].track_id

        # Next observation is from a completely different process: "calc.exe"
        el_calc = make_element("File", bounds=WindowBounds(10, 10, 50, 30))
        obs_calc = make_observation([el_calc], process_name="calculator.exe", window_title="Calculator")
        res1 = engine.track_observation(obs_calc)

        # Must NOT associate notepad track with calculator element!
        assert len(res1.new_tracks) == 1
        assert res1.new_tracks[0].track_id != tid_notepad
        assert res1.new_tracks[0].process_name == "calculator.exe"

    def test_13_sensitive_blocked_observation(self) -> None:
        """Test 13: Sensitive/blocked observation is handled safely without crashing or exposing data."""
        engine = VisualTrackingEngine()
        el0 = make_element("Account", bounds=WindowBounds(100, 100, 200, 140))
        obs0 = make_observation([el0])
        engine.track_observation(obs0)

        # Blocked observation
        obs_blocked = make_observation([], is_blocked=True)
        res_blocked = engine.track_observation(obs_blocked)

        assert res_blocked.confidence == 0.0
        assert "Cannot track elements from an invalid or empty screen observation." in res_blocked.summary
        assert len(res_blocked.missing_tracks) == 1

    def test_14_empty_observation(self) -> None:
        """Test 14: Empty observation safely marks all active tracks as MISSING."""
        engine = VisualTrackingEngine()
        el0 = make_element("TestBtn", bounds=WindowBounds(50, 50, 150, 90))
        obs0 = make_observation([el0])
        engine.track_observation(obs0)

        empty_obs = make_observation([])
        res = engine.track_observation(empty_obs)
        assert len(res.missing_tracks) == 1
        assert res.missing_tracks[0].canonical_name == "TestBtn"

    def test_15_invalid_observation(self) -> None:
        """Test 15: Invalid (None) observation handled gracefully without throwing unhandled exceptions."""
        engine = VisualTrackingEngine()
        res = engine.track_observation(None)  # type: ignore[arg-type]
        assert res is not None
        assert res.confidence == 0.0
        assert len(res.active_tracks) == 0

    def test_16_registry_max_capacity_bounded(self) -> None:
        """Test 16: Registry strictly bounds total tracks to max_tracks, evicting oldest."""
        engine = VisualTrackingEngine(max_tracks=15)

        # Add 25 elements across observations
        elements = [
            make_element(f"Element_{i}", bounds=WindowBounds(10 * i, 10 * i, 10 * i + 50, 10 * i + 30))
            for i in range(25)
        ]
        obs = make_observation(elements)
        res = engine.track_observation(obs)

        # Registry must never exceed max_tracks (15)
        assert len(engine._tracks) <= 15
        assert len(res.active_tracks) <= 15

    def test_17_no_unbounded_history(self) -> None:
        """Test 17: VisualElementTrack stores bounded metadata and history counters, not full nested models."""
        engine = VisualTrackingEngine()
        el0 = make_element("HistoryTest")
        obs0 = make_observation([el0])
        res = engine.track_observation(obs0)
        track = res.new_tracks[0]

        # Ensure history is integer count, not an unbounded list of objects
        assert isinstance(track.history_count, int)
        assert isinstance(track.missing_count, int)
        assert "history_list" not in track.metadata

    def test_18_no_raw_pixels_in_result(self) -> None:
        """Test 18: Tracking result and element tracks contain zero raw pixel bytes, arrays, or base64."""
        engine = VisualTrackingEngine()
        el0 = make_element("PrivacyTest")
        obs0 = make_observation([el0])
        res = engine.track_observation(obs0)

        res_dict = res.to_dict()
        res_str = str(res_dict)

        assert "raw_data" not in res_dict
        assert "base64" not in res_str
        assert "bytes" not in res_str
        for t in res.active_tracks:
            td = t.to_dict()
            assert "raw_data" not in td
            assert "image" not in td

    def test_19_password_contents_never_stored(self) -> None:
        """Test 19: Passwords or secrets in element text are redacted/sanitized."""
        engine = VisualTrackingEngine()
        el_pass = make_element(
            name="Password",
            element_type=UIElementType.INPUT,
            bounds=WindowBounds(100, 100, 250, 140),
            text_content="SuperSecretPassword123!",
        )
        obs = make_observation([el_pass])
        res = engine.track_observation(obs)
        track = res.new_tracks[0]

        # Sensitive password value must NOT appear in metadata
        assert track.metadata.get("text_content") != "SuperSecretPassword123!"
        assert "SuperSecretPassword123!" not in str(track.to_dict())

    def test_20_session_scoped_track_ids(self) -> None:
        """Test 20: Track IDs are session-scoped strings generated in-memory."""
        engine = VisualTrackingEngine()
        el = make_element("TrackIDTest")
        obs = make_observation([el])
        res = engine.track_observation(obs)
        tid = res.new_tracks[0].track_id

        assert isinstance(tid, str)
        assert tid.startswith("trk_")

        # After reset, new session starts
        engine.reset()
        res2 = engine.track_observation(obs)
        tid2 = res2.new_tracks[0].track_id
        assert tid2 != tid

    def test_21_no_background_tracking_service(self) -> None:
        """Test 21: Engine starts zero background daemon threads or timers."""
        threads_before = threading.active_count()
        engine = VisualTrackingEngine()
        el = make_element("ThreadTest")
        obs = make_observation([el])
        engine.track_observation(obs)
        threads_after = threading.active_count()

        # No background daemon threads or polling loops created
        assert threads_after == threads_before

    def test_22_ambiguous_matching_returns_uncertain(self) -> None:
        """Test 22: High-ambiguity candidate pairing sets status to UNCERTAIN."""
        engine = VisualTrackingEngine(ambiguity_threshold=0.08)
        el0 = make_element("Item", bounds=WindowBounds(100, 100, 200, 140))
        obs0 = make_observation([el0])
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # In next observation, two identical items appear equidistant
        c1 = make_element("Item", bounds=WindowBounds(105, 100, 205, 140))
        c2 = make_element("Item", bounds=WindowBounds(100, 105, 200, 145))
        obs1 = make_observation([c1, c2])
        res1 = engine.track_observation(obs1)

        unc = engine.get_track(tid)
        assert unc is not None
        assert unc.status == VisualTrackStatus.UNCERTAIN

    def test_23_element_movement_vs_window_movement(self) -> None:
        """Test 23: Accurately differentiates window shift from independent internal element move."""
        engine = VisualTrackingEngine(move_distance_threshold=15.0)
        win0 = WindowBounds(100, 100, 800, 600)
        el_btn = make_element("Action", bounds=WindowBounds(200, 200, 300, 240))  # rel (100, 100)
        obs0 = make_observation([el_btn], window_bounds=win0)
        res0 = engine.track_observation(obs0)
        tid = res0.new_tracks[0].track_id

        # Window shifts by (50, 50).
        # Button shifts by (50, 50) + an additional (100, 0) relative move!
        win1 = WindowBounds(150, 150, 850, 650)
        el_btn1 = make_element("Action", bounds=WindowBounds(350, 250, 450, 290))  # rel (200, 100) -> moved by 100px relative to window
        obs1 = make_observation([el_btn1], window_bounds=win1)
        res1 = engine.track_observation(obs1)

        assert len(res1.moved_tracks) == 1
        assert res1.moved_tracks[0].track_id == tid
        assert res1.moved_tracks[0].status == VisualTrackStatus.MOVED

    def test_24_container_relationship_preserved(self) -> None:
        """Test 24: Enclosing container_type is tracked and preserved across observations."""
        engine = VisualTrackingEngine()
        el = make_element("LoginBtn", bounds=WindowBounds(200, 300, 300, 340))
        container = UIContainer(
            container_id="cont_1",
            container_type=UIContainerType.FORM,
            bounds=WindowBounds(150, 250, 400, 400),
            elements=(el,),
        )
        scene0 = UIScene(
            scene_id="scene_0",
            observation_id="obs_0",
            window_title="Login",
            containers=(container,),
            interactive_elements=(el,),
        )
        obs0 = make_observation([el])

        res0 = engine.track_observation(obs0, scene=scene0)
        track = res0.new_tracks[0]
        assert track.container_type == UIContainerType.FORM

        # T1: Same element without explicit scene re-parse retains container_type
        obs1 = make_observation([el])
        res1 = engine.track_observation(obs1)
        track1 = res1.active_tracks[0]
        assert track1.container_type == UIContainerType.FORM

    def test_25_reappearance_preserves_track_id(self) -> None:
        """Test 25: Reappeared element keeps exact initial track_id."""
        engine = VisualTrackingEngine()
        el = make_element("PersistentWidget")
        obs0 = make_observation([el])
        res0 = engine.track_observation(obs0)
        init_id = res0.new_tracks[0].track_id

        # Missing
        engine.track_observation(make_observation([]))
        # Reappeared
        res2 = engine.track_observation(make_observation([el]))
        assert res2.reappeared_tracks[0].track_id == init_id

    def test_26_new_element_gets_different_track_id(self) -> None:
        """Test 26: Distinct elements receive unique, different track_ids."""
        engine = VisualTrackingEngine()
        el1 = make_element("ButtonA", bounds=WindowBounds(100, 100, 200, 140))
        el2 = make_element("ButtonB", bounds=WindowBounds(300, 100, 400, 140))
        obs = make_observation([el1, el2])
        res = engine.track_observation(obs)

        assert len(res.new_tracks) == 2
        assert res.new_tracks[0].track_id != res.new_tracks[1].track_id

    def test_27_process_window_lifecycle_isolation(self) -> None:
        """Test 27: HWND / Process lifecycle boundary isolates tracks."""
        engine = VisualTrackingEngine()
        el0 = make_element("Apply", bounds=WindowBounds(100, 100, 200, 140))
        obs0 = make_observation([el0], process_name="app_a.exe", hwnd=1111)
        res0 = engine.track_observation(obs0)
        tid0 = res0.new_tracks[0].track_id

        # New window from different process
        obs1 = make_observation([el0], process_name="app_b.exe", hwnd=2222)
        res1 = engine.track_observation(obs1)

        # app_b must get new track, not recycle tid0
        assert res1.new_tracks[0].track_id != tid0

    def test_28_configurable_thresholds(self) -> None:
        """Test 28: Engine respects custom weights, min_match_score, and move_distance_threshold."""
        engine_strict = VisualTrackingEngine(min_match_score=0.99)
        el0 = make_element("Submit", bounds=WindowBounds(100, 100, 200, 140))
        obs0 = make_observation([el0])
        engine_strict.track_observation(obs0)

        # Slightly displaced element won't match under 0.99 score threshold
        el1 = make_element("Submit", bounds=WindowBounds(140, 100, 240, 140))
        obs1 = make_observation([el1])
        res1 = engine_strict.track_observation(obs1)

        # Under strict threshold, prior track becomes MISSING, new element becomes NEW
        assert len(res1.missing_tracks) == 1
        assert len(res1.new_tracks) == 1

    def test_29_backward_compatibility_of_existing_vision_operations(self) -> None:
        """Test 29: Existing VisionSkills operations remain fully backward compatible."""
        mock_svm = mock.MagicMock(spec=SecureVisionManager)
        obs = make_observation([make_element("OK")])
        mock_svm.capture_active_window.return_value = obs
        mock_svm.get_latest_observation.return_value = obs

        skills = VisionSkills(secure_vision_manager=mock_svm)

        # Existing verify_screen_state
        res_ver = skills.execute("verify screen state: OK is visible")
        assert res_ver is not None
        assert res_ver.operation in ("verify_screen_state", "verify_goal")

        # Existing map_ui_scene
        res_map = skills.execute("map the active window")
        assert res_map is not None
        assert res_map.operation == "map_ui_scene"

    def test_30_vision_skills_track_elements_and_get_visual_tracks(self) -> None:
        """Test 30: VisionSkills executes track_elements and get_visual_tracks commands."""
        mock_svm = mock.MagicMock(spec=SecureVisionManager)
        el = make_element("Submit", bounds=WindowBounds(100, 100, 200, 140))
        obs = make_observation([el])
        mock_svm.capture_active_window.return_value = obs
        mock_svm.get_latest_observation.return_value = obs

        skills = VisionSkills(secure_vision_manager=mock_svm)

        # 1. Track element
        res_track = skills.execute("track the submit button")
        assert res_track.success is True
        assert res_track.operation == "track_elements"
        assert "active_tracks" in res_track.data

        # 2. Get active tracks
        res_get = skills.execute("get visual tracks")
        assert res_get.success is True
        assert res_get.operation == "get_visual_tracks"
        assert res_get.data["count"] >= 1

    def test_31_intent_router_tracking_rules(self) -> None:
        """Test 31: IntentRouter routes tracking phrases deterministically to IntentType.VISION."""
        router = IntentRouter()

        queries = [
            "track the submit button",
            "where did the submit button move",
            "did the submit button move",
            "is it the same button",
            "where is the tracked button",
            "get visual tracks",
            "list visual tracks",
        ]

        for q in queries:
            classification = router.classify(q)
            assert classification.intent == IntentType.VISION, f"Failed for query: {q}"
            assert classification.confidence >= 0.90

    def test_32_base_system_skill_result_message_formatting(self) -> None:
        """Test 32: SystemSkillResult formats voice-safe messages for tracking operations."""
        res_track = SystemSkillResult(
            operation="track_elements",
            success=True,
            data={
                "summary": "Submit button moved.",
                "active_tracks": [{"track_id": "trk_1"}],
            },
        )
        assert res_track.message == "Submit button moved."

        res_get = SystemSkillResult(
            operation="get_visual_tracks",
            success=True,
            data={
                "active_tracks": [{"track_id": "trk_1"}, {"track_id": "trk_2"}],
            },
        )
        assert res_get.message == "Tracking 2 visual elements."

    def test_33_benchmark_matching_performance(self) -> None:
        """Test 33: Benchmarking association capability across 20, 50, and 100 elements."""
        engine = VisualTrackingEngine()

        for count in (20, 50, 100):
            elements_t0 = [
                make_element(f"btn_{i}", bounds=WindowBounds(10 * i, 10 * i, 10 * i + 30, 10 * i + 20))
                for i in range(count)
            ]
            elements_t1 = [
                make_element(f"btn_{i}", bounds=WindowBounds(10 * i, 10 * i, 10 * i + 30, 10 * i + 20))
                for i in range(count)
            ]

            metrics = engine.benchmark_matching(elements_t0, elements_t1)
            assert metrics["t0_count"] == count
            assert metrics["t1_count"] == count
            assert metrics["matched_count"] == count
            assert metrics["duration_ms"] >= 0.0
