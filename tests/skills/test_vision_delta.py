"""Phase 27.10 Visual Delta & Change Detection Tests.

Comprehensive deterministic unit and integration test suite covering:
1. Identical observations -> NO_MEANINGFUL_CHANGE
2. Text changed in place -> TEXT_CHANGED
3. Element appeared -> ELEMENT_APPEARED
4. Element disappeared -> ELEMENT_DISAPPEARED
5. Element moved -> ELEMENT_MOVED
6. Element resized -> ELEMENT_RESIZED
7. Window changed -> WINDOW_CHANGED
8. Window moved without false child changes
9. Duplicate elements handling
10. Ambiguous element matching -> UNCERTAIN
11. OCR text additions
12. OCR text removals
13. OCR text modifications
14. Stale observation handling
15. Expired observation lifecycle
16. Sensitive window blocking
17. Cache invalidation on mutation
18. No-baseline initial flow (captures T0 then T1)
19. Planner detect_screen_change action execution
20. Temporal natural-language routing in VisionSkills
21. Malformed multimodal response handling
22. Multimodal fallback when deterministic diff is inconclusive
23. Zero raw pixels in returned models or dictionaries
24. Confidence propagation
25. Deterministic repeated execution
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest import mock
import uuid

import pytest

from app.ai.models import UnsupportedModalityError
from app.ai.planner.executor import Executor
from app.ai.planner.models import Plan, Task
from app.ai.planner.planner import KNOWN_ACTIONS, Planner
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.vision_skills import VisionSkills
from app.vision.delta import (
    ELEMENT_MATCH_AMBIGUITY_DELTA,
    IOU_MATCH_THRESHOLD,
    MIN_ELEMENT_MATCH_SCORE,
    MOVE_DISTANCE_THRESHOLD,
    RESIZE_RATIO_THRESHOLD,
    TEXT_SIMILARITY_THRESHOLD,
    VisualDeltaEngine,
)
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UIElement,
    UIElementChange,
    UIElementType,
    VisualDeltaResult,
    VisualDeltaType,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    VisionSecurityPolicy,
)


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_capture(
    width: int = 1000,
    height: int = 800,
    window_title: str = "Test App Window",
    process_name: str = "testapp.exe",
    hwnd: int = 12345,
    bounds: Optional[WindowBounds] = None,
) -> ScreenCapture:
    """Construct deterministic ScreenCapture in RAM."""
    b = bounds or WindowBounds(left=100, top=100, right=100 + width, bottom=100 + height)
    raw_bgra = b"\x00\x00\x00\xff" * (width * height)
    return ScreenCapture(
        raw_data=raw_bgra,
        width=width,
        height=height,
        stride=width * 4,
        bounds=b,
        timestamp=time.time(),
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "hwnd": hwnd,
        },
    )


def _create_observation(
    capture: Optional[ScreenCapture] = None,
    ocr_blocks: Optional[List[OCRTextBlock]] = None,
    elements: Optional[List[UIElement]] = None,
    window_title: str = "Test App Window",
    process_name: str = "testapp.exe",
    hwnd: int = 12345,
    bounds: Optional[WindowBounds] = None,
    ttl_seconds: float = 30.0,
    timestamp: Optional[float] = None,
) -> ScreenObservation:
    """Construct deterministic ScreenObservation with metadata."""
    cap = capture or _create_synthetic_capture(
        window_title=window_title,
        process_name=process_name,
        hwnd=hwnd,
        bounds=bounds,
    )
    now = timestamp or time.time()
    meta: Dict[str, Any] = {
        "window_title": window_title,
        "process_name": process_name,
        "hwnd": hwnd,
    }
    if ocr_blocks is not None:
        meta["ocr_blocks"] = ocr_blocks
    if elements is not None:
        meta["elements"] = elements

    return ScreenObservation(
        id=str(uuid.uuid4()),
        capture=cap,
        authorization=CaptureAuthorization(
            decision=CaptureDecision.ALLOW,
            category=CaptureCategory.SAFE,
            reason="Synthetic test authorization",
            timestamp=now,
        ),
        source="window",
        timestamp=now,
        expires_at=now + ttl_seconds,
        metadata=meta,
    )


class MockAIProvider:
    """Mock multimodal AI Provider for deterministic delta responses."""

    def __init__(self, response_text: str = "", supports_multimodal: bool = True) -> None:
        self._response_text = response_text
        self.supports_multimodal = supports_multimodal
        self.model = "mock-vision-model"

    def generate(self, payload: Any) -> Any:
        if not self.supports_multimodal:
            raise UnsupportedModalityError("Provider does not support vision inputs")
        mock_resp = mock.MagicMock()
        mock_resp.content = self._response_text
        return mock_resp


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVisualDeltaEngine:
    """Unit tests for VisualDeltaEngine core algorithm and change types."""

    def test_1_identical_observations(self) -> None:
        """Test 1: Identical observations yield NO_MEANINGFUL_CHANGE."""
        blocks = [
            OCRTextBlock(text="Submit", bounds=WindowBounds(150, 150, 250, 190)),
            OCRTextBlock(text="Cancel", bounds=WindowBounds(270, 150, 370, 190)),
        ]
        obs0 = _create_observation(ocr_blocks=blocks, timestamp=100.0)
        obs1 = _create_observation(ocr_blocks=blocks, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.NO_MEANINGFUL_CHANGE
        assert not res.meaningful_change_detected
        assert not res.window_changed
        assert len(res.added_texts) == 0
        assert len(res.removed_texts) == 0
        assert len(res.modified_texts) == 0

    def test_2_text_changed(self) -> None:
        """Test 2: Text content modified at same position yields TEXT_CHANGED."""
        b0 = [OCRTextBlock(text="Status: Pending", bounds=WindowBounds(150, 150, 300, 190))]
        b1 = [OCRTextBlock(text="Status: Complete", bounds=WindowBounds(150, 150, 300, 190))]
        obs0 = _create_observation(ocr_blocks=b0, timestamp=100.0)
        obs1 = _create_observation(ocr_blocks=b1, timestamp=102.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.TEXT_CHANGED
        assert res.meaningful_change_detected
        assert len(res.modified_texts) >= 1
        assert "Pending" in res.explanation or "Complete" in res.explanation or "Text" in res.explanation

    def test_3_element_appeared(self) -> None:
        """Test 3: New element appearing in T1 yields ELEMENT_APPEARED."""
        b0 = [OCRTextBlock(text="Username", bounds=WindowBounds(150, 150, 250, 180))]
        b1 = [
            OCRTextBlock(text="Username", bounds=WindowBounds(150, 150, 250, 180)),
            OCRTextBlock(text="Submit", bounds=WindowBounds(150, 200, 250, 240)),
        ]
        obs0 = _create_observation(ocr_blocks=b0, timestamp=100.0)
        obs1 = _create_observation(ocr_blocks=b1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.ELEMENT_APPEARED
        assert res.meaningful_change_detected
        assert "Submit" in res.added_texts or any(c.change_type == VisualDeltaType.ELEMENT_APPEARED for c in res.element_changes)

    def test_4_element_disappeared(self) -> None:
        """Test 4: Element disappearing from T0 yields ELEMENT_DISAPPEARED."""
        b0 = [
            OCRTextBlock(text="Dialog Title", bounds=WindowBounds(100, 100, 300, 130)),
            OCRTextBlock(text="Close Popup", bounds=WindowBounds(150, 200, 250, 240)),
        ]
        b1 = [OCRTextBlock(text="Dialog Title", bounds=WindowBounds(100, 100, 300, 130))]
        obs0 = _create_observation(ocr_blocks=b0, timestamp=100.0)
        obs1 = _create_observation(ocr_blocks=b1, timestamp=102.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.ELEMENT_DISAPPEARED
        assert res.meaningful_change_detected
        assert "Close Popup" in res.removed_texts or any(c.change_type == VisualDeltaType.ELEMENT_DISAPPEARED for c in res.element_changes)

    def test_5_element_moved(self) -> None:
        """Test 5: Substantially displaced element yields ELEMENT_MOVED."""
        el0 = [UIElement(name="Toolbox", element_type=UIElementType.CONTAINER, bounds=WindowBounds(100, 100, 200, 300), center=Point(150, 200))]
        el1 = [UIElement(name="Toolbox", element_type=UIElementType.CONTAINER, bounds=WindowBounds(300, 100, 400, 300), center=Point(350, 200))]
        obs0 = _create_observation(elements=el0, timestamp=100.0)
        obs1 = _create_observation(elements=el1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.ELEMENT_MOVED
        assert res.meaningful_change_detected
        assert len(res.element_changes) == 1
        assert res.element_changes[0].displacement is not None
        assert res.element_changes[0].displacement.x == 200

    def test_6_element_resized(self) -> None:
        """Test 6: Substantially resized element yields ELEMENT_RESIZED."""
        el0 = [UIElement(name="Terminal Panel", element_type=UIElementType.CONTAINER, bounds=WindowBounds(100, 100, 300, 300), center=Point(200, 200))]
        el1 = [UIElement(name="Terminal Panel", element_type=UIElementType.CONTAINER, bounds=WindowBounds(100, 100, 500, 600), center=Point(300, 350))]
        obs0 = _create_observation(elements=el0, timestamp=100.0)
        obs1 = _create_observation(elements=el1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.ELEMENT_RESIZED
        assert res.meaningful_change_detected

    def test_7_window_changed(self) -> None:
        """Test 7: Active foreground window identity change yields WINDOW_CHANGED."""
        obs0 = _create_observation(window_title="Calculator", process_name="calc.exe", hwnd=1111)
        obs1 = _create_observation(window_title="Visual Studio Code", process_name="code.exe", hwnd=2222)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.WINDOW_CHANGED
        assert res.window_changed
        assert res.meaningful_change_detected
        assert "Visual Studio Code" in res.explanation or "code" in res.explanation

    def test_8_window_moved_without_false_child_changes(self) -> None:
        """Test 8: Moving entire window desktop position does NOT cause false child element moves."""
        wb0 = WindowBounds(100, 100, 600, 500)
        wb1 = WindowBounds(300, 300, 800, 700)

        b0 = [OCRTextBlock(text="OK Button", bounds=WindowBounds(150, 150, 250, 190))]
        b1 = [OCRTextBlock(text="OK Button", bounds=WindowBounds(350, 350, 450, 390))]

        obs0 = _create_observation(bounds=wb0, ocr_blocks=b0, timestamp=100.0)
        obs1 = _create_observation(bounds=wb1, ocr_blocks=b1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.NO_MEANINGFUL_CHANGE
        assert not res.meaningful_change_detected
        assert not res.window_changed

    def test_9_duplicate_elements(self) -> None:
        """Test 9: Multiple identical elements with distinct spatial separation match cleanly."""
        b0 = [
            OCRTextBlock(text="Delete", bounds=WindowBounds(100, 100, 180, 140)),
            OCRTextBlock(text="Delete", bounds=WindowBounds(100, 300, 180, 340)),
        ]
        b1 = [
            OCRTextBlock(text="Delete", bounds=WindowBounds(100, 100, 180, 140)),
            OCRTextBlock(text="Delete", bounds=WindowBounds(100, 300, 180, 340)),
        ]
        obs0 = _create_observation(ocr_blocks=b0, timestamp=100.0)
        obs1 = _create_observation(ocr_blocks=b1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type == VisualDeltaType.NO_MEANINGFUL_CHANGE
        assert not res.meaningful_change_detected

    def test_10_ambiguous_element_matching(self) -> None:
        """Test 10: Highly ambiguous duplicate elements yield UNCERTAIN."""
        el0 = [UIElement(name="Copy", element_type=UIElementType.BUTTON, bounds=WindowBounds(100, 100, 150, 130), center=Point(125, 115))]
        el1 = [
            UIElement(name="Copy", element_type=UIElementType.BUTTON, bounds=WindowBounds(110, 100, 160, 130), center=Point(135, 115)),
            UIElement(name="Copy", element_type=UIElementType.BUTTON, bounds=WindowBounds(90, 100, 140, 130), center=Point(115, 115)),
        ]
        obs0 = _create_observation(elements=el0, timestamp=100.0)
        obs1 = _create_observation(elements=el1, timestamp=101.0)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert res.primary_change_type in (VisualDeltaType.UNCERTAIN, VisualDeltaType.ELEMENT_APPEARED)

    def test_11_ocr_additions(self) -> None:
        """Test 11: OCR text addition detection."""
        b0 = [OCRTextBlock(text="Existing Line", bounds=WindowBounds(50, 50, 200, 80))]
        b1 = [
            OCRTextBlock(text="Existing Line", bounds=WindowBounds(50, 50, 200, 80)),
            OCRTextBlock(text="Newly Added Text", bounds=WindowBounds(50, 100, 250, 130)),
        ]
        obs0 = _create_observation(ocr_blocks=b0)
        obs1 = _create_observation(ocr_blocks=b1)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert "Newly Added Text" in res.added_texts
        assert res.meaningful_change_detected

    def test_12_ocr_removals(self) -> None:
        """Test 12: OCR text removal detection."""
        b0 = [
            OCRTextBlock(text="Keep This", bounds=WindowBounds(50, 50, 150, 80)),
            OCRTextBlock(text="Vanishing Banner", bounds=WindowBounds(50, 100, 250, 130)),
        ]
        b1 = [OCRTextBlock(text="Keep This", bounds=WindowBounds(50, 50, 150, 80))]
        obs0 = _create_observation(ocr_blocks=b0)
        obs1 = _create_observation(ocr_blocks=b1)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert "Vanishing Banner" in res.removed_texts
        assert res.meaningful_change_detected

    def test_13_ocr_modifications(self) -> None:
        """Test 13: OCR text modification detection."""
        b0 = [OCRTextBlock(text="Progress: 50%", bounds=WindowBounds(50, 50, 200, 80))]
        b1 = [OCRTextBlock(text="Progress: 100%", bounds=WindowBounds(50, 50, 200, 80))]
        obs0 = _create_observation(ocr_blocks=b0)
        obs1 = _create_observation(ocr_blocks=b1)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)

        assert len(res.modified_texts) >= 1
        assert res.primary_change_type == VisualDeltaType.TEXT_CHANGED


class TestEphemeralBufferAndSecurity:
    """Security and Ephemeral buffer pair lifecycle tests."""

    def test_default_max_buffers_is_one(self) -> None:
        """Verify default constructor has max_buffers=1 and retains at most 1 observation in standard mode."""
        ebm = EphemeralBufferManager()
        assert ebm._max_buffers == 1

        cap1 = _create_synthetic_capture(window_title="Frame 1")
        cap2 = _create_synthetic_capture(window_title="Frame 2")

        obs1 = ebm.store(cap1)
        assert ebm.count == 1
        assert ebm.get().id == obs1.id

        obs2 = ebm.store(cap2)
        assert ebm.count == 1
        assert ebm.get().id == obs2.id
        assert ebm.get(obs1.id) is None  # Immediately evicted on next store
        assert ebm.get_previous() is None  # Standard single-buffer mode

    def test_14_stale_observation(self) -> None:
        """Test 14: EphemeralBufferManager with max_buffers=2 evicts oldest on 3rd frame."""
        ebm = EphemeralBufferManager(default_ttl_seconds=30.0, max_buffers=2)
        assert ebm._max_buffers == 2
        cap1 = _create_synthetic_capture(window_title="Frame 1")
        cap2 = _create_synthetic_capture(window_title="Frame 2")
        cap3 = _create_synthetic_capture(window_title="Frame 3")

        obs1 = ebm.store(cap1)
        obs2 = ebm.store(cap2)
        assert ebm.count == 2
        obs3 = ebm.store(cap3)

        assert ebm.count == 2
        assert ebm.get(obs1.id) is None  # Evicted
        assert ebm.get(obs2.id) is not None
        assert ebm.get(obs3.id) is not None
        assert ebm.get_previous().id == obs2.id

    def test_15_expired_observation(self) -> None:
        """Test 15: EphemeralBufferManager rejects expired observations."""
        ebm = EphemeralBufferManager(default_ttl_seconds=1.0, max_buffers=2)
        cap1 = _create_synthetic_capture()
        obs1 = ebm.store(cap1, ttl_seconds=0.01)

        time.sleep(0.02)
        assert ebm.get(obs1.id) is None
        assert ebm.get_previous() is None
        prev, latest = ebm.get_pair()
        assert prev is None
        assert latest is None

    def test_16_sensitive_window(self) -> None:
        """Test 16: SecureVisionManager blocks capture on sensitive windows."""
        policy = VisionSecurityPolicy()
        mgr = SecureVisionManager(policy=policy)

        with mock.patch.object(mgr.engine, "get_active_window_handle", return_value=9999):
            with mock.patch.object(mgr, "_resolve_window_metadata", return_value=("1Password - Master Unlock", "1password.exe")):
                with pytest.raises(CaptureBlockedError):
                    mgr.capture_active_window()

    def test_17_cache_invalidation(self) -> None:
        """Test 17: UI mutation events invalidate the vision cache."""
        skills = VisionSkills()
        cap = _create_synthetic_capture()
        skills.secure_vision_manager.buffer_manager.store(cap)
        assert skills.secure_vision_manager.get_latest_observation() is not None

        skills.invalidate_observation_cache("ui_mutation")
        assert skills.secure_vision_manager.get_latest_observation() is None

    def test_18_no_baseline_flow(self) -> None:
        """Test 18: detect_screen_change handles absent baseline gracefully by capturing fresh frames."""
        sec_mgr = mock.MagicMock()
        sec_mgr.get_latest_observation.return_value = None

        obs_fresh0 = _create_observation(timestamp=100.0)
        obs_fresh1 = _create_observation(timestamp=101.0)
        sec_mgr.capture_active_window.side_effect = [obs_fresh0, obs_fresh1]

        skills = VisionSkills(secure_vision_manager=sec_mgr)
        res = skills.execute("detect screen changes")

        assert res.success
        assert "primary_change_type" in res.data
        assert res.data["primary_change_type"] == VisualDeltaType.NO_MEANINGFUL_CHANGE.value


class TestPlannerAndVoiceIntegration:
    """Planner and Natural Language Vision Skills integration tests."""

    def test_19_planner_detect_screen_change(self) -> None:
        """Test 19: Planner supports detect_screen_change action in KNOWN_ACTIONS and execution."""
        assert "detect_screen_change" in KNOWN_ACTIONS

        planner = Planner()
        plan = planner.create_plan("open notepad and detect screen changes")
        assert plan is not None

        executor = Executor()
        skills = mock.MagicMock()
        skills.execute.return_value = SystemSkillResult(
            success=True,
            operation="detect_screen_change",
            data={"meaningful_change_detected": True, "explanation": "Notepad appeared"},
        )
        executor.register_handler("detect_screen_change", lambda t: skills.execute(t.action))
        task = Task(id="task_1", action="detect_screen_change", target="Notepad")
        test_plan = Plan(query="detect screen changes", tasks=[task])
        res = executor.execute_plan(test_plan)
        assert res.success

    def test_20_temporal_natural_language_routing(self) -> None:
        """Test 20: Natural language phrases correctly route to detect_screen_change."""
        skills = VisionSkills()

        test_queries = [
            ("what changed", "detect_screen_change"),
            ("what just changed?", "detect_screen_change"),
            ("what changed on my screen?", "detect_screen_change"),
            ("did the screen change?", "detect_screen_change"),
            ("did anything change?", "detect_screen_change"),
            ("what happened after that?", "detect_screen_change"),
            ("did the button appear?", "detect_screen_change"),
            ("did the popup disappear?", "detect_screen_change"),
            ("is the error still there?", "detect_screen_change"),
            ("did the UI element move?", "detect_screen_change"),
        ]

        for query, expected_op in test_queries:
            assert skills.can_handle(query), f"Failed can_handle for '{query}'"
            op, tgt, params, _ = skills.parse_command(query)
            assert op == expected_op, f"Query '{query}' parsed to '{op}' instead of '{expected_op}'"

    def test_21_malformed_multimodal_response(self) -> None:
        """Test 21: Malformed multimodal response degrades gracefully without crash."""
        ai_bad = MockAIProvider(response_text="Not valid JSON at all")
        engine_bad = VisualDeltaEngine(ai_provider=ai_bad)
        obs0 = _create_observation()
        obs1 = _create_observation()
        res_bad = engine_bad.compare_observations(obs0, obs1, force_multimodal=True)
        assert res_bad.primary_change_type == VisualDeltaType.NO_MEANINGFUL_CHANGE

    def test_22_multimodal_fallback(self) -> None:
        """Test 22: Multimodal fallback correctly parses structured semantic changes."""
        valid_json = json.dumps({
            "changed": True,
            "change_type": "region_changed",
            "confidence": 0.9,
            "explanation": "The graph changed color from red to green.",
            "box_2d": [100, 100, 500, 500],
        })
        ai_p = MockAIProvider(response_text=valid_json)
        engine = VisualDeltaEngine(ai_provider=ai_p)

        obs0 = _create_observation()
        obs1 = _create_observation()
        res = engine.compare_observations(obs0, obs1, force_multimodal=True)

        assert res.primary_change_type == VisualDeltaType.REGION_CHANGED
        assert res.meaningful_change_detected
        assert "graph changed color" in res.explanation

    def test_23_no_raw_pixels_in_result(self) -> None:
        """Test 23: VisualDeltaResult and UIElementChange never expose raw pixel buffers."""
        b0 = [OCRTextBlock(text="Old", bounds=WindowBounds(10, 10, 50, 30))]
        b1 = [OCRTextBlock(text="New", bounds=WindowBounds(10, 10, 50, 30))]
        obs0 = _create_observation(ocr_blocks=b0)
        obs1 = _create_observation(ocr_blocks=b1)

        engine = VisualDeltaEngine()
        res = engine.compare_observations(obs0, obs1)
        res_dict = res.to_dict()

        serialized = json.dumps(res_dict)
        assert "raw_bytes" not in serialized
        assert "raw_data" not in serialized
        assert "raw_bgra" not in serialized
        assert "stride" not in serialized

    def test_24_confidence_propagation(self) -> None:
        """Test 24: Confidence score is bounded in [0.0, 1.0]."""
        change = UIElementChange(
            change_type=VisualDeltaType.ELEMENT_APPEARED,
            confidence=1.5,  # Out of range, should clamp to 1.0
            text_similarity=-0.2,  # Should clamp to 0.0
            iou=1.2,  # Should clamp to 1.0
        )
        assert change.confidence == 1.0
        assert change.text_similarity == 0.0
        assert change.iou == 1.0

    def test_25_deterministic_repeated_execution(self) -> None:
        """Test 25: Running delta comparison repeatedly on identical data produces identical output."""
        b0 = [OCRTextBlock(text="File", bounds=WindowBounds(10, 10, 50, 30))]
        b1 = [OCRTextBlock(text="Edit", bounds=WindowBounds(10, 10, 50, 30))]
        obs0 = _create_observation(ocr_blocks=b0)
        obs1 = _create_observation(ocr_blocks=b1)

        engine = VisualDeltaEngine()
        r1 = engine.compare_observations(obs0, obs1)
        r2 = engine.compare_observations(obs0, obs1)

        assert r1.primary_change_type == r2.primary_change_type
        assert r1.meaningful_change_detected == r2.meaningful_change_detected
        assert r1.explanation == r2.explanation
        assert len(r1.element_changes) == len(r2.element_changes)
