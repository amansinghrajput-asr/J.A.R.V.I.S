"""Phase 27.11 Visual UI Scene Parsing & Interactive Element Mapping Tests.

Comprehensive deterministic unit and integration test suite covering:
1. Basic scene parsing (header, content, status bar)
2. Form container and label-to-input field association
3. Sidebar navigation container detection
4. Dialog modal container detection
5. Toolbar container detection
6. Tab panel container detection
7. Table/grid container detection
8. Interactive widget inventory & conservative classification
9. Label to input right-side adjacency
10. Label to input below-label adjacency
11. Input without label (retained in interactive elements)
12. Label without input (retained without hallucinating input)
13. Duplicate labels conservative handling
14. Ambiguous container defaults to CONTENT_AREA / UNCERTAIN
15. Malformed OCR geometry handling
16. Empty / minimal scene handling
17. Sensitive window blocking
18. Cache reuse on unchanged window
19. Cache invalidation on UI mutation
20. VLM unavailable fallback (100% deterministic success)
21. Malformed VLM response fallback
22. Successful VLM enhancement merge
23. Zero raw pixels in UIScene, UIContainer, FormField
24. Confidence range clamping [0.0, 1.0]
25. Deterministic repeated execution
26. Planner map_ui_scene action planning and execution
27. Natural language command routing
28. Container filtering parameter
29. Asynchronous scene parsing
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest import mock
import uuid

import pytest

from app.ai.intent_router import IntentRouter, IntentType
from app.ai.planner.executor import Executor
from app.ai.planner.models import Plan, Task
from app.ai.planner.planner import KNOWN_ACTIONS, Planner
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.vision_skills import VisionSkills
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    FormField,
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.scene import VisualSceneParser
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
        source="window",
        timestamp=time.time(),
        bounds=b,
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "hwnd": hwnd,
        },
    )


def _create_observation(
    capture: Optional[ScreenCapture] = None,
    elements: Optional[List[UIElement]] = None,
    ocr_blocks: Optional[List[OCRTextBlock]] = None,
    window_title: str = "Test App Window",
    process_name: str = "testapp.exe",
    bounds: Optional[WindowBounds] = None,
    timestamp: Optional[float] = None,
    ttl: float = 30.0,
) -> ScreenObservation:
    """Construct deterministic ScreenObservation with structured metadata."""
    cap = capture or _create_synthetic_capture(
        window_title=window_title,
        process_name=process_name,
        bounds=bounds,
    )
    meta: Dict[str, Any] = {
        "hwnd": 12345,
        "window_title": window_title,
        "process_name": process_name,
        "bounds": cap.bounds.to_tuple() if cap and cap.bounds else None,
    }
    if elements is not None:
        meta["elements"] = elements
    if ocr_blocks is not None:
        meta["ocr_blocks"] = ocr_blocks

    now = timestamp if timestamp is not None else time.time()
    return ScreenObservation(
        id=str(uuid.uuid4()),
        capture=cap,
        authorization=CaptureAuthorization(
            decision=CaptureDecision.ALLOW,
            category=CaptureCategory.SAFE,
            reason="Test allowed",
            timestamp=now,
        ),
        source="window",
        timestamp=now,
        expires_at=now + ttl,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Test Suite: Visual UI Scene Parsing & Interactive Element Mapping
# ---------------------------------------------------------------------------


class TestVisualSceneParsing:
    """Core deterministic scene parsing tests."""

    def test_1_basic_scene_parsing(self) -> None:
        """Test 1: Parses basic window into header, main content, and status bar containers."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="My Application v1.0", bounds=WindowBounds(20, 10, 250, 40)),
            OCRTextBlock(text="Welcome to the dashboard.", bounds=WindowBounds(50, 200, 350, 240)),
            OCRTextBlock(text="Status: Ready", bounds=WindowBounds(20, 750, 150, 780)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, window_title="My Application")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert scene.window_title == "My Application"
        assert len(scene.containers) >= 2
        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.HEADER in container_types
        assert UIContainerType.STATUS_BAR in container_types
        assert scene.confidence > 0.8
        assert "My Application" in scene.summary

    def test_2_form_container_and_field_association(self) -> None:
        """Test 2: Form fields binding label-to-input and forming a FORM container."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Username:", bounds=WindowBounds(50, 100, 150, 130)),
            OCRTextBlock(text="Password:*", bounds=WindowBounds(50, 160, 150, 190)),
            OCRTextBlock(text="Login", bounds=WindowBounds(100, 240, 200, 280)),
        ]
        elements = [
            UIElement(name="Username Input", element_type=UIElementType.INPUT, bounds=WindowBounds(170, 95, 400, 135), center=Point(285, 115)),
            UIElement(name="Password Input", element_type=UIElementType.INPUT, bounds=WindowBounds(170, 155, 400, 195), center=Point(285, 175)),
            UIElement(name="Login", element_type=UIElementType.BUTTON, bounds=WindowBounds(100, 240, 200, 280), center=Point(150, 260)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, elements=elements, window_title="Login Form")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert len(scene.form_fields) == 2
        labels = [f.label for f in scene.form_fields]
        assert "Username:" in labels
        assert "Password:" in labels

        # Verify password is required due to trailing asterisk
        pw_field = next(f for f in scene.form_fields if "Password" in f.label)
        assert pw_field.is_required

        # Verify FORM container was created
        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.FORM in container_types

    def test_3_sidebar_container_detection(self) -> None:
        """Test 3: Detects vertical navigation list as a SIDEBAR container."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Home", bounds=WindowBounds(20, 150, 120, 180)),
            OCRTextBlock(text="Analytics", bounds=WindowBounds(20, 200, 120, 230)),
            OCRTextBlock(text="Settings", bounds=WindowBounds(20, 250, 120, 280)),
            OCRTextBlock(text="Main Chart View", bounds=WindowBounds(400, 300, 700, 350)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, window_title="Analytics Portal")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.SIDEBAR in container_types

    def test_4_dialog_modal_container_detection(self) -> None:
        """Test 4: Detects modal dialog with OK/Cancel buttons as DIALOG container."""
        wb = WindowBounds(left=200, top=200, right=600, bottom=500)
        blocks = [
            OCRTextBlock(text="Confirm File Deletion", bounds=WindowBounds(220, 220, 450, 250)),
            OCRTextBlock(text="Are you sure you want to delete this file?", bounds=WindowBounds(220, 280, 550, 320)),
            OCRTextBlock(text="OK", bounds=WindowBounds(350, 420, 420, 460)),
            OCRTextBlock(text="Cancel", bounds=WindowBounds(450, 420, 540, 460)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, window_title="Delete Confirmation Dialog")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.DIALOG in container_types

    def test_5_toolbar_container_detection(self) -> None:
        """Test 5: Detects horizontal row of action buttons as TOOLBAR container."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        elements = [
            UIElement(name="Cut", element_type=UIElementType.BUTTON, bounds=WindowBounds(50, 80, 100, 110), center=Point(75, 95)),
            UIElement(name="Copy", element_type=UIElementType.BUTTON, bounds=WindowBounds(110, 80, 160, 110), center=Point(135, 95)),
            UIElement(name="Paste", element_type=UIElementType.BUTTON, bounds=WindowBounds(170, 80, 220, 110), center=Point(195, 95)),
        ]
        obs = _create_observation(bounds=wb, elements=elements, window_title="Text Editor")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.TOOLBAR in container_types

    def test_6_tab_panel_container_detection(self) -> None:
        """Test 6: Detects tab navigation elements as TAB_PANEL container."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        elements = [
            UIElement(name="General Tab", element_type=UIElementType.TAB, bounds=WindowBounds(50, 60, 150, 90), center=Point(100, 75)),
            UIElement(name="Security Tab", element_type=UIElementType.TAB, bounds=WindowBounds(160, 60, 260, 90), center=Point(210, 75)),
        ]
        obs = _create_observation(bounds=wb, elements=elements, window_title="Options")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.TAB_PANEL in container_types

    def test_7_interactive_widget_classification_conservative(self) -> None:
        """Test 7: Classifies interactive controls conservatively without false positives."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Submit", bounds=WindowBounds(100, 100, 200, 140)),
            OCRTextBlock(text="[x] Remember me", bounds=WindowBounds(100, 160, 250, 190)),
            OCRTextBlock(text="https://github.com", bounds=WindowBounds(100, 210, 300, 240)),
            OCRTextBlock(text="General paragraph text about company history.", bounds=WindowBounds(100, 300, 600, 400)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        elem_types = [e.element_type for e in scene.interactive_elements]
        assert UIElementType.BUTTON in elem_types
        assert UIElementType.CHECKBOX in elem_types
        assert UIElementType.LINK in elem_types
        # General paragraph text must NOT be in interactive_elements
        assert not any("General paragraph" in e.name for e in scene.interactive_elements)

    def test_8_label_to_input_below_adjacent(self) -> None:
        """Test 8: Associates label placed directly above an input control."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Search:", bounds=WindowBounds(50, 100, 150, 125)),
        ]
        elements = [
            UIElement(name="Search Box", element_type=UIElementType.INPUT, bounds=WindowBounds(50, 135, 300, 175), center=Point(175, 155)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, elements=elements)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert len(scene.form_fields) == 1
        assert scene.form_fields[0].label == "Search:"
        assert scene.form_fields[0].input_element.name == "Search Box"

    def test_9_input_without_label(self) -> None:
        """Test 9: Orphan input widget without label is preserved in interactive_elements."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        elements = [
            UIElement(name="Standalone Input", element_type=UIElementType.INPUT, bounds=WindowBounds(100, 100, 300, 140), center=Point(200, 120)),
        ]
        obs = _create_observation(bounds=wb, elements=elements)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert len(scene.interactive_elements) == 1
        assert len(scene.form_fields) == 0  # No label fabricated

    def test_10_label_without_input(self) -> None:
        """Test 10: Label without matching input does not fabricate an input."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Note: Application will restart.", bounds=WindowBounds(50, 100, 350, 130)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert len(scene.form_fields) == 0
        assert len(scene.interactive_elements) == 0

    def test_11_duplicate_labels_handled_conservatively(self) -> None:
        """Test 11: Duplicate labels with multiple inputs resolve cleanly by proximity."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Port:", bounds=WindowBounds(50, 100, 120, 130)),
            OCRTextBlock(text="Port:", bounds=WindowBounds(50, 200, 120, 230)),
        ]
        elements = [
            UIElement(name="HTTP Port", element_type=UIElementType.INPUT, bounds=WindowBounds(140, 95, 250, 135), center=Point(195, 115)),
            UIElement(name="HTTPS Port", element_type=UIElementType.INPUT, bounds=WindowBounds(140, 195, 250, 235), center=Point(195, 215)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks, elements=elements)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert len(scene.form_fields) == 2
        assigned_inputs = {f.input_element.name for f in scene.form_fields}
        assert "HTTP Port" in assigned_inputs
        assert "HTTPS Port" in assigned_inputs

    def test_12_ambiguous_container_defaults_to_content_area(self) -> None:
        """Test 12: Unstructured floating elements default to CONTENT_AREA without hallucination."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Random Body Text Fragment", bounds=WindowBounds(300, 300, 500, 340)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        container_types = [c.container_type for c in scene.containers]
        assert UIContainerType.CONTENT_AREA in container_types

    def test_13_malformed_ocr_geometry(self) -> None:
        """Test 13: Handles empty or inverted OCR bounding boxes gracefully."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="Invalid Box", bounds=WindowBounds(100, 100, 50, 50)),  # Inverted
            OCRTextBlock(text="Zero Box", bounds=WindowBounds(0, 0, 0, 0)),  # Empty
            OCRTextBlock(text="Valid Text", bounds=WindowBounds(100, 100, 300, 140)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert scene is not None
        assert "Valid Text" in scene.summary or len(scene.containers) >= 1

    def test_14_empty_or_minimal_scene(self) -> None:
        """Test 14: Blank or minimal window parses safely without exceptions."""
        obs = _create_observation(ocr_blocks=[], elements=[], window_title="Blank App")

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)

        assert scene.window_title == "Blank App"
        assert len(scene.containers) == 0
        assert len(scene.interactive_elements) == 0
        assert "no distinct interactive controls" in scene.summary

    def test_15_async_scene_parsing(self) -> None:
        """Test 15: Asynchronous parse_scene_async runs without blocking."""
        obs = _create_observation()
        parser = VisualSceneParser()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            scene = loop.run_until_complete(parser.parse_scene_async(obs))
            assert scene is not None
            assert scene.scene_id is not None
        finally:
            loop.close()


class TestSecurityAndObservationLifecycle:
    """Security guardrails, privacy, and caching lifecycle tests."""

    def test_16_sensitive_window_blocked(self) -> None:
        """Test 16: SecureVisionManager blocks scene parsing on sensitive windows."""
        policy = VisionSecurityPolicy()
        mgr = SecureVisionManager(policy=policy)

        with mock.patch.object(mgr.engine, "get_active_window_handle", return_value=9999):
            with mock.patch.object(mgr, "_resolve_window_metadata", return_value=("Bitwarden - Master Password", "bitwarden.exe")):
                with pytest.raises(CaptureBlockedError):
                    mgr.capture_active_window()

    def test_17_cache_reuse_for_repeated_queries(self) -> None:
        """Test 17: Repeated scene queries reuse active unexpired observation."""
        skills = VisionSkills()
        cap = _create_synthetic_capture(window_title="Doc Window", process_name="word.exe", hwnd=12345)
        skills.secure_vision_manager.buffer_manager.store(cap, metadata={"hwnd": 12345, "window_title": "Doc Window", "process_name": "word.exe", "bounds": cap.bounds.to_tuple()})

        with mock.patch.object(skills.secure_vision_manager, "check_active_window_authorized") as mock_auth:
            mock_auth.return_value = CaptureAuthorization(
                decision=CaptureDecision.ALLOW,
                category=CaptureCategory.SAFE,
                reason="Permitted",
                timestamp=time.time(),
            )
            with mock.patch.object(skills.secure_vision_manager, "get_active_window_identity", return_value=(12345, "Doc Window", "word.exe", cap.bounds.to_tuple())):
                res1 = skills.execute({"operation": "map_ui_scene", "parameters": {"reuse_cache": True}})
                assert res1.success
                assert res1.data.get("reused_cache") is True

    def test_18_cache_invalidation(self) -> None:
        """Test 18: Cache invalidation forces fresh capture on next scene mapping."""
        skills = VisionSkills()
        cap = _create_synthetic_capture()
        skills.secure_vision_manager.buffer_manager.store(cap)
        assert skills.secure_vision_manager.get_latest_observation() is not None

        skills.invalidate_observation_cache("ui_mutation")
        assert skills.secure_vision_manager.get_latest_observation() is None

    def test_19_no_raw_pixels_in_scene_models(self) -> None:
        """Test 19: UIScene, UIContainer, and FormField models contain zero raw pixel bytes."""
        wb = WindowBounds(left=0, top=0, right=800, bottom=600)
        elements = [
            UIElement(name="OK", element_type=UIElementType.BUTTON, bounds=WindowBounds(50, 50, 150, 90), center=Point(100, 70)),
        ]
        obs = _create_observation(bounds=wb, elements=elements)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs)
        data = scene.to_dict()
        serialized = json.dumps(data)

        assert "raw_bytes" not in serialized
        assert "stride" not in serialized
        assert "b'" not in serialized


class TestMultimodalFallback:
    """Multimodal VLM enhancement and fallback tests."""

    def test_20_vlm_unavailable_fallback_succeeds(self) -> None:
        """Test 20: Scene parser succeeds 100% deterministically when VLM is unavailable."""
        obs = _create_observation()
        parser = VisualSceneParser(ai_provider=None)

        scene = parser.parse_scene(obs, force_multimodal=True)
        assert scene is not None
        assert scene.confidence >= 0.5

    def test_21_malformed_vlm_response_handled_gracefully(self) -> None:
        """Test 21: Malformed non-JSON VLM response falls back cleanly to deterministic parse."""
        mock_ai = mock.MagicMock()
        mock_ai.supports_multimodal = True
        mock_ai.generate.return_value = "Sorry, I cannot parse this image as JSON."

        obs = _create_observation()
        parser = VisualSceneParser(ai_provider=mock_ai)

        scene = parser.parse_scene(obs, force_multimodal=True)
        assert scene is not None
        assert scene.scene_id is not None

    def test_22_vlm_enhancement_success(self) -> None:
        """Test 22: Successful VLM response enhances scene with additional containers and elements."""
        mock_ai = mock.MagicMock()
        mock_ai.supports_multimodal = True
        vlm_json = json.dumps({
            "summary": "Application dashboard with top navigation.",
            "containers": [
                {"type": "toolbar", "label": "VLM Toolbar", "box_2d": [100, 50, 200, 950], "confidence": 0.9},
            ],
            "interactive_elements": [
                {"name": "Custom Canvas Tool", "type": "button", "box_2d": [120, 60, 180, 150], "confidence": 0.9},
            ],
        })
        mock_ai.generate.return_value = f"```json\n{vlm_json}\n```"

        obs = _create_observation()
        parser = VisualSceneParser(ai_provider=mock_ai)

        scene = parser.parse_scene(obs, force_multimodal=True)
        assert scene is not None
        container_labels = [c.label for c in scene.containers if c.label]
        assert "VLM Toolbar" in container_labels
        assert any(e.name == "Custom Canvas Tool" for e in scene.interactive_elements)


class TestPlannerAndSkillIntegration:
    """Planner, natural language routing, and skill presentation tests."""

    def test_23_planner_map_ui_scene_action(self) -> None:
        """Test 23: Planner recognizes map_ui_scene action in KNOWN_ACTIONS and executes it."""
        assert "map_ui_scene" in KNOWN_ACTIONS

        planner = Planner()
        plan = planner.create_plan("map the ui scene of the active window")
        assert plan is not None

        executor = Executor()
        skills = mock.MagicMock()
        skills.execute.return_value = SystemSkillResult(
            success=True,
            operation="map_ui_scene",
            data={"summary": "Mapped 4 controls on the screen.", "interactive_elements": [{"name": "OK"}]},
        )
        executor.register_handler("map_ui_scene", lambda t: skills.execute(t.action))
        task = Task(id="task_1", action="map_ui_scene", target="active_window")
        test_plan = Plan(query="map the ui scene of the active window", tasks=[task])
        res = executor.execute_plan(test_plan)
        assert res.success

    def test_24_natural_language_routing(self) -> None:
        """Test 24: Natural language queries route correctly to map_ui_scene."""
        skills = VisionSkills()

        queries = [
            "map the ui scene",
            "what interactive controls are on this screen?",
            "list all buttons and inputs in this window",
            "what forms are open?",
            "what buttons and fields are available?",
            "what interactive elements are here?",
            "inspect the ui layout",
        ]

        for q in queries:
            assert skills.can_handle(q), f"Failed to match: {q}"
            op, target, params, _ = skills.parse_command(q)
            assert op == "map_ui_scene", f"Expected map_ui_scene for '{q}', got '{op}'"

    def test_25_router_intent_classification(self) -> None:
        """Test 25: IntentRouter classifies scene mapping queries under IntentType.VISION."""
        router = IntentRouter()
        res = router.classify("what interactive controls are available on screen?")
        assert res.intent == IntentType.VISION

    def test_26_container_filter_parameter(self) -> None:
        """Test 26: container_filter parameter filters returned containers."""
        wb = WindowBounds(left=0, top=0, right=1000, bottom=800)
        blocks = [
            OCRTextBlock(text="App Header", bounds=WindowBounds(20, 10, 200, 40)),
            OCRTextBlock(text="Status Ready", bounds=WindowBounds(20, 750, 150, 780)),
        ]
        obs = _create_observation(bounds=wb, ocr_blocks=blocks)

        parser = VisualSceneParser()
        scene = parser.parse_scene(obs, container_filter="header")

        assert len(scene.containers) == 1
        assert scene.containers[0].container_type == UIContainerType.HEADER

    def test_27_confidence_range_clamping(self) -> None:
        """Test 27: UIContainer, FormField, and UIScene clamp confidence to [0.0, 1.0]."""
        c = UIContainer(container_id="c1", container_type=UIContainerType.HEADER, bounds=WindowBounds(0, 0, 100, 50), confidence=1.5)
        assert c.confidence == 1.0

        f = FormField(
            field_id="f1",
            label="Test:",
            label_bounds=WindowBounds(0, 0, 50, 20),
            input_element=UIElement(
                name="Input",
                element_type=UIElementType.INPUT,
                bounds=WindowBounds(60, 0, 150, 20),
                center=Point(105, 10),
            ),
            confidence=-0.5,
        )
        assert f.confidence == 0.0

        s = UIScene(scene_id="s1", observation_id="o1", window_title="Test", confidence=2.0)
        assert s.confidence == 1.0

    def test_28_deterministic_repeated_execution(self) -> None:
        """Test 28: Repeated executions produce identical UIScene structures."""
        obs = _create_observation()
        parser = VisualSceneParser()

        s1 = parser.parse_scene(obs)
        s2 = parser.parse_scene(obs)

        assert s1.window_title == s2.window_title
        assert len(s1.containers) == len(s2.containers)
        assert len(s1.interactive_elements) == len(s2.interactive_elements)
        assert len(s1.form_fields) == len(s2.form_fields)
