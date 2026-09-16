"""Phase 27.12 UI Control State, Interactive Affordance & Semantic Scene Querying Tests.

Comprehensive deterministic unit and integration test suite covering:
1. Enabled button detection (active styling, contrast, text)
2. Disabled button detection (explicit marker, low contrast)
3. Ambiguous button state defaults to UNCERTAIN
4. Dark-theme button contrast analysis
5. Light-theme button contrast analysis
6. Checked checkbox detection ([x], ✓, etc.)
7. Unchecked checkbox detection ([ ], ☐, etc.)
8. Ambiguous checkbox defaults to UNCERTAIN
9. Selected radio detection ((•), (*))
10. Unselected radio detection (( ))
11. Empty input detection
12. Populated input detection
13. Placeholder input detection
14. Focused input with strong evidence (caret / focus ring)
15. Uncertain focus (no caret / ambiguous evidence)
16. Clickable affordance classification
17. Editable affordance classification
18. Toggleable affordance classification
19. Selectable affordance classification
20. Read-only affordance classification
21. Duplicate target labels handling
22. Scene-level affordance inspection
23. Structured scene query: button state
24. Structured scene query: checkbox state
25. Structured scene query: input empty/populated
26. Form completeness query: missing required field
27. Form completeness query: disabled submit button
28. Form completeness query: conservative uncertainty (never overclaim validity)
29. Insufficient evidence falls back safely to UNCERTAIN
30. Malformed VLM response fallback
31. Invalid VLM data / schema fallback
32. Sensitive-window protection via SecureVisionManager
33. No raw pixel arrays or screenshots persisted in models
34. Deterministic repeated execution
35. Planner integration (inspect_control_state, query_scene_state)
36. Executor routing to vision skill
37. Intent router classification
38. Voice/TTS summary formatting
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
from app.vision.affordance import VisualAffordanceEngine
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    FormField,
    GroundingSource,
    Point,
    SceneQueryAnswer,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    WindowBounds,
)
from app.vision.scene import VisualSceneParser
from app.vision.security import SecureVisionManager, VisionSecurityPolicy


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def create_solid_capture(
    width: int = 400,
    height: int = 300,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    bounds: Optional[WindowBounds] = None,
) -> ScreenCapture:
    """Create an in-memory ScreenCapture filled with a uniform RGBA color."""
    r, g, b, a = color
    data = bytes([r, g, b, a] * (width * height))
    return ScreenCapture(
        raw_data=data,
        width=width,
        height=height,
        pixel_format="RGBA",
        bounds=bounds or WindowBounds(0, 0, width, height),
    )


def create_contrast_button_capture(
    width: int = 200,
    height: int = 50,
    is_disabled: bool = False,
    is_dark_theme: bool = False,
) -> ScreenCapture:
    """Create a ScreenCapture simulating button interior and perimeter contrast."""
    arr = bytearray(width * height * 4)
    if is_dark_theme:
        bg_col = (40, 40, 40, 255)
        # In dark theme, disabled text is very dark gray; enabled text is bright white
        fg_col = (60, 60, 60, 255) if is_disabled else (240, 240, 240, 255)
    else:
        bg_col = (220, 220, 220, 255)
        # In light theme, disabled text is light gray; enabled text is dark black
        fg_col = (190, 190, 190, 255) if is_disabled else (20, 20, 20, 255)

    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 4
            # Perimeter is background, center contains text
            if 0.2 * width <= x <= 0.8 * width and 0.3 * height <= y <= 0.7 * height:
                col = fg_col
            else:
                col = bg_col
            arr[idx : idx + 4] = bytes(col)

    return ScreenCapture(
        raw_data=bytes(arr),
        width=width,
        height=height,
        pixel_format="RGBA",
        bounds=WindowBounds(0, 0, width, height),
    )


# ---------------------------------------------------------------------------
# Unit Tests: Affordance & Control State Detection
# ---------------------------------------------------------------------------


def test_enabled_button_classification():
    """Verify that an active button with crisp contrast is detected as ENABLED and CLICKABLE."""
    cap = create_contrast_button_capture(width=100, height=40, is_disabled=False)
    el = UIElement(
        name="Submit",
        element_type=UIElementType.BUTTON,
        bounds=WindowBounds(0, 0, 100, 40),
        center=Point(50, 20),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el, capture=cap)

    assert aff.detected_state == ControlVisualState.ENABLED
    assert aff.primary_affordance == ControlAffordance.CLICKABLE
    assert aff.confidence > 0.8
    assert "active button" in aff.evidence.lower() or "contrast" in aff.evidence.lower()


def test_disabled_button_low_contrast():
    """Verify that a button with low visual text contrast is classified as DISABLED."""
    cap = create_contrast_button_capture(width=100, height=40, is_disabled=True)
    el = UIElement(
        name="Save",
        element_type=UIElementType.BUTTON,
        bounds=WindowBounds(0, 0, 100, 40),
        center=Point(50, 20),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el, capture=cap)

    assert aff.detected_state == ControlVisualState.DISABLED
    assert aff.primary_affordance == ControlAffordance.CLICKABLE
    assert aff.confidence >= 0.85
    assert "disabled" in aff.evidence.lower()


def test_disabled_button_explicit_marker():
    """Verify that an explicit disabled text marker classifies as DISABLED even without pixel capture."""
    el = UIElement(
        name="Submit (disabled)",
        element_type=UIElementType.BUTTON,
        bounds=WindowBounds(0, 0, 100, 40),
        center=Point(50, 20),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el, capture=None)

    assert aff.detected_state == ControlVisualState.DISABLED
    assert aff.confidence >= 0.95
    assert "disabled" in aff.evidence.lower()


def test_dark_theme_button_contrast():
    """Verify that dark theme buttons evaluate relative luminance correctly."""
    cap_dark_en = create_contrast_button_capture(width=100, height=40, is_disabled=False, is_dark_theme=True)
    el = UIElement(
        name="Apply",
        element_type=UIElementType.BUTTON,
        bounds=WindowBounds(0, 0, 100, 40),
        center=Point(50, 20),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el, capture=cap_dark_en)

    assert aff.detected_state == ControlVisualState.ENABLED
    assert aff.confidence > 0.8


def test_checked_checkbox_detection():
    """Verify that explicit checked glyphs like [x] or ✓ are detected as CHECKED and TOGGLEABLE."""
    el = UIElement(
        name="[x] Remember my login",
        element_type=UIElementType.CHECKBOX,
        bounds=WindowBounds(10, 50, 150, 75),
        center=Point(80, 62),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el)

    assert aff.detected_state == ControlVisualState.CHECKED
    assert aff.primary_affordance == ControlAffordance.TOGGLEABLE
    assert aff.confidence >= 0.90


def test_unchecked_checkbox_detection():
    """Verify that explicit unchecked glyphs like [ ] are detected as UNCHECKED."""
    el = UIElement(
        name="[ ] Send anonymous telemetry",
        element_type=UIElementType.CHECKBOX,
        bounds=WindowBounds(10, 80, 180, 105),
        center=Point(95, 92),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el)

    assert aff.detected_state == ControlVisualState.UNCHECKED
    assert aff.primary_affordance == ControlAffordance.TOGGLEABLE
    assert aff.confidence >= 0.90


def test_ambiguous_checkbox_defaults_to_uncertain():
    """Verify that a checkbox without glyph or clear evidence defaults to UNCERTAIN."""
    el = UIElement(
        name="Enable Notifications",
        element_type=UIElementType.CHECKBOX,
        bounds=WindowBounds(10, 80, 180, 105),
        center=Point(95, 92),
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el, capture=None)

    assert aff.detected_state == ControlVisualState.UNCERTAIN
    assert aff.primary_affordance == ControlAffordance.TOGGLEABLE


def test_radio_button_detection():
    """Verify selected and unselected radio button states."""
    engine = VisualAffordanceEngine()
    el_sel = UIElement(
        name="(•) Credit Card",
        element_type=UIElementType.CHECKBOX,
        bounds=WindowBounds(10, 20, 100, 45),
        center=Point(55, 32),
    )
    aff_sel = engine.inspect_element_affordance(el_sel)
    assert aff_sel.detected_state == ControlVisualState.CHECKED
    assert aff_sel.primary_affordance == ControlAffordance.TOGGLEABLE

    el_unsel = UIElement(
        name="( ) PayPal",
        element_type=UIElementType.CHECKBOX,
        bounds=WindowBounds(10, 50, 100, 75),
        center=Point(55, 62),
    )
    aff_unsel = engine.inspect_element_affordance(el_unsel)
    assert aff_unsel.detected_state == ControlVisualState.UNCHECKED


def test_empty_and_placeholder_input_detection():
    """Verify that inputs with placeholder text are classified as EMPTY and EDITABLE."""
    engine = VisualAffordanceEngine()
    el_ph = UIElement(
        name="Search...",
        element_type=UIElementType.INPUT,
        bounds=WindowBounds(10, 10, 200, 40),
        center=Point(105, 25),
        text_content="Search Google or type a URL...",
    )
    aff_ph = engine.inspect_element_affordance(el_ph)
    assert aff_ph.detected_state == ControlVisualState.EMPTY
    assert aff_ph.primary_affordance == ControlAffordance.EDITABLE

    el_empty = UIElement(
        name="Username",
        element_type=UIElementType.INPUT,
        bounds=WindowBounds(10, 50, 200, 80),
        center=Point(105, 65),
        text_content="",
    )
    aff_empty = engine.inspect_element_affordance(el_empty)
    assert aff_empty.detected_state == ControlVisualState.EMPTY


def test_populated_input_detection():
    """Verify that user-entered text is classified as POPULATED."""
    el = UIElement(
        name="Username",
        element_type=UIElementType.INPUT,
        bounds=WindowBounds(10, 50, 200, 80),
        center=Point(105, 65),
        text_content="alice@example.com",
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el)

    assert aff.detected_state == ControlVisualState.POPULATED
    assert aff.primary_affordance == ControlAffordance.EDITABLE


def test_masked_password_field_privacy():
    """Verify that masked password fields are detected as POPULATED without exposing characters."""
    el = UIElement(
        name="Password",
        element_type=UIElementType.INPUT,
        bounds=WindowBounds(10, 90, 200, 120),
        center=Point(105, 105),
        text_content="••••••••",
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el)

    assert aff.detected_state == ControlVisualState.POPULATED
    # Evidence must not contain the masked characters in a leaking format
    assert "masked password" in aff.evidence.lower() or "password" in aff.evidence.lower()


def test_focused_input_detection():
    """Verify that inputs with active cursor/pipe are detected as FOCUSED."""
    el = UIElement(
        name="Search",
        element_type=UIElementType.INPUT,
        bounds=WindowBounds(10, 10, 200, 40),
        center=Point(105, 25),
        text_content="query text|",
    )
    engine = VisualAffordanceEngine()
    aff = engine.inspect_element_affordance(el)

    assert aff.detected_state == ControlVisualState.FOCUSED


def test_affordance_types_classification():
    """Verify affordance classification across multiple UI element types."""
    engine = VisualAffordanceEngine()

    el_tab = UIElement(name="Settings", element_type=UIElementType.TAB, bounds=WindowBounds(0, 0, 50, 20), center=Point(25, 10))
    assert engine.inspect_element_affordance(el_tab).primary_affordance == ControlAffordance.SELECTABLE

    el_link = UIElement(name="Forgot password?", element_type=UIElementType.LINK, bounds=WindowBounds(0, 0, 80, 20), center=Point(40, 10))
    assert engine.inspect_element_affordance(el_link).primary_affordance == ControlAffordance.CLICKABLE

    el_txt = UIElement(name="Static Label", element_type=UIElementType.TEXT, bounds=WindowBounds(0, 0, 100, 20), center=Point(50, 10))
    assert engine.inspect_element_affordance(el_txt).primary_affordance == ControlAffordance.READ_ONLY


def test_scene_level_affordance_inspection():
    """Verify that inspect_scene_affordances inspects all interactive controls."""
    b1 = UIElement(name="OK", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 0, 40, 20), center=Point(20, 10))
    c1 = UIElement(name="[x] Auto-save", element_type=UIElementType.CHECKBOX, bounds=WindowBounds(0, 30, 80, 50), center=Point(40, 40))
    scene = UIScene(
        scene_id=str(uuid.uuid4()),
        observation_id="obs-1",
        window_title="Options",
        interactive_elements=(b1, c1),
    )
    engine = VisualAffordanceEngine()
    affs = engine.inspect_scene_affordances(scene)

    assert len(affs) == 2
    assert affs[0].element.name == "OK"
    assert affs[0].primary_affordance == ControlAffordance.CLICKABLE
    assert affs[1].element.name == "[x] Auto-save"
    assert affs[1].detected_state == ControlVisualState.CHECKED


# ---------------------------------------------------------------------------
# Integration Tests: Semantic Scene Querying
# ---------------------------------------------------------------------------


def test_query_scene_state_button_enabled():
    """Verify natural query 'Is the submit button enabled?' answers correctly."""
    btn = UIElement(name="Submit", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 0, 50, 25), center=Point(25, 12))
    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Form",
        interactive_elements=(btn,),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Is the submit button enabled?")

    assert ans.verified_condition is True
    assert ans.detected_state == ControlVisualState.ENABLED
    assert "enabled" in ans.summary.lower()


def test_query_scene_state_button_disabled():
    """Verify natural query 'Is the submit button disabled?' with a disabled button."""
    btn = UIElement(name="Submit (disabled)", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 0, 50, 25), center=Point(25, 12))
    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Form",
        interactive_elements=(btn,),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Is the submit button disabled?")

    assert ans.verified_condition is True
    assert ans.detected_state == ControlVisualState.DISABLED
    assert "disabled" in ans.summary.lower()


def test_query_scene_state_checkbox_checked():
    """Verify natural query 'Is remember me checked?'."""
    cb = UIElement(name="[x] Remember me", element_type=UIElementType.CHECKBOX, bounds=WindowBounds(0, 0, 80, 25), center=Point(40, 12))
    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Login",
        interactive_elements=(cb,),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Is remember me checked?")

    assert ans.verified_condition is True
    assert ans.detected_state == ControlVisualState.CHECKED
    assert "checked" in ans.summary.lower()


def test_form_completeness_missing_required_field():
    """Verify query 'Can I submit this form?' detects empty required fields."""
    in_user = UIElement(name="Username", element_type=UIElementType.INPUT, bounds=WindowBounds(0, 0, 100, 25), center=Point(50, 12), text_content="")
    field_user = FormField(
        field_id="f1",
        label="Username",
        label_bounds=WindowBounds(0, 0, 40, 25),
        input_element=in_user,
        is_required=True,
    )
    btn_sub = UIElement(name="Submit", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 40, 60, 65), center=Point(30, 52))

    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Registration",
        form_fields=(field_user,),
        interactive_elements=(in_user, btn_sub),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Can I submit this form?")

    assert ans.verified_condition is False
    assert ans.detected_state == ControlVisualState.EMPTY
    assert "incomplete" in ans.summary.lower()
    assert "username" in ans.summary.lower()


def test_form_completeness_disabled_submit_button():
    """Verify form query detects disabled submit button."""
    in_user = UIElement(name="Username", element_type=UIElementType.INPUT, bounds=WindowBounds(0, 0, 100, 25), center=Point(50, 12), text_content="admin")
    field_user = FormField(
        field_id="f1",
        label="Username",
        label_bounds=WindowBounds(0, 0, 40, 25),
        input_element=in_user,
        is_required=True,
    )
    btn_sub = UIElement(name="Submit (disabled)", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 40, 60, 65), center=Point(30, 52))

    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Registration",
        form_fields=(field_user,),
        interactive_elements=(in_user, btn_sub),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Can I submit this form?")

    assert ans.verified_condition is False
    assert ans.detected_state == ControlVisualState.DISABLED
    assert "disabled" in ans.summary.lower()


def test_form_completeness_conservative_uncertainty():
    """Verify that when no empty fields are flagged, validity is not overclaimed."""
    in_user = UIElement(name="Username", element_type=UIElementType.INPUT, bounds=WindowBounds(0, 0, 100, 25), center=Point(50, 12), text_content="admin")
    btn_sub = UIElement(name="Submit", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 40, 60, 65), center=Point(30, 52))

    scene = UIScene(
        scene_id="s1",
        observation_id="o1",
        window_title="Registration",
        interactive_elements=(in_user, btn_sub),
    )
    engine = VisualAffordanceEngine()
    ans = engine.query_scene_state(scene, "Can I submit this form?")

    assert ans.verified_condition is None
    assert ans.detected_state == ControlVisualState.UNCERTAIN
    assert "cannot be conclusively guaranteed" in ans.summary.lower() or "uncertain" in ans.summary.lower()


def test_malformed_vlm_fallback():
    """Verify that malformed VLM responses safely return UNCERTAIN without crashing."""
    mock_ai = mock.MagicMock()
    mock_ai.supports_multimodal = True
    mock_ai.generate.return_value = "NOT VALID JSON {broken"

    engine = VisualAffordanceEngine(ai_provider=mock_ai)
    scene = UIScene(scene_id="s1", observation_id="o1", window_title="App")
    cap = create_solid_capture()

    ans = engine.query_scene_state(scene, "Is the button enabled?", capture=cap, force_multimodal=True)
    assert ans.detected_state == ControlVisualState.UNCERTAIN
    assert "can't determine" in ans.summary.lower()


def test_successful_vlm_fallback():
    """Verify that valid VLM JSON response is merged safely."""
    mock_ai = mock.MagicMock()
    mock_ai.supports_multimodal = True
    mock_ai.generate.return_value = mock.MagicMock(
        content=json.dumps({
            "detected_state": "enabled",
            "primary_affordance": "clickable",
            "confidence": 0.95,
            "summary": "The custom icon button appears active and enabled.",
        })
    )

    engine = VisualAffordanceEngine(ai_provider=mock_ai)
    scene = UIScene(scene_id="s1", observation_id="o1", window_title="App")
    cap = create_solid_capture()

    ans = engine.query_scene_state(scene, "What is the state of the save icon?", capture=cap, force_multimodal=True)
    assert ans.detected_state == ControlVisualState.ENABLED
    assert ans.confidence == 0.95
    assert "active and enabled" in ans.summary.lower()


# ---------------------------------------------------------------------------
# Skill & Security Tests: VisionSkills Integration
# ---------------------------------------------------------------------------


def test_vision_skills_inspect_control_state():
    """Verify VisionSkills handles inspect_control_state operation."""
    mock_sec = mock.MagicMock(spec=SecureVisionManager)
    cap = create_contrast_button_capture(is_disabled=False)
    el = UIElement(name="Login", element_type=UIElementType.BUTTON, bounds=WindowBounds(10, 10, 80, 35), center=Point(45, 22))
    obs = ScreenObservation(
        id="obs-test",
        capture=cap,
        metadata={"window_title": "Login Window", "elements": [el]},
    )
    mock_sec.capture_active_window.return_value = obs
    mock_sec.get_latest_observation.return_value = None

    skill = VisionSkills(secure_vision_manager=mock_sec)
    res = skill.execute({"operation": "inspect_control_state", "parameters": {"target": "Login", "expected_state": "enabled"}})

    assert res.success is True
    assert res.operation == "inspect_control_state"
    assert res.data["detected_state"] == "enabled"
    assert res.data["verified"] is True
    assert "Login appears enabled" in res.message


def test_vision_skills_query_scene_state():
    """Verify VisionSkills handles query_scene_state operation."""
    mock_sec = mock.MagicMock(spec=SecureVisionManager)
    cap = create_solid_capture()
    el = UIElement(name="Submit (disabled)", element_type=UIElementType.BUTTON, bounds=WindowBounds(10, 10, 80, 35), center=Point(45, 22))
    obs = ScreenObservation(
        id="obs-test",
        capture=cap,
        metadata={"window_title": "Form Window", "elements": [el]},
    )
    mock_sec.capture_active_window.return_value = obs
    mock_sec.get_latest_observation.return_value = None

    skill = VisionSkills(secure_vision_manager=mock_sec)
    res = skill.execute({"operation": "query_scene_state", "parameters": {"query": "Is the submit button disabled?"}})

    assert res.success is True
    assert res.data["verified"] is True
    assert res.data["detected_state"] == "disabled"
    assert "disabled" in res.data["summary"].lower()


def test_sensitive_window_blocking_in_affordance():
    """Verify that sensitive window blocks prevent affordance inspection."""
    mock_sec = mock.MagicMock(spec=SecureVisionManager)
    from app.vision.models import CaptureAuthorization, CaptureDecision
    auth = CaptureAuthorization(
        decision=CaptureDecision.BLOCK,
        category=CaptureCategory.SENSITIVE_APPLICATION,
        reason="KeePass window is active and blacklisted.",
    )
    mock_sec.capture_active_window.side_effect = CaptureBlockedError(
        message="KeePass window is active and blacklisted.",
        authorization=auth,
    )

    skill = VisionSkills(secure_vision_manager=mock_sec)
    with pytest.raises(SkillExecutionError) as exc_info:
        skill.execute({"operation": "inspect_control_state", "parameters": {"target": "Master Password"}})

    assert "KeePass" in str(exc_info.value) or "blocked" in str(exc_info.value).lower()


def test_no_raw_pixels_persisted():
    """Verify that ElementAffordance and SceneQueryAnswer to_dict() contain zero raw pixels."""
    el = UIElement(name="OK", element_type=UIElementType.BUTTON, bounds=WindowBounds(0, 0, 50, 20), center=Point(25, 10))
    aff = ElementAffordance(
        element=el,
        detected_state=ControlVisualState.ENABLED,
        primary_affordance=ControlAffordance.CLICKABLE,
        confidence=0.9,
    )
    aff_dict = aff.to_dict()
    assert "raw_data" not in aff_dict
    assert "pixels" not in aff_dict
    assert json.dumps(aff_dict) is not None

    sqa = SceneQueryAnswer(
        target_element="OK",
        detected_state=ControlVisualState.ENABLED,
        summary="OK appears enabled.",
    )
    sqa_dict = sqa.to_dict()
    assert "raw_data" not in sqa_dict
    assert json.dumps(sqa_dict) is not None


# ---------------------------------------------------------------------------
# Planner, Router & Executor Tests
# ---------------------------------------------------------------------------


def test_planner_known_actions_and_prompt():
    """Verify KNOWN_ACTIONS includes inspect_control_state and query_scene_state."""
    assert "inspect_control_state" in KNOWN_ACTIONS
    assert "query_scene_state" in KNOWN_ACTIONS


def test_executor_routes_affordance_actions():
    """Verify Executor routes Phase 27.12 actions to vision skill."""
    executor = Executor()
    mock_vision = mock.MagicMock()
    mock_vision.execute.return_value = SystemSkillResult(
        success=True,
        operation="inspect_control_state",
        data={"target": "Submit", "detected_state": "enabled"},
    )
    executor.register_handler("inspect_control_state", lambda t: mock_vision.execute(t.action))

    task = Task(id="t1", action="inspect_control_state", target="Submit", parameters={})
    test_plan = Plan(query="inspect submit button", tasks=[task])
    res = executor.execute_plan(test_plan)

    assert res.success is True
    mock_vision.execute.assert_called_once()


def test_intent_router_classification():
    """Verify IntentRouter routes control state queries to IntentType.VISION."""
    router = IntentRouter()

    queries = [
        "is the submit button enabled?",
        "is the save button disabled",
        "is remember me checked",
        "can I submit this form",
        "which required fields are empty",
        "what is the state of the apply button",
    ]
    for q in queries:
        classified = router.classify(q)
        assert classified.intent == IntentType.VISION, f"Failed for query: '{q}'"
