"""Phase 27.17 Visual Action Grounding & Precondition Validation Tests.

Comprehensive deterministic unit and integration test suite covering:
1. Submit target resolution
2. Save target resolution
3. Input field resolution
4. Toggle resolution
5. Modal target inside modal
6. Background target blocked by modal
7. Ambiguous target
8. Disabled control
9. Empty required field
10. Non-actionable control
11. SAFE classification
12. MUTATING classification
13. DESTRUCTIVE classification
14. Unknown/ambiguous safety -> confirmation required
15. Valid target coordinate
16. Invalid bounds
17. Unsafe coordinate -> UNCERTAIN
18. VisualGoalSpec generation
19. Sensitive observation
20. metadata is_sensitive
21. metadata blocked
22. unauthorized observation
23. invalid observation
24. protected transition isolation
25. secret redaction
26. no raw pixel persistence
27. no background threads/polling
28. intent routing
29. voice-safe formatting
30. backward compatibility of existing vision operations
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
from app.vision.action_grounding import VisualActionGroundingEngine
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
    VisualActionFeasibilityStatus,
    VisualActionGroundingResult,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualSituation,
    VisualSituationType,
    WindowBounds,
)
from app.vision.security import SecureVisionManager, VisionSecurityPolicy


# ---------------------------------------------------------------------------
# Synthetic Fixture Helpers
# ---------------------------------------------------------------------------


def make_observation(
    window_title: str = "TestApp - Editor",
    process_name: str = "testapp.exe",
    is_sensitive: bool = False,
    blocked: bool = False,
    is_allowed: bool = True,
    is_valid: bool = True,
    observation_id: str = "obs_safe_1",
) -> ScreenObservation:
    """Create a synthetic ScreenObservation for action grounding testing."""
    bounds = WindowBounds(left=0, top=0, right=1280, bottom=720)
    auth = CaptureAuthorization(
        decision=CaptureDecision.ALLOW if is_allowed else CaptureDecision.BLOCK,
        category=CaptureCategory.SAFE if is_allowed else CaptureCategory.CREDENTIAL_INTERFACE,
        reason="Test policy",
    )
    meta = {
        "window_title": window_title,
        "process_name": process_name,
        "is_sensitive": is_sensitive,
        "blocked": blocked,
    }
    cap = None
    if is_valid and not blocked and is_allowed:
        cap = ScreenCapture(
            raw_data=b"\x00" * 64,
            width=1280,
            height=720,
            bounds=bounds,
            timestamp=time.time(),
            source="active_window",
            metadata=dict(meta),
        )

    obs = ScreenObservation(
        id=observation_id,
        capture=cap,
        authorization=auth,
        metadata=meta,
    )
    if not is_valid:
        obs.capture = None
    return obs


def make_element(
    canonical_name: str,
    element_type: UIElementType = UIElementType.BUTTON,
    text: Optional[str] = None,
    bounds: Optional[WindowBounds] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UIElement:
    """Create a synthetic UIElement."""
    meta = dict(metadata or {})
    b = bounds or WindowBounds(left=100, top=100, right=200, bottom=140)
    return UIElement(
        name=canonical_name,
        element_type=element_type,
        bounds=b,
        center=b.center,
        confidence=0.95,
        text_content=text or canonical_name,
        metadata=meta,
    )


def make_container(
    title: str,
    container_type: UIContainerType = UIContainerType.FORM,
    elements: Optional[List[UIElement]] = None,
    bounds: Optional[WindowBounds] = None,
    is_modal: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> UIContainer:
    """Create a synthetic UIContainer."""
    meta = dict(metadata or {})
    if is_modal:
        meta["is_modal"] = True
    b = bounds or WindowBounds(left=50, top=50, right=800, bottom=600)
    return UIContainer(
        container_id=f"cnt_{uuid.uuid4().hex[:6]}",
        container_type=container_type,
        bounds=b,
        elements=tuple(elements or []),
        label=title,
        confidence=0.95,
        metadata=meta,
    )


def make_affordance(
    element: UIElement,
    detected_state: ControlVisualState = ControlVisualState.ENABLED,
    primary_affordance: ControlAffordance = ControlAffordance.CLICKABLE,
) -> ElementAffordance:
    """Create an ElementAffordance for a UIElement."""
    return ElementAffordance(
        element=element,
        detected_state=detected_state,
        primary_affordance=primary_affordance,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVisualActionGrounding:
    """Deterministic unit test suite for Phase 27.17 Visual Action Grounding Engine."""

    # 1. Submit target resolution
    def test_01_submit_target_resolution(self) -> None:
        obs = make_observation()
        btn = make_element("Submit", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Registration",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.FEASIBLE
        assert result.target is not None
        assert result.target.target_element_name == "Submit"
        assert result.target.action_type == VisualActionType.CLICK
        assert result.target.target_point is not None

    # 2. Save target resolution
    def test_02_save_target_resolution(self) -> None:
        obs = make_observation()
        btn = make_element("Save", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Document Editor",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("where should I click to save", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.FEASIBLE
        assert result.target is not None
        assert result.target.target_element_name == "Save"

    # 3. Input field resolution
    def test_03_input_field_resolution(self) -> None:
        obs = make_observation()
        input_fld = make_element("Email Address", UIElementType.INPUT, text="")
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Login",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(input_fld,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("enter email", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.FEASIBLE
        assert result.target is not None
        assert result.target.action_type == VisualActionType.TYPE_TEXT
        assert result.target.target_element_name == "Email Address"

    # 4. Toggle resolution
    def test_04_toggle_resolution(self) -> None:
        obs = make_observation()
        chk = make_element("Remember Me", UIElementType.CHECKBOX)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Login",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(chk,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("toggle remember me", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.FEASIBLE
        assert result.target is not None
        assert result.target.action_type == VisualActionType.TOGGLE
        assert result.target.target_element_name == "Remember Me"

    # 5. Modal target inside modal
    def test_05_modal_target_inside_modal(self) -> None:
        obs = make_observation()
        close_btn = make_element(
            "Close",
            UIElementType.BUTTON,
            bounds=WindowBounds(left=300, top=200, right=380, bottom=240),
        )
        modal = make_container(
            "Alert Dialog",
            container_type=UIContainerType.DIALOG,
            elements=[close_btn],
            bounds=WindowBounds(left=200, top=150, right=600, bottom=400),
            is_modal=True,
        )
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            containers=(modal,),
            interactive_elements=(close_btn,),
        )
        sit = VisualSituation(
            situation_id="sit_1",
            timestamp=time.time(),
            observation_id=obs.id,
            situation_type=VisualSituationType.MODAL_DIALOG,
            active_modal=modal,
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("dismiss popup", obs, situation=sit, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.FEASIBLE
        assert result.target is not None
        assert result.target.target_element_name == "Close"
        assert result.target.target_point is not None

    # 6. Background target blocked by modal
    def test_06_background_target_blocked_by_modal(self) -> None:
        obs = make_observation()
        bg_btn = make_element(
            "Save",
            UIElementType.BUTTON,
            bounds=WindowBounds(left=50, top=50, right=150, bottom=90),
        )
        modal_btn = make_element(
            "Cancel",
            UIElementType.BUTTON,
            bounds=WindowBounds(left=300, top=200, right=380, bottom=240),
        )
        modal = make_container(
            "Confirm Dialog",
            container_type=UIContainerType.DIALOG,
            elements=[modal_btn],
            bounds=WindowBounds(left=200, top=150, right=600, bottom=400),
            is_modal=True,
        )
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            containers=(modal,),
            interactive_elements=(bg_btn, modal_btn),
        )
        sit = VisualSituation(
            situation_id="sit_1",
            timestamp=time.time(),
            observation_id=obs.id,
            situation_type=VisualSituationType.MODAL_DIALOG,
            active_modal=modal,
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Save", obs, situation=sit, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.BLOCKED_BY_MODAL
        assert result.target is not None
        assert result.target.feasibility == VisualActionFeasibilityStatus.BLOCKED_BY_MODAL
        assert result.target.target_point is None

    # 7. Ambiguous target
    def test_07_ambiguous_target(self) -> None:
        obs = make_observation()
        btn1 = make_element("Submit", UIElementType.BUTTON, bounds=WindowBounds(100, 100, 200, 140))
        btn2 = make_element("Submit", UIElementType.BUTTON, bounds=WindowBounds(100, 300, 200, 340))
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn1, btn2),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.UNCERTAIN
        assert result.metadata.get("requires_confirmation") is True

    # 8. Disabled control
    def test_08_disabled_control(self) -> None:
        obs = make_observation()
        btn = make_element("Submit", UIElementType.BUTTON)
        aff = make_affordance(btn, detected_state=ControlVisualState.DISABLED)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene, affordances=(aff,))

        assert result.status == VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED
        assert result.target is not None
        assert result.target.feasibility == VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED
        assert result.target.target_point is None

    # 9. Empty required field
    def test_09_empty_required_field(self) -> None:
        obs = make_observation()
        req_field = make_element(
            "Username",
            UIElementType.INPUT,
            text="",
            metadata={"is_required": True, "state": "empty"},
        )
        submit_btn = make_element("Submit", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Registration",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(req_field, submit_btn),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("can I submit this form", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.BLOCKED_UNFILLED_PREREQUISITES
        assert result.target is not None
        assert result.target.feasibility == VisualActionFeasibilityStatus.BLOCKED_UNFILLED_PREREQUISITES
        assert result.target.target_point is None

    # 10. Non-actionable control
    def test_10_non_actionable_control(self) -> None:
        obs = make_observation()
        static_lbl = make_element("Section Header", UIElementType.TEXT)
        aff = make_affordance(static_lbl, primary_affordance=ControlAffordance.READ_ONLY)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(static_lbl,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("click Section Header", obs, scene=scene, affordances=(aff,))

        assert result.status == VisualActionFeasibilityStatus.UNCERTAIN
        assert result.target is not None
        assert result.target.target_point is None

    # 11. SAFE classification
    def test_11_safe_classification(self) -> None:
        obs = make_observation()
        nav_btn = make_element("Next Tab", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(nav_btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("navigate Next Tab", obs, scene=scene)

        assert result.target is not None
        assert result.target.safety_tier == VisualActionSafetyTier.SAFE
        assert result.target.requires_confirmation is False

    # 12. MUTATING classification
    def test_12_mutating_classification(self) -> None:
        obs = make_observation()
        fld = make_element("Name Field", UIElementType.INPUT)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Profile",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(fld,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("enter Name Field", obs, scene=scene)

        assert result.target is not None
        assert result.target.safety_tier == VisualActionSafetyTier.MUTATING

    # 13. DESTRUCTIVE classification
    def test_13_destructive_classification(self) -> None:
        obs = make_observation()
        del_btn = make_element("Delete Account", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Settings",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(del_btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("delete account", obs, scene=scene)

        assert result.target is not None
        assert result.target.safety_tier == VisualActionSafetyTier.DESTRUCTIVE
        assert result.target.requires_confirmation is True

    # 14. Unknown/ambiguous safety -> confirmation required
    def test_14_unknown_safety_requires_confirmation(self) -> None:
        obs = make_observation()
        btn = make_element("Execute Custom Hook", UIElementType.BUTTON)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Terminal",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Execute Custom Hook", obs, scene=scene)

        assert result.target is not None
        assert result.target.safety_tier == VisualActionSafetyTier.DESTRUCTIVE
        assert result.target.requires_confirmation is True

    # 15. Valid target coordinate
    def test_15_valid_target_coordinate(self) -> None:
        obs = make_observation()
        b = WindowBounds(left=200, top=150, right=400, bottom=250)
        btn = make_element("Submit", UIElementType.BUTTON, bounds=b)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene)

        assert result.target is not None
        pt = result.target.target_point
        assert pt is not None
        assert b.left < pt.x < b.right
        assert b.top < pt.y < b.bottom
        assert pt.x == 300
        assert pt.y == 200

    # 16. Invalid bounds
    def test_16_invalid_bounds(self) -> None:
        obs = make_observation()
        # Inverted or 0-width bounds
        b = WindowBounds(left=200, top=150, right=200, bottom=250)
        btn = make_element("Submit", UIElementType.BUTTON, bounds=b)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.UNCERTAIN
        assert result.target is not None
        assert result.target.target_point is None

    # 17. Unsafe coordinate -> UNCERTAIN
    def test_17_unsafe_coordinate_uncertain(self) -> None:
        obs = make_observation()
        b = WindowBounds(left=100, top=100, right=101, bottom=101)  # 1x1 too small
        btn = make_element("Submit", UIElementType.BUTTON, bounds=b)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(btn,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs, scene=scene)

        assert result.status == VisualActionFeasibilityStatus.UNCERTAIN
        assert result.target is not None
        assert result.target.target_point is None

    # 18. VisualGoalSpec generation
    def test_18_visual_goal_spec_generation(self) -> None:
        obs = make_observation()
        chk = make_element("Enable Notifications", UIElementType.CHECKBOX)
        scene = UIScene(
            scene_id="scn_1",
            observation_id=obs.id,
            window_title="Settings",
            window_bounds=WindowBounds(0, 0, 1280, 720),
            interactive_elements=(chk,),
        )
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("toggle Enable Notifications", obs, scene=scene)

        assert result.target is not None
        spec = result.target.expected_outcome
        assert spec is not None
        assert spec.criterion == VisualGoalCriterion.ELEMENT_STATE
        assert spec.expected_state == ControlVisualState.CHECKED

    # 19. Sensitive observation
    def test_19_sensitive_observation(self) -> None:
        obs = make_observation(is_sensitive=True)
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs)

        assert result.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert result.target is None
        assert "protected sensitive information" in result.summary

    # 20. metadata is_sensitive
    def test_20_metadata_is_sensitive(self) -> None:
        obs = make_observation()
        obs.metadata["is_sensitive"] = True
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs)

        assert result.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert result.target is None

    # 21. metadata blocked
    def test_21_metadata_blocked(self) -> None:
        obs = make_observation(blocked=True)
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs)

        assert result.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert result.target is None

    # 22. unauthorized observation
    def test_22_unauthorized_observation(self) -> None:
        obs = make_observation(is_allowed=False)
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs)

        assert result.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert result.target is None

    # 23. invalid observation
    def test_23_invalid_observation(self) -> None:
        obs = make_observation(is_valid=False)
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Submit", obs)

        assert result.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert result.target is None

    # 24. protected transition isolation
    def test_24_protected_transition_isolation(self) -> None:
        engine = VisualActionGroundingEngine()

        obs_a = make_observation(window_title="Safe App A", observation_id="obs_a")
        btn_a = make_element("Submit A", UIElementType.BUTTON)
        scene_a = UIScene("scn_a", obs_a.id, "Safe App A", WindowBounds(0, 0, 800, 600), interactive_elements=(btn_a,))
        res_a = engine.ground_action("Click Submit A", obs_a, scene=scene_a)
        assert res_a.status == VisualActionFeasibilityStatus.FEASIBLE

        # Sensitive transition
        obs_sec = make_observation(window_title="1Password Secret Vault", is_sensitive=True, observation_id="obs_sec")
        res_sec = engine.ground_action("Click Submit A", obs_sec)
        assert res_sec.status == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED
        assert res_sec.target is None

        # Safe B: verify no bleed from A or Protected
        obs_b = make_observation(window_title="Safe App B", observation_id="obs_b")
        btn_b = make_element("Submit B", UIElementType.BUTTON)
        scene_b = UIScene("scn_b", obs_b.id, "Safe App B", WindowBounds(0, 0, 800, 600), interactive_elements=(btn_b,))
        res_b = engine.ground_action("Click Submit B", obs_b, scene=scene_b)
        assert res_b.status == VisualActionFeasibilityStatus.FEASIBLE
        assert res_b.target is not None
        assert res_b.target.target_element_name == "Submit B"

    # 25. secret redaction
    def test_25_secret_redaction(self) -> None:
        obs = make_observation()
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("enter password API_KEY_SECRET_123", obs)

        assert result.status == VisualActionFeasibilityStatus.UNCERTAIN
        assert "API_KEY" not in result.summary
        assert "password" not in result.summary.lower()

    # 26. no raw pixel persistence
    def test_26_no_raw_pixel_persistence(self) -> None:
        obs = make_observation()
        btn = make_element("Save", UIElementType.BUTTON)
        scene = UIScene("scn_1", obs.id, "Editor", WindowBounds(0, 0, 800, 600), interactive_elements=(btn,))
        engine = VisualActionGroundingEngine()
        result = engine.ground_action("Click Save", obs, scene=scene)

        data = result.to_dict()
        # Verify no image byte arrays, base64 strings, or buffers
        serialized = str(data)
        assert "raw_data" not in serialized
        assert "\\x00" not in serialized
        assert "base64" not in serialized

    # 27. no background threads/polling
    def test_27_no_background_threads(self) -> None:
        t_before = threading.active_count()
        engine = VisualActionGroundingEngine()
        obs = make_observation()
        btn = make_element("Save", UIElementType.BUTTON)
        scene = UIScene("scn_1", obs.id, "Editor", WindowBounds(0, 0, 800, 600), interactive_elements=(btn,))
        engine.ground_action("Click Save", obs, scene=scene)
        t_after = threading.active_count()

        assert t_after == t_before

    # 28. intent routing
    def test_28_intent_routing(self) -> None:
        router = IntentRouter(auto_register_in_container=False)

        queries = [
            ("where should I click to save", IntentType.VISION),
            ("which button should I click", IntentType.VISION),
            ("can I submit this form", IntentType.VISION),
            ("is this button ready", IntentType.VISION),
            ("how do I dismiss this popup", IntentType.VISION),
            ("locate the control for submitting", IntentType.VISION),
            ("identify the control for saving", IntentType.VISION),
        ]

        for query, expected_intent in queries:
            res = router.classify(query)
            assert res.intent == expected_intent, f"Query '{query}' classified as {res.intent} instead of {expected_intent}"

    # 29. voice-safe formatting
    def test_29_voice_safe_formatting(self) -> None:
        # Feasible
        data_feasible = {
            "status": "FEASIBLE",
            "target_element_name": "Submit",
            "summary": "Control 'Submit' is ready and feasible for click.",
            "requires_confirmation": False,
        }
        res_feas = SystemSkillResult(operation="ground_visual_action", success=True, data=data_feasible)
        assert "Submit" in res_feas.message
        assert "(" not in res_feas.message  # No raw coordinates dumped

        # Blocked by modal
        data_modal = {
            "status": "BLOCKED_BY_MODAL",
            "target_element_name": "Save",
            "summary": "Control 'Save' is blocked by an active modal dialog.",
        }
        res_modal = SystemSkillResult(operation="ground_visual_action", success=True, data=data_modal)
        assert "blocked by an active modal dialog" in res_modal.message

        # Sensitive protected
        data_prot = {
            "status": "SENSITIVE_PROTECTED",
            "summary": "Action grounding blocked: active screen context contains protected sensitive information.",
        }
        res_prot = SystemSkillResult(operation="ground_visual_action", success=True, data=data_prot)
        assert "Action grounding blocked" in res_prot.message

    # 30. backward compatibility of existing vision operations
    def test_30_backward_compatibility(self) -> None:
        policy = VisionSecurityPolicy()
        sec_mgr = SecureVisionManager(policy=policy)
        skill = VisionSkills(secure_vision_manager=sec_mgr)

        assert skill.can_handle("explain this window")
        assert skill.can_handle("what is happening on screen")
        assert skill.can_handle("where should I click to save")
        assert skill.can_handle("can I submit this form")

        op, tgt, params, _ = skill.parse_command("where should I click to save")
        assert op == "ground_visual_action"
        assert "intent" in params
