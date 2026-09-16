"""Phase 27.13 Visual Task Outcome & Goal-State Verification Engine Tests.

Comprehensive deterministic unit and integration test suite covering:
1. WINDOW_PRESENT verified
2. WINDOW_PRESENT not verified
3. WINDOW_ABSENT verified
4. WINDOW_ABSENT not verified
5. TEXT_PRESENT
6. TEXT_ABSENT
7. ELEMENT_STATE enabled
8. ELEMENT_STATE disabled
9. ELEMENT_STATE checked
10. ELEMENT_STATE unchecked
11. ELEMENT_STATE empty
12. CONTAINER_PRESENT
13. VISUAL_DELTA expected change
14. VISUAL_DELTA missing change
15. Insufficient evidence → UNCERTAIN
16. Conflicting evidence → UNCERTAIN
17. Blocked capture → BLOCKED
18. Evidence chain generation
19. Sanitized evidence
20. No raw pixel persistence
21. No sensitive content logging
22. VLM fallback
23. VLM unavailable
24. Invalid VLM output
25. verify_screen_state backward compatibility
26. Natural-language goal parsing
27. Task expected_visual_goal default None
28. Executor without visual goal unchanged
29. Executor with visual goal verifies post-condition
30. VERIFIED post-condition
31. NOT_VERIFIED post-condition
32. UNCERTAIN post-condition
33. Planner integration
34. Intent-router integration
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest import mock
import uuid

import pytest

from app.ai.intent_router import IntentRouter, IntentType
from app.ai.planner.executor import Executor
from app.ai.planner.models import Plan, Task, TaskStatus
from app.ai.planner.planner import KNOWN_ACTIONS, Planner
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.vision_skills import VisionSkills
from app.vision.affordance import VisualAffordanceEngine
from app.vision.delta import VisualDeltaEngine
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
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementChange,
    UIElementType,
    UIScene,
    VisualDeltaResult,
    VisualDeltaType,
    VisualEvidenceItem,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualOutcomeType,
    VisualVerificationResult,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.scene import VisualSceneParser
from app.vision.security import SecureVisionManager, VisionSecurityPolicy
from app.vision.verification import VisualVerificationEngine, parse_visual_goal


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def create_mock_capture(
    width: int = 400,
    height: int = 300,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    metadata: Optional[Dict[str, Any]] = None,
) -> ScreenCapture:
    """Create an in-memory ScreenCapture filled with a uniform RGBA color."""
    r, g, b, a = color
    data = bytes([r, g, b, a] * (width * height))
    return ScreenCapture(
        raw_data=data,
        width=width,
        height=height,
        pixel_format="RGBA",
        bounds=WindowBounds(0, 0, width, height),
        metadata=metadata or {},
    )


def create_mock_observation(
    window_title: str = "Test App",
    process_name: str = "testapp.exe",
    text_lines: Optional[List[str]] = None,
    blocked: bool = False,
    block_reason: str = "",
) -> ScreenObservation:
    """Create a populated ScreenObservation for verification tests."""
    meta = {
        "window_title": window_title,
        "process_name": process_name,
        "blocked": blocked,
        "block_reason": block_reason,
    }
    cap = None if blocked else create_mock_capture(metadata=meta)
    blocks: List[OCRTextBlock] = []
    full_text = ""
    if text_lines:
        full_text = "\n".join(text_lines)
        for i, line in enumerate(text_lines):
            blocks.append(
                OCRTextBlock(
                    text=line,
                    confidence=0.95,
                    bounds=WindowBounds(10, 10 + i * 25, 200, 20),
                )
            )
    ocr_res = OCRResult(text=full_text, blocks=tuple(blocks)) if text_lines else None
    return ScreenObservation(
        capture=cap,
        ocr_result=ocr_res,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestVisualVerificationEngine:
    """Deterministic-first tests for VisualVerificationEngine."""

    def setup_method(self) -> None:
        self.mock_ocr = MockOCRProvider()
        self.engine = VisualVerificationEngine(ocr_provider=self.mock_ocr)

    def test_01_window_present_verified(self) -> None:
        """1. Verify WINDOW_PRESENT returns VERIFIED when window title matches."""
        obs = create_mock_observation(window_title="Untitled - Notepad", process_name="notepad.exe")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert res.confidence >= 0.8
        assert "Notepad" in res.explanation
        assert len(res.evidence_chain) >= 1
        assert res.evidence_chain[0].source == "window_title"

    def test_02_window_present_not_verified(self) -> None:
        """2. Verify WINDOW_PRESENT returns NOT_VERIFIED when window title does not match."""
        obs = create_mock_observation(window_title="Calculator", process_name="calc.exe")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.NOT_VERIFIED
        assert "not the active window" in res.explanation
        assert len(res.evidence_chain) >= 1

    def test_03_window_absent_verified(self) -> None:
        """3. Verify WINDOW_ABSENT returns VERIFIED when window is closed/no longer active."""
        obs = create_mock_observation(window_title="Desktop Explorer", process_name="explorer.exe")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_ABSENT, target="Notepad")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "no longer open" in res.explanation

    def test_04_window_absent_not_verified(self) -> None:
        """4. Verify WINDOW_ABSENT returns NOT_VERIFIED when window is still open."""
        obs = create_mock_observation(window_title="Untitled - Notepad", process_name="notepad.exe")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_ABSENT, target="Notepad")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.NOT_VERIFIED
        assert "still open" in res.explanation

    def test_05_text_present_verified(self) -> None:
        """5. Verify TEXT_PRESENT returns VERIFIED when OCR finds the target text."""
        obs = create_mock_observation(text_lines=["File Download Complete", "100% finished"])
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.TEXT_PRESENT, target="Download Complete")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert res.confidence >= 0.8
        assert "Download Complete" in res.explanation

    def test_06_text_absent_verified(self) -> None:
        """6. Verify TEXT_ABSENT returns VERIFIED when text is absent."""
        obs = create_mock_observation(text_lines=["All operations succeeded", "Status: OK"])
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.TEXT_ABSENT, target="Error 404")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "not present" in res.explanation

    def test_07_element_state_enabled(self) -> None:
        """7. Verify ELEMENT_STATE evaluates enabled button affordance."""
        bounds = WindowBounds(50, 100, 80, 30)
        elem = UIElement(
            name="Submit",
            element_type=UIElementType.BUTTON,
            bounds=bounds,
            center=bounds.center,
            confidence=0.9,
        )
        mock_affordance = mock.MagicMock()
        mock_affordance.inspect_element_state.return_value = ElementAffordance(
            element=elem,
            detected_state=ControlVisualState.ENABLED,
            primary_affordance=ControlAffordance.CLICKABLE,
            confidence=0.9,
            evidence="High contrast styling",
        )
        engine = VisualVerificationEngine(affordance_engine=mock_affordance)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="Submit",
            expected_state=ControlVisualState.ENABLED,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "enabled" in res.explanation

    def test_08_element_state_disabled(self) -> None:
        """8. Verify ELEMENT_STATE detects disabled button state correctly."""
        bounds = WindowBounds(50, 100, 80, 30)
        elem = UIElement(
            name="Submit",
            element_type=UIElementType.BUTTON,
            bounds=bounds,
            center=bounds.center,
            confidence=0.9,
        )
        mock_affordance = mock.MagicMock()
        mock_affordance.inspect_element_state.return_value = ElementAffordance(
            element=elem,
            detected_state=ControlVisualState.DISABLED,
            primary_affordance=ControlAffordance.READ_ONLY,
            confidence=0.88,
            evidence="Low contrast disabled styling",
        )
        engine = VisualVerificationEngine(affordance_engine=mock_affordance)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="Submit",
            expected_state=ControlVisualState.ENABLED,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.NOT_VERIFIED
        assert "disabled" in res.explanation

    def test_09_element_state_checked(self) -> None:
        """9. Verify ELEMENT_STATE detects checked checkbox state."""
        bounds = WindowBounds(20, 50, 150, 20)
        elem = UIElement(
            name="Agree to terms",
            element_type=UIElementType.CHECKBOX,
            bounds=bounds,
            center=bounds.center,
            confidence=0.92,
        )
        mock_affordance = mock.MagicMock()
        mock_affordance.inspect_element_state.return_value = ElementAffordance(
            element=elem,
            detected_state=ControlVisualState.CHECKED,
            primary_affordance=ControlAffordance.TOGGLEABLE,
            confidence=0.92,
            evidence="Checked glyph detected",
        )
        engine = VisualVerificationEngine(affordance_engine=mock_affordance)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="Agree to terms",
            expected_state=ControlVisualState.CHECKED,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "checked" in res.explanation

    def test_10_element_state_unchecked(self) -> None:
        """10. Verify ELEMENT_STATE detects unchecked checkbox state."""
        bounds = WindowBounds(20, 50, 150, 20)
        elem = UIElement(
            name="Agree to terms",
            element_type=UIElementType.CHECKBOX,
            bounds=bounds,
            center=bounds.center,
            confidence=0.90,
        )
        mock_affordance = mock.MagicMock()
        mock_affordance.inspect_element_state.return_value = ElementAffordance(
            element=elem,
            detected_state=ControlVisualState.UNCHECKED,
            primary_affordance=ControlAffordance.TOGGLEABLE,
            confidence=0.90,
            evidence="Unchecked box detected",
        )
        engine = VisualVerificationEngine(affordance_engine=mock_affordance)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="Agree to terms",
            expected_state=ControlVisualState.UNCHECKED,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "unchecked" in res.explanation

    def test_11_element_state_empty(self) -> None:
        """11. Verify ELEMENT_STATE detects empty input state."""
        bounds = WindowBounds(50, 50, 200, 30)
        elem = UIElement(
            name="Username",
            element_type=UIElementType.INPUT,
            bounds=bounds,
            center=bounds.center,
            confidence=0.90,
        )
        mock_affordance = mock.MagicMock()
        mock_affordance.inspect_element_state.return_value = ElementAffordance(
            element=elem,
            detected_state=ControlVisualState.EMPTY,
            primary_affordance=ControlAffordance.EDITABLE,
            confidence=0.88,
            evidence="No text inside input box",
        )
        engine = VisualVerificationEngine(affordance_engine=mock_affordance)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="Username",
            expected_state=ControlVisualState.EMPTY,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "empty" in res.explanation

    def test_12_container_present(self) -> None:
        """12. Verify CONTAINER_PRESENT detects dialog or form containers."""
        mock_scene = mock.MagicMock(spec=VisualSceneParser)
        container = UIContainer(
            container_id="cont-dlg",
            container_type=UIContainerType.DIALOG,
            bounds=WindowBounds(100, 100, 400, 300),
            label="Save As",
            confidence=0.92,
        )
        mock_scene.parse_scene.return_value = UIScene(
            scene_id="s1",
            observation_id="o1",
            window_title="Notepad",
            containers=(container,),
            confidence=0.92,
        )
        engine = VisualVerificationEngine(scene_parser=mock_scene)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.CONTAINER_PRESENT,
            target="Save As",
            container_type=UIContainerType.DIALOG,
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "dialog" in res.explanation.lower()

    def test_13_visual_delta_expected_change(self) -> None:
        """13. Verify VISUAL_DELTA verifies when meaningful change is detected."""
        mock_delta = mock.MagicMock(spec=VisualDeltaEngine)
        bounds = WindowBounds(10, 10, 80, 30)
        elem = UIElement(
            name="OK",
            element_type=UIElementType.BUTTON,
            bounds=bounds,
            center=bounds.center,
        )
        change = UIElementChange(
            change_type=VisualDeltaType.ELEMENT_APPEARED,
            after_element=elem,
            confidence=0.9,
        )
        delta_res = VisualDeltaResult(
            delta_id="d1",
            before_observation_id="o0",
            after_observation_id="o1",
            primary_change_type=VisualDeltaType.ELEMENT_APPEARED,
            element_changes=(change,),
            added_texts=("OK",),
            meaningful_change_detected=True,
            confidence=0.90,
            explanation="Button 'OK' appeared on screen.",
        )
        mock_delta.compare_observations.return_value = delta_res
        engine = VisualVerificationEngine(delta_engine=mock_delta)
        prior_obs = create_mock_observation(window_title="Screen 1")
        curr_obs = create_mock_observation(window_title="Screen 2")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.VISUAL_DELTA, target="OK")
        res = engine.verify_goal(spec, curr_obs, prior_observation=prior_obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "OK" in res.explanation

    def test_14_visual_delta_missing_change(self) -> None:
        """14. Verify VISUAL_DELTA returns NOT_VERIFIED when no changes occur."""
        mock_delta = mock.MagicMock(spec=VisualDeltaEngine)
        delta_res = VisualDeltaResult(
            delta_id="d1",
            before_observation_id="o0",
            after_observation_id="o1",
            primary_change_type=VisualDeltaType.NO_MEANINGFUL_CHANGE,
            meaningful_change_detected=False,
            confidence=0.95,
            explanation="No visual change detected.",
        )
        mock_delta.compare_observations.return_value = delta_res
        engine = VisualVerificationEngine(delta_engine=mock_delta)
        prior_obs = create_mock_observation()
        curr_obs = create_mock_observation()
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.VISUAL_DELTA, target="screen_change")
        res = engine.verify_goal(spec, curr_obs, prior_observation=prior_obs)

        assert res.outcome == VisualOutcomeType.NOT_VERIFIED
        assert "No visual changes" in res.explanation

    def test_15_insufficient_evidence_uncertain(self) -> None:
        """15. Verify insufficient evidence returns UNCERTAIN conservatively."""
        obs = create_mock_observation(window_title="", process_name="")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.UNCERTAIN
        assert "Uncertain" in res.explanation

    def test_16_conflicting_evidence_uncertain(self) -> None:
        """16. Verify missing baseline in delta verification returns UNCERTAIN."""
        obs = create_mock_observation()
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.VISUAL_DELTA, target="button")
        res = self.engine.verify_goal(spec, obs, prior_observation=None)

        assert res.outcome == VisualOutcomeType.UNCERTAIN
        assert "baseline" in res.explanation.lower()

    def test_17_blocked_capture_returns_blocked(self) -> None:
        """17. Verify security-blocked observation cleanly returns BLOCKED."""
        obs = create_mock_observation(blocked=True, block_reason="Active window matches sensitive pattern")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Bank")
        res = self.engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.BLOCKED
        assert "blocked by visual security policy" in res.explanation.lower()
        assert len(res.evidence_chain) == 1
        assert res.evidence_chain[0].source == "security_policy"

    def test_18_evidence_chain_generation(self) -> None:
        """18. Verify structured evidence chain items are created with calibrated confidence."""
        obs = create_mock_observation(window_title="Calculator", process_name="calc.exe")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Calculator")
        res = self.engine.verify_goal(spec, obs)

        assert len(res.evidence_chain) >= 1
        item = res.evidence_chain[0]
        assert isinstance(item, VisualEvidenceItem)
        assert item.source in ("window_title", "process_name", "ocr_text")
        assert 0.0 <= item.confidence <= 1.0

    def test_19_sanitized_evidence(self) -> None:
        """19. Verify evidence contains no passwords or raw OCR dumps."""
        obs = create_mock_observation(text_lines=["Password: supersecret123", "Login"])
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.TEXT_PRESENT, target="Login")
        res = self.engine.verify_goal(spec, obs)

        for item in res.evidence_chain:
            assert "supersecret123" not in item.description

    def test_20_no_raw_pixel_persistence(self) -> None:
        """20. Verify VisualVerificationResult contains zero raw pixel buffers or base64."""
        obs = create_mock_observation(window_title="App")
        spec = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="App")
        res = self.engine.verify_goal(spec, obs)

        d = res.to_dict()
        d_str = str(d)
        assert "raw_data" not in d
        assert "base64" not in d_str
        assert "image_bytes" not in d_str

    def test_21_no_sensitive_content_logging(self) -> None:
        """21. Verify VisualGoalSpec dictionary does not leak sensitive information."""
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.ELEMENT_STATE,
            target="PIN",
            expected_state=ControlVisualState.EMPTY,
        )
        d = spec.to_dict()
        assert "password" not in d

    def test_22_vlm_fallback(self) -> None:
        """22. Verify multimodal VLM fallback is invoked for CUSTOM_SEMANTIC queries."""
        mock_ai = mock.MagicMock()
        mock_ai.supports_multimodal = True
        mock_ai.generate.return_value = mock.MagicMock(
            content="VERIFIED: The graph clearly shows an upward trend."
        )
        engine = VisualVerificationEngine(ai_provider=mock_ai)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
            target="The graph shows an upward trend",
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.VERIFIED
        assert "upward trend" in res.explanation
        assert res.evaluation_source == "multimodal_vlm"

    def test_23_vlm_unavailable(self) -> None:
        """23. Verify VLM unavailable returns UNCERTAIN safely without exception."""
        engine = VisualVerificationEngine(ai_provider=None)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
            target="Is the painting realistic?",
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.UNCERTAIN
        assert "insufficient" in res.explanation.lower()

    def test_24_invalid_vlm_output(self) -> None:
        """24. Verify unparsable VLM output defaults to UNCERTAIN."""
        mock_ai = mock.MagicMock()
        mock_ai.supports_multimodal = True
        mock_ai.generate.return_value = mock.MagicMock(
            content="I am not sure what is visible in this frame."
        )
        engine = VisualVerificationEngine(ai_provider=mock_ai)
        obs = create_mock_observation()
        spec = VisualGoalSpec(
            criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
            target="Did the action finish?",
        )
        res = engine.verify_goal(spec, obs)

        assert res.outcome == VisualOutcomeType.UNCERTAIN


class TestVisionSkillsVerificationIntegration:
    """Integration tests for VisionSkills verify_screen_state and verify_goal."""

    def setup_method(self) -> None:
        self.mock_capture_engine = mock.MagicMock()
        self.mock_capture_engine.capture_active_window.return_value = create_mock_capture(
            metadata={"window_title": "Calculator", "process_name": "calc.exe"}
        )
        self.mock_sec_manager = mock.MagicMock(spec=SecureVisionManager)
        obs = ScreenObservation(
            capture=self.mock_capture_engine.capture_active_window.return_value,
            metadata={"window_title": "Calculator", "process_name": "calc.exe"},
        )
        self.mock_sec_manager.capture_active_window.return_value = obs
        self.mock_sec_manager.capture_screen.return_value = obs
        self.mock_sec_manager.get_latest_observation.return_value = obs

        self.skill = VisionSkills(
            secure_vision_manager=self.mock_sec_manager,
        )

    def test_25_verify_screen_state_backward_compatibility(self) -> None:
        """25. Verify string-based condition returns backward-compatible dict keys."""
        res = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "verify Calculator is open"},
        })

        assert res.success
        assert "verified" in res.data
        assert "status" in res.data
        assert "reason" in res.data
        assert "reused_cache" in res.data
        assert res.data["verified"] is True
        assert res.data["status"] == "verified"

    def test_26_natural_language_goal_parsing(self) -> None:
        """26. Verify parse_visual_goal maps natural-language patterns accurately."""
        # Window present
        spec1 = parse_visual_goal("verify Notepad is open")
        assert spec1.criterion == VisualGoalCriterion.WINDOW_PRESENT
        assert spec1.target.lower() == "notepad"

        # Window absent
        spec2 = parse_visual_goal("did calculator close?")
        assert spec2.criterion == VisualGoalCriterion.WINDOW_ABSENT
        assert spec2.target.lower() == "calculator"

        # Button enabled
        spec3 = parse_visual_goal("is the Submit button enabled?")
        assert spec3.criterion == VisualGoalCriterion.ELEMENT_STATE
        assert spec3.target.lower() == "submit"
        assert spec3.expected_state == ControlVisualState.ENABLED

        # Checkbox checked
        spec4 = parse_visual_goal("is the checkbox checked?")
        assert spec4.criterion == VisualGoalCriterion.ELEMENT_STATE
        assert spec4.expected_state == ControlVisualState.CHECKED

        # Dialog open
        spec5 = parse_visual_goal("is the login dialog open?")
        assert spec5.criterion == VisualGoalCriterion.CONTAINER_PRESENT
        assert spec5.container_type == UIContainerType.DIALOG

        # Delta query
        spec6 = parse_visual_goal("did the screen change?")
        assert spec6.criterion == VisualGoalCriterion.VISUAL_DELTA


class TestPlannerExecutorIntegration:
    """Tests for Task model and Executor post-condition visual verification."""

    def test_27_task_expected_visual_goal_default_none(self) -> None:
        """27. Verify Task.expected_visual_goal defaults to None."""
        task = Task(action="open_app", target="notepad")
        assert task.expected_visual_goal is None

        d = task.to_dict()
        assert "expected_visual_goal" not in d

        deserialized = Task.from_dict(d)
        assert deserialized.expected_visual_goal is None

    def test_28_executor_without_visual_goal_unchanged(self) -> None:
        """28. Verify Executor executes tasks without visual goal without verification overhead."""
        handler = mock.MagicMock(return_value="OK")
        executor = Executor(handlers={"custom_action": handler})
        task = Task(action="custom_action", target="test")

        res_task, success, res_str, err = executor.execute_task_sync(task)
        assert success
        assert res_task.status == TaskStatus.COMPLETED
        assert handler.call_count == 1

    def test_29_executor_with_visual_goal_verifies_post_condition(self) -> None:
        """29. Verify Executor executes handler and then verifies visual post-condition."""
        handler = mock.MagicMock(return_value="OK")
        mock_vision = mock.MagicMock(spec=VisionSkills)
        mock_vision.verify_goal.return_value = VisualVerificationResult(
            verification_id="v1",
            observation_id="o1",
            outcome=VisualOutcomeType.VERIFIED,
            goal_spec=VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"),
            confidence=0.95,
            explanation="Verified. Notepad is open.",
        )
        container = ServiceContainer()
        container.register_singleton("vision", mock_vision, allow_override=True)

        executor = Executor(container_instance=container, handlers={"open_app": handler})
        task = Task(
            action="open_app",
            target="notepad",
            expected_visual_goal=VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"
            ),
        )

        res_task, success, res_str, err = executor.execute_task_sync(task)
        assert success
        assert res_task.status == TaskStatus.COMPLETED
        assert mock_vision.verify_goal.call_count == 1
        assert "verification_result" in res_task.parameters

    def test_30_executor_not_verified_post_condition_fails_task(self) -> None:
        """30. Verify NOT_VERIFIED post-condition marks task as FAILED."""
        handler = mock.MagicMock(return_value="OK")
        mock_vision = mock.MagicMock(spec=VisionSkills)
        mock_vision.verify_goal.return_value = VisualVerificationResult(
            verification_id="v1",
            observation_id="o1",
            outcome=VisualOutcomeType.NOT_VERIFIED,
            goal_spec=VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"),
            confidence=0.90,
            explanation="Not verified. Notepad did not appear on screen.",
        )
        container = ServiceContainer()
        container.register_singleton("vision", mock_vision, allow_override=True)

        executor = Executor(container_instance=container, handlers={"open_app": handler})
        task = Task(
            action="open_app",
            target="notepad",
            expected_visual_goal=VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"
            ),
        )

        res_task, success, res_str, err = executor.execute_task_sync(task)
        assert not success
        assert res_task.status == TaskStatus.FAILED
        assert "NOT_VERIFIED" in str(err)

    def test_31_executor_uncertain_post_condition_fails_task(self) -> None:
        """31. Verify UNCERTAIN post-condition does not falsely mark task completed."""
        handler = mock.MagicMock(return_value="OK")
        mock_vision = mock.MagicMock(spec=VisionSkills)
        mock_vision.verify_goal.return_value = VisualVerificationResult(
            verification_id="v1",
            observation_id="o1",
            outcome=VisualOutcomeType.UNCERTAIN,
            goal_spec=VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"),
            confidence=0.40,
            explanation="Uncertain. Could not confirm whether window opened.",
        )
        container = ServiceContainer()
        container.register_singleton("vision", mock_vision, allow_override=True)

        executor = Executor(container_instance=container, handlers={"open_app": handler})
        task = Task(
            action="open_app",
            target="notepad",
            expected_visual_goal=VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Notepad"
            ),
        )

        res_task, success, res_str, err = executor.execute_task_sync(task)
        assert not success
        assert res_task.status == TaskStatus.FAILED
        assert "UNCERTAIN" in str(err)

    def test_32_executor_blocked_post_condition_fails_task(self) -> None:
        """32. Verify BLOCKED post-condition fails cleanly preserving security outcome."""
        handler = mock.MagicMock(return_value="OK")
        mock_vision = mock.MagicMock(spec=VisionSkills)
        mock_vision.verify_goal.return_value = VisualVerificationResult(
            verification_id="v1",
            observation_id="o1",
            outcome=VisualOutcomeType.BLOCKED,
            goal_spec=VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Banking"),
            confidence=1.0,
            explanation="Verification blocked by visual security policy.",
        )
        container = ServiceContainer()
        container.register_singleton("vision", mock_vision, allow_override=True)

        executor = Executor(container_instance=container, handlers={"open_app": handler})
        task = Task(
            action="open_app",
            target="banking",
            expected_visual_goal=VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Banking"
            ),
        )

        res_task, success, res_str, err = executor.execute_task_sync(task)
        assert not success
        assert res_task.status == TaskStatus.FAILED
        assert "BLOCKED" in str(err)

    def test_33_planner_integration(self) -> None:
        """33. Verify Planner allows tasks with expected visual goals."""
        planner = Planner()
        goal = VisualGoalSpec(criterion=VisualGoalCriterion.WINDOW_PRESENT, target="Chrome")
        task = Task(action="open_app", target="chrome", expected_visual_goal=goal)
        plan = Plan(query="open chrome", tasks=[task])

        assert len(plan.tasks) == 1
        assert plan.tasks[0].expected_visual_goal.target == "Chrome"

    def test_34_intent_router_classification(self) -> None:
        """34. Verify IntentRouter classifies visual verification queries into IntentType.VISION."""
        router = IntentRouter()
        queries = [
            "did notepad open",
            "is calculator open",
            "did chrome close",
            "is the submit button enabled",
            "is the checkbox checked",
            "did the error disappear",
            "verify the dialog",
            "check whether the form is complete",
        ]
        for q in queries:
            res = router.classify(q)
            assert res.intent == IntentType.VISION, f"Failed for query: '{q}' -> got {res.intent}"
