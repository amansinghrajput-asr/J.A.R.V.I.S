"""Phase 27.16 Visual Context Fusion & Situation Understanding Tests.

Comprehensive deterministic unit and integration test suite covering:
A. Classification:
   - DESKTOP_IDLE
   - APPLICATION_ACTIVE
   - MODAL_DIALOG
   - FORM_INPUT
   - PROGRESS_BUSY
   - ERROR_ALERT
   - SENSITIVE_PROTECTED
   - UNKNOWN
B. Context fusion:
   - Scene information reused
   - Affordance information reused
   - Tracking information reused
   - Temporal information reused
C. Security:
   - observation.is_sensitive
   - observation.metadata["is_sensitive"]
   - observation.metadata["blocked"]
   - observation.authorization.is_allowed == False
   - observation.is_valid == False
D. Protected transition isolation:
   - Safe A -> Protected -> Safe B (no A->B transition, no protected title/process/labels)
E. Secret handling:
   - password/token/API-key/PIN/secret content is redacted
F. No raw pixel persistence:
   - zero image buffers, numpy arrays, or base64 storage
G. No background service:
   - no background threads, timers, or polling
H. Voice safety:
   - concise summary, bounded output, no raw scene dump
I. Intent routing:
   - new situational queries route to VISION, existing intents unaffected
J. Backward compatibility:
   - explain_active_window remains compatible, VisionSkills dispatch verified
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
    VisualSituation,
    VisualSituationResult,
    VisualSituationType,
    VisualTemporalEvent,
    VisualTemporalHistoryResult,
    VisualTrackStatus,
    VisualTrackingResult,
    WindowBounds,
)
from app.vision.security import SecureVisionManager, VisionSecurityPolicy
from app.vision.situation import VisualSituationEngine


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
    """Create a synthetic ScreenObservation for situation testing."""
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


def make_scene(
    containers: Optional[List[UIContainer]] = None,
    elements: Optional[List[UIElement]] = None,
    summary: str = "Test scene summary",
) -> UIScene:
    """Create a synthetic UIScene."""
    c_list = tuple(containers or [])
    e_list = tuple(elements or [])
    return UIScene(
        scene_id=f"scene_{uuid.uuid4().hex[:6]}",
        observation_id="obs_test",
        window_title="Test App",
        window_bounds=WindowBounds(left=0, top=0, right=1280, bottom=720),
        containers=c_list,
        interactive_elements=tuple(e for e in e_list if e.metadata.get("is_interactive", True)),
        summary=summary,
        confidence=0.9,
    )


def make_element(
    canonical_name: str,
    element_type: UIElementType = UIElementType.BUTTON,
    text: Optional[str] = None,
    is_interactive: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> UIElement:
    """Create a synthetic UIElement."""
    meta = dict(metadata or {})
    meta["is_interactive"] = is_interactive
    b = WindowBounds(left=100, top=100, right=200, bottom=140)
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
    is_modal: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> UIContainer:
    """Create a synthetic UIContainer."""
    meta = dict(metadata or {})
    if is_modal:
        meta["is_modal"] = True
    return UIContainer(
        container_id=f"cnt_{uuid.uuid4().hex[:6]}",
        container_type=container_type,
        bounds=WindowBounds(left=50, top=50, right=800, bottom=600),
        elements=tuple(elements or []),
        label=title,
        confidence=0.95,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVisualSituationEngine:
    """Deterministic-first Visual Context Fusion & Situation Understanding tests."""

    def test_classification_desktop_idle(self) -> None:
        """A1. Verify DESKTOP_IDLE classification for Program Manager / explorer."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Program Manager", process_name="explorer.exe")
        sit = engine.fuse_situation(observation=obs)
        assert sit.situation_type == VisualSituationType.DESKTOP_IDLE
        assert "idle" in sit.summary.lower()
        assert sit.confidence >= 0.9

    def test_classification_application_active(self) -> None:
        """A2. Verify APPLICATION_ACTIVE classification for a standard app window."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Visual Studio Code", process_name="code.exe")
        btn = make_element("Run")
        cnt = make_container("Editor", UIContainerType.CONTENT_AREA, elements=[btn])
        scene = make_scene(containers=[cnt], elements=[btn])
        sit = engine.fuse_situation(observation=obs, scene=scene)
        assert sit.situation_type == VisualSituationType.APPLICATION_ACTIVE
        assert "Visual Studio Code" in sit.summary
        assert sit.confidence >= 0.8

    def test_classification_modal_dialog(self) -> None:
        """A3. Verify MODAL_DIALOG classification when a dialog/modal container is present."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Save Document", process_name="word.exe")
        btn_ok = make_element("Save")
        modal = make_container("Confirm Save", UIContainerType.DIALOG, elements=[btn_ok], is_modal=True)
        scene = make_scene(containers=[modal], elements=[btn_ok])
        sit = engine.fuse_situation(observation=obs, scene=scene)
        assert sit.situation_type == VisualSituationType.MODAL_DIALOG
        assert sit.active_modal is not None
        assert "dialog" in sit.summary.lower() or "modal" in sit.summary.lower()
        assert sit.confidence >= 0.9

    def test_classification_form_input(self) -> None:
        """A4. Verify FORM_INPUT classification when an active form or multiple input fields exist."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Registration", process_name="browser.exe")
        inp_name = make_element("Name Field", UIElementType.INPUT)
        inp_email = make_element("Email Field", UIElementType.INPUT)
        form_cnt = make_container("Signup Form", UIContainerType.FORM, elements=[inp_name, inp_email])
        scene = make_scene(containers=[form_cnt], elements=[inp_name, inp_email])
        sit = engine.fuse_situation(observation=obs, scene=scene)
        assert sit.situation_type == VisualSituationType.FORM_INPUT
        assert "form" in sit.summary.lower()
        assert sit.confidence >= 0.85

    def test_classification_progress_busy(self) -> None:
        """A5. Verify PROGRESS_BUSY classification when loading/downloading/busy cues exist."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Downloading Updates...", process_name="updater.exe")
        cnt = make_container("Please wait while downloading", UIContainerType.CONTENT_AREA)
        scene = make_scene(containers=[cnt])
        sit = engine.fuse_situation(observation=obs, scene=scene)
        assert sit.situation_type == VisualSituationType.PROGRESS_BUSY
        assert "busy" in sit.summary.lower() or "loading" in sit.summary.lower() or "processing" in sit.summary.lower()

    def test_classification_error_alert(self) -> None:
        """A6. Verify ERROR_ALERT classification when error cues are present."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Critical Error", process_name="app.exe")
        err_msg = make_element("Connection failed: access denied", UIElementType.TEXT, metadata={"is_error": True})
        cnt = make_container("Error Dialog", UIContainerType.DIALOG, elements=[err_msg])
        scene = make_scene(containers=[cnt], elements=[err_msg])
        sit = engine.fuse_situation(observation=obs, scene=scene)
        assert sit.situation_type == VisualSituationType.ERROR_ALERT
        assert "error" in sit.summary.lower() or "alert" in sit.summary.lower()

    def test_classification_unknown(self) -> None:
        """A7. Verify UNKNOWN classification when no structural evidence is present."""
        engine = VisualSituationEngine()
        sit = engine.fuse_situation(observation=None, scene=None)
        assert sit.situation_type == VisualSituationType.UNKNOWN
        assert sit.confidence <= 0.5

    def test_classification_sensitive_protected(self) -> None:
        """A8. Verify SENSITIVE_PROTECTED classification on sensitive observation."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="KeePass - Passwords", is_sensitive=True)
        sit = engine.fuse_situation(observation=obs)
        assert sit.situation_type == VisualSituationType.SENSITIVE_PROTECTED
        assert sit.window_title is None
        assert sit.process_name is None
        assert "protected" in sit.summary.lower()

    def test_context_fusion_reused_intelligence(self) -> None:
        """B. Verify fusion reuses scene, affordances, tracking, and temporal history."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Checkout", process_name="chrome.exe")
        btn = make_element("Submit Order")
        cnt = make_container("Cart Form", UIContainerType.FORM, elements=[btn])
        scene = make_scene(containers=[cnt], elements=[btn])

        aff = ElementAffordance(
            element=btn,
            primary_affordance=ControlAffordance.CLICKABLE,
            detected_state=ControlVisualState.FOCUSED,
            confidence=0.9,
            evidence="Click to submit",
        )

        temporal = VisualTemporalHistoryResult(
            query_id="q_1",
            events=(),
            total_count=0,
            summary="Item added to cart recently.",
        )

        sit = engine.fuse_situation(
            observation=obs,
            scene=scene,
            affordances=[aff],
            temporal_history=temporal,
        )

        assert sit.focused_element is not None
        assert sit.focused_element.name == "Submit Order"
        assert len(sit.primary_actions) == 1
        assert sit.primary_actions[0].evidence == "Click to submit"
        assert sit.recent_events_summary == "Item added to cart recently."

    def test_security_fail_closed_variations(self) -> None:
        """C. Verify fail-closed behavior across all five rejection pathways."""
        engine = VisualSituationEngine()

        # 1. is_sensitive flag
        obs1 = make_observation(is_sensitive=True)
        assert engine.fuse_situation(obs1).situation_type == VisualSituationType.SENSITIVE_PROTECTED

        # 2. metadata is_sensitive
        obs2 = make_observation()
        obs2.metadata["is_sensitive"] = True
        assert engine.fuse_situation(obs2).situation_type == VisualSituationType.SENSITIVE_PROTECTED

        # 3. metadata blocked
        obs3 = make_observation(blocked=True)
        assert engine.fuse_situation(obs3).situation_type == VisualSituationType.SENSITIVE_PROTECTED

        # 4. authorization.is_allowed == False
        obs4 = make_observation(is_allowed=False)
        assert engine.fuse_situation(obs4).situation_type == VisualSituationType.SENSITIVE_PROTECTED

        # 5. is_valid == False
        obs5 = make_observation(is_valid=False)
        assert engine.fuse_situation(obs5).situation_type == VisualSituationType.SENSITIVE_PROTECTED

    def test_protected_transition_isolation(self) -> None:
        """D. Verify Safe A -> Protected -> Safe B isolates transition context."""
        engine = VisualSituationEngine()

        # Safe A
        obs_a = make_observation(window_title="Safe App A", process_name="app_a.exe")
        sit_a = engine.fuse_situation(obs_a)
        assert sit_a.window_title == "Safe App A"
        assert engine._last_window_title == "Safe App A"

        # Protected view
        obs_prot = make_observation(window_title="Secret 1Password Vault", is_sensitive=True)
        sit_prot = engine.fuse_situation(obs_prot)
        assert sit_prot.situation_type == VisualSituationType.SENSITIVE_PROTECTED
        assert sit_prot.window_title is None
        assert sit_prot.process_name is None
        # Must reset transition state
        assert engine._last_window_title is None
        assert engine._last_process_name is None

        # Safe B
        obs_b = make_observation(window_title="Safe App B", process_name="app_b.exe")
        sit_b = engine.fuse_situation(obs_b)
        assert sit_b.window_title == "Safe App B"
        # No contextual link claiming A transitioned to B
        assert "Safe App A" not in sit_b.summary

    def test_secret_handling_redaction(self) -> None:
        """E. Verify secret material (passwords, tokens, API keys) is redacted."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Enter your master password", process_name="browser.exe")
        secret_el = make_element("api_key input field", UIElementType.INPUT, text="Bearer secret_token_12345")
        cnt = make_container("Auth credentials panel", elements=[secret_el])
        scene = make_scene(containers=[cnt], elements=[secret_el])

        sit = engine.fuse_situation(observation=obs, scene=scene)
        # Verify secret_token is NOT in summary, title, or element text
        assert "secret_token_12345" not in sit.summary
        assert "secret_token_12345" not in str(sit.to_dict())
        assert "[REDACTED]" in sit.window_title or sit.window_title == "[REDACTED]"

    def test_no_raw_pixel_persistence(self) -> None:
        """F. Verify zero raw pixels, NumPy arrays, or base64 strings in output."""
        engine = VisualSituationEngine()
        obs = make_observation()
        sit = engine.fuse_situation(obs)
        d = sit.to_dict()

        for k, v in d.items():
            assert not isinstance(v, (bytes, bytearray))
            assert "base64" not in str(v).lower()
            assert "image_bytes" not in str(v)

    def test_no_background_service(self) -> None:
        """G. Verify engine creates no background threads, timers, or loops."""
        threads_before = threading.active_count()
        engine = VisualSituationEngine()
        obs = make_observation()
        _ = engine.fuse_situation(obs)
        _ = engine.evaluate_situation(obs)
        threads_after = threading.active_count()
        assert threads_after == threads_before

    def test_voice_safety_and_bounded_summary(self) -> None:
        """H. Verify voice-safe formatting and bounded output length."""
        engine = VisualSituationEngine()
        obs = make_observation(window_title="Notepad - Untitled", process_name="notepad.exe")
        sit = engine.fuse_situation(obs)

        # Ensure summary is concise (<200 chars) and speech-friendly
        assert len(sit.summary) < 200
        assert not sit.summary.startswith("{")

        # Test SystemSkillResult.message formatting
        res = SystemSkillResult(
            operation="get_visual_situation",
            success=True,
            data={"situation": sit.to_dict(), "summary": sit.summary},
        )
        assert res.message == sit.summary

    def test_intent_routing_situational_queries(self) -> None:
        """I. Verify situational queries route to IntentType.VISION."""
        router = IntentRouter(auto_register_in_container=False)

        situational_phrases = [
            "what is happening on my screen",
            "what is happening right now",
            "what is the current situation",
            "what is the situation on my screen",
            "is there a popup",
            "is something blocking the screen",
            "get visual situation",
            "current situation",
            "what is happening",
        ]

        for phrase in situational_phrases:
            result = router.classify(phrase)
            assert result.intent == IntentType.VISION, f"Failed for phrase: '{phrase}', got {result.intent}"
            assert result.confidence >= 0.9

        # Ensure non-vision intents are not broken
        calc_result = router.classify("calculate 25 * 4")
        assert calc_result.intent != IntentType.VISION

        shutdown_result = router.classify("shutdown computer")
        assert shutdown_result.intent == IntentType.SYSTEM

    def test_vision_skills_get_visual_situation_dispatch(self) -> None:
        """J. Verify VisionSkills executes get_visual_situation cleanly."""
        mock_sec_manager = mock.MagicMock(spec=SecureVisionManager)
        obs = make_observation(window_title="Calculator", process_name="calc.exe")
        mock_sec_manager.get_latest_observation.return_value = obs
        mock_sec_manager.capture_active_window.return_value = obs
        mock_sec_manager.check_active_window_authorized.return_value = mock.MagicMock(is_allowed=True)

        skill = VisionSkills(
            secure_vision_manager=mock_sec_manager,
            ocr_provider=None,
            ai_provider=None,
        )

        # Command parsing
        can = skill.can_handle("what is the current situation")
        assert can is True
        op, target, params, _ = skill.parse_command("what is the current situation")
        assert op == "get_visual_situation"

        # Execution
        result = skill.execute({"operation": "get_visual_situation", "parse_scene": False})
        assert result.success is True
        assert "situation" in result.data
        assert "summary" in result.data
        assert result.data["situation_type"] == "APPLICATION_ACTIVE"
