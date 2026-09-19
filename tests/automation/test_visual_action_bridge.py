"""Comprehensive Verification Suite for Visual Action Execution Bridge & Verified Interaction.

Phase 27.18 - Visual Action Execution Bridge & Verified Interaction.

Covers:
- Category A: Zero real input assertion (MockInputBackend event count > 0, Win32 count == 0)
- Category B: TTL and monotonic timestamp validation
- Category C: Window matching and foreground switch TOCTOU prevention
- Category D: Geometry shift and boundary containment checks
- Category E: Control state, disabled checks, and ambiguity rejection
- Category F: Modal dialog isolation and modal dismissal handling
- Category G: Security policies, confirmation gateway tokens, and sensitive context rejection
- Category H: Privacy safeguards and secret text redaction
- Category I: Post-action visual goal verification integration
- Category J: Backward compatibility with BaseSystemSkill and Executor
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import pytest

from app.automation.input import (
    MockInputBackend,
    VirtualInputBackend,
    Win32InputBackend,
)
from app.automation.visual_action_adapter import (
    ActionPreflightValidator,
    PreflightCheckResult,
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)
from app.skills.system.interaction_skills import InteractionSkills
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.vision.models import (
    Point,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    VisualActionFeasibilityStatus,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualSituation,
    VisualSituationType,
    WindowBounds,
)


@pytest.fixture(autouse=True)
def ensure_zero_real_win32_calls() -> None:
    """Strict test safeguard: Win32 real input is never invoked."""
    Win32InputBackend.reset_invocation_count()
    Win32InputBackend.enable_test_safety_guard()
    yield
    assert Win32InputBackend.invocation_count == 0, (
        f"CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked {Win32InputBackend.invocation_count} times!"
    )


def make_target(
    action_type: VisualActionType = VisualActionType.CLICK,
    name: str = "Submit Button",
    point: Optional[Point] = Point(200, 300),
    bounds: Optional[WindowBounds] = WindowBounds(150, 250, 250, 350),
    safety_tier: VisualActionSafetyTier = VisualActionSafetyTier.SAFE,
    feasibility: VisualActionFeasibilityStatus = VisualActionFeasibilityStatus.FEASIBLE,
    requires_confirmation: bool = False,
    confidence: float = 0.95,
    window_handle: Optional[int] = 12345,
    grounded_at: Optional[float] = None,
    input_text: Optional[str] = None,
    expected_outcome: Optional[VisualGoalSpec] = None,
) -> VisualActionTarget:
    """Helper creating test VisualActionTarget instances."""
    return VisualActionTarget(
        target_id="act_test_001",
        action_type=action_type,
        target_element_name=name,
        target_point=point,
        bounds=bounds,
        safety_tier=safety_tier,
        feasibility=feasibility,
        requires_confirmation=requires_confirmation,
        confidence=confidence,
        reason="Target is feasible.",
        expected_outcome=expected_outcome,
        metadata={"element_name": name},
        window_handle=window_handle,
        grounded_at=grounded_at if grounded_at is not None else time.monotonic(),
        input_text=input_text,
    )


class TestVisualActionBridge:
    """Test suite for ActionPreflightValidator, VisualActionAdapter, and InteractionSkills."""

    # --------------------------------------------------------------------------
    # Category A: Zero Real Input Assertion
    # --------------------------------------------------------------------------

    def test_01_zero_real_input_assertion(self) -> None:
        """Verify MockInputBackend processes events and Win32 backend invocation count is 0."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        target = make_target(action_type=VisualActionType.CLICK)

        res = adapter.execute_target(target)
        assert res.success is True
        assert res.status == VisualActionResultStatus.SUCCESS

        # Explicit assertion: MockInputBackend recorded event
        assert mock.event_count("click") == 1
        # Explicit assertion: Real Win32 backend remained untouched
        assert Win32InputBackend.invocation_count == 0

    # --------------------------------------------------------------------------
    # Category B: Monotonic TTL Validation
    # --------------------------------------------------------------------------

    def test_02_ttl_valid_passes(self) -> None:
        """Target grounded within TTL window passes preflight."""
        validator = ActionPreflightValidator(max_ttl_seconds=5.0)
        target = make_target(grounded_at=time.monotonic() - 1.0)

        chk = validator.validate(target)
        assert chk.passed is True

    def test_03_ttl_expired_fails_closed(self) -> None:
        """Target grounded beyond TTL window fails closed with PREFLIGHT_STALE_COORDINATES."""
        validator = ActionPreflightValidator(max_ttl_seconds=5.0)
        # Grounded 10 seconds ago
        target = make_target(grounded_at=time.monotonic() - 10.0)

        chk = validator.validate(target)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES
        assert "stale" in chk.reason.lower()

    # --------------------------------------------------------------------------
    # Category C: Window & Foreground TOCTOU Checks
    # --------------------------------------------------------------------------

    def test_04_matching_foreground_window_passes(self) -> None:
        """Active window matching target HWND passes preflight."""
        validator = ActionPreflightValidator()
        target = make_target(window_handle=44556)
        win_info = {"hwnd": 44556, "bounds": WindowBounds(100, 100, 800, 600)}

        chk = validator.validate(target, current_window_info=win_info)
        assert chk.passed is True

    def test_05_switched_foreground_window_fails_closed(self) -> None:
        """Active window HWND mismatch fails closed with PREFLIGHT_WINDOW_MISMATCH."""
        validator = ActionPreflightValidator()
        target = make_target(window_handle=44556)
        # Another window (e.g. terminal or browser) stole foreground
        win_info = {"hwnd": 99999, "bounds": WindowBounds(100, 100, 800, 600)}

        chk = validator.validate(target, current_window_info=win_info)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH
        assert "switched" in chk.reason.lower()

    # --------------------------------------------------------------------------
    # Category D: Geometry & Boundary Containment
    # --------------------------------------------------------------------------

    def test_06_window_geometry_shift_fails_closed(self) -> None:
        """Target window resized or moved significantly fails with PREFLIGHT_WINDOW_MISMATCH."""
        validator = ActionPreflightValidator(bounds_tolerance_px=20.0)
        target = make_target(
            window_handle=100,
            point=Point(200, 200),
            bounds=WindowBounds(100, 100, 300, 300),
        )
        object.__setattr__(target, "metadata", {"window_bounds": WindowBounds(0, 0, 500, 500)})
        # Window resized to 800x800
        win_info = {"hwnd": 100, "bounds": WindowBounds(0, 0, 800, 800)}

        chk = validator.validate(target, current_window_info=win_info)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH
        assert "shifted incompatibly" in chk.reason

    def test_07_point_outside_control_bounds_fails(self) -> None:
        """Target point lying outside element bounds fails with PREFLIGHT_STALE_COORDINATES."""
        validator = ActionPreflightValidator()
        # Point (999, 999) is outside control bounds (10, 10, 100, 50)
        target = make_target(
            point=Point(999, 999),
            bounds=WindowBounds(10, 10, 100, 50),
        )

        chk = validator.validate(target)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES
        assert "outside control bounds" in chk.reason

    # --------------------------------------------------------------------------
    # Category E: Control State, Disabled Checks, and Ambiguity
    # --------------------------------------------------------------------------

    def test_08_disabled_control_in_current_scene_fails(self) -> None:
        """Target control that became disabled fails closed with PRECONDITION_FAILED."""
        validator = ActionPreflightValidator()
        target = make_target(
            name="Save",
            point=Point(50, 30),
            bounds=WindowBounds(10, 10, 100, 50),
        )

        # Current scene has the same element marked disabled in metadata
        disabled_el = UIElement(
            name="Save",
            element_type=UIElementType.BUTTON,
            bounds=WindowBounds(10, 10, 100, 50),
            center=Point(55, 30),
            metadata={"is_enabled": False},
        )
        scene = UIScene(
            scene_id="s1",
            observation_id="o1",
            window_title="App",
            window_bounds=WindowBounds(0, 0, 1000, 1000),
            interactive_elements=(disabled_el,),
        )

        chk = validator.validate(target, current_scene=scene)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PRECONDITION_FAILED
        assert "disabled" in chk.reason.lower()

    def test_09_ambiguous_target_confidence_fails(self) -> None:
        """Low confidence target resolution fails closed with PRECONDITION_FAILED."""
        validator = ActionPreflightValidator(min_confidence=0.7)
        target = make_target(confidence=0.4)

        chk = validator.validate(target)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PRECONDITION_FAILED
        assert "confidence" in chk.reason.lower()

    def test_10_non_feasible_target_fails(self) -> None:
        """Target with feasibility other than FEASIBLE fails preflight."""
        validator = ActionPreflightValidator()
        target = make_target(feasibility=VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED)

        chk = validator.validate(target)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PRECONDITION_FAILED

    # --------------------------------------------------------------------------
    # Category F: Modal Dialog State Checks
    # --------------------------------------------------------------------------

    def test_11_target_inside_modal_passes(self) -> None:
        """Target control located inside active modal container passes preflight."""
        validator = ActionPreflightValidator()
        modal = UIContainer(
            container_id="c_modal",
            container_type=UIContainerType.DIALOG,
            bounds=WindowBounds(100, 100, 500, 500),
        )
        sit = VisualSituation(
            situation_id="sit_modal_1",
            timestamp=time.time(),
            observation_id="obs_modal_1",
            situation_type=VisualSituationType.MODAL_DIALOG,
            active_modal=modal,
        )
        # Target is at (200, 200), inside (100, 100, 500, 500)
        target = make_target(
            point=Point(200, 200),
            bounds=WindowBounds(150, 150, 250, 250),
        )

        chk = validator.validate(target, current_situation=sit)
        assert chk.passed is True

    def test_12_target_outside_modal_fails(self) -> None:
        """Target control outside active modal fails closed with PREFLIGHT_MODAL_CHANGED."""
        validator = ActionPreflightValidator()
        modal = UIContainer(
            container_id="c_modal",
            container_type=UIContainerType.DIALOG,
            bounds=WindowBounds(400, 400, 800, 800),
        )
        sit = VisualSituation(
            situation_id="sit_modal_2",
            timestamp=time.time(),
            observation_id="obs_modal_2",
            situation_type=VisualSituationType.MODAL_DIALOG,
            active_modal=modal,
        )
        # Target is at (50, 50), outside modal
        target = make_target(
            point=Point(50, 50),
            bounds=WindowBounds(10, 10, 90, 90),
        )

        chk = validator.validate(target, current_situation=sit)
        assert chk.passed is False
        assert chk.failure_status == VisualActionResultStatus.PREFLIGHT_MODAL_CHANGED
        assert "modal" in chk.reason.lower()

    def test_13_dismiss_modal_action_passes(self) -> None:
        """DISMISS_MODAL action permitted even when modal is active."""
        validator = ActionPreflightValidator()
        modal = UIContainer(
            container_id="c_modal",
            container_type=UIContainerType.DIALOG,
            bounds=WindowBounds(400, 400, 800, 800),
        )
        sit = VisualSituation(
            situation_id="sit_modal_3",
            timestamp=time.time(),
            observation_id="obs_modal_3",
            situation_type=VisualSituationType.MODAL_DIALOG,
            active_modal=modal,
        )
        # Close button on the modal header
        target = make_target(
            action_type=VisualActionType.DISMISS_MODAL,
            name="Close",
            point=Point(780, 420),
            bounds=WindowBounds(760, 410, 790, 430),
        )

        chk = validator.validate(target, current_situation=sit)
        assert chk.passed is True

    # --------------------------------------------------------------------------
    # Category G: Security Policies & Confirmation Gateway
    # --------------------------------------------------------------------------

    def test_14_safe_action_proceeds_without_confirmation(self) -> None:
        """Safe visual click executes directly."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        target = make_target(
            safety_tier=VisualActionSafetyTier.SAFE,
            requires_confirmation=False,
        )

        res = adapter.execute_target(target)
        assert res.success is True
        assert mock.event_count("click") == 1

    def test_15_mutating_destructive_action_unconfirmed_fails(self) -> None:
        """Destructive action without verified confirmation fails with CONFIRMATION_REQUIRED."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        target = make_target(
            safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
            requires_confirmation=True,
        )

        res = adapter.execute_target(target, confirmation_verified=False)
        assert res.success is False
        assert res.status == VisualActionResultStatus.CONFIRMATION_REQUIRED
        assert mock.event_count() == 0  # Zero input dispatched!

    def test_16_destructive_action_with_verified_confirmation_executes(self) -> None:
        """Destructive action with verified confirmation token executes cleanly."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        target = make_target(
            safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
            requires_confirmation=True,
        )

        res = adapter.execute_target(target, confirmation_verified=True)
        assert res.success is True
        assert mock.event_count("click") == 1

    def test_17_sensitive_context_transition_fails_before_dispatch(self) -> None:
        """Active window or situation transitioning to sensitive fails with SECURITY_BLOCKED."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        target = make_target()

        # Window transitioned to sensitive password dialog
        win_info = {"hwnd": 12345, "is_sensitive": True}

        res = adapter.execute_target(target, current_window_info=win_info)
        assert res.success is False
        assert res.status == VisualActionResultStatus.SECURITY_BLOCKED
        assert mock.event_count() == 0

    # --------------------------------------------------------------------------
    # Category H: Input Text Privacy & Redaction
    # --------------------------------------------------------------------------

    def test_18_input_text_privacy_invariants(self) -> None:
        """Input text payload is never leaked into logs, summaries, or serialized results."""
        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock)
        secret_text = "MasterSecretPassword_999!"
        target = make_target(
            action_type=VisualActionType.TYPE_TEXT,
            name="Password Box",
            point=Point(200, 200),
            bounds=WindowBounds(150, 150, 250, 250),
            input_text=secret_text,
        )

        res = adapter.execute_target(target)
        assert res.success is True
        assert mock.event_count("type_text") == 1

        # Check VisualActionResult serialization
        res_dict = res.to_dict()
        assert secret_text not in str(res)
        assert secret_text not in str(res_dict)

        # Check metadata guarantees
        meta = res_dict["metadata"]
        assert meta["input_present"] is True
        assert meta["input_length"] == len(secret_text)
        assert meta["input_redacted"] is True

        # Check Target serialization
        target_dict = target.to_dict()
        assert secret_text not in str(target_dict)
        assert target_dict["input_present"] is True
        assert target_dict["input_length"] == len(secret_text)

    # --------------------------------------------------------------------------
    # Category I: Post-Action Visual Goal Verification Integration
    # --------------------------------------------------------------------------

    def test_19_verification_success(self) -> None:
        """Post-action verification VERIFIED returns SUCCESS."""
        class MockVerifier:
            def verify_goal(self, goal: Any, target: Optional[str] = None) -> Dict[str, Any]:
                return {"outcome": "VERIFIED", "reason": "Element state changed as expected"}

        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock, vision_engine=MockVerifier())
        goal = VisualGoalSpec(criterion=VisualGoalCriterion.CONTAINER_PRESENT, target="Saved")
        target = make_target(expected_outcome=goal)

        res = adapter.execute_target(target)
        assert res.success is True
        assert res.status == VisualActionResultStatus.SUCCESS
        assert res.verification_dict is not None
        assert res.verification_dict["outcome"] == "VERIFIED"

    def test_20_verification_failure_not_verified(self) -> None:
        """Post-action verification NOT_VERIFIED marks action as VERIFICATION_FAILED."""
        class MockVerifier:
            def verify_goal(self, goal: Any, target: Optional[str] = None) -> Dict[str, Any]:
                return {"outcome": "NOT_VERIFIED", "reason": "Element state did not change"}

        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock, vision_engine=MockVerifier())
        goal = VisualGoalSpec(criterion=VisualGoalCriterion.CONTAINER_PRESENT, target="State")
        target = make_target(expected_outcome=goal)

        res = adapter.execute_target(target)
        assert res.success is False
        assert res.status == VisualActionResultStatus.VERIFICATION_FAILED
        assert "NOT_VERIFIED" in res.reason

    def test_21_verification_uncertain(self) -> None:
        """Post-action verification UNCERTAIN marks action as VERIFICATION_UNCERTAIN."""
        class MockVerifier:
            def verify_goal(self, goal: Any, target: Optional[str] = None) -> Dict[str, Any]:
                return {"outcome": "UNCERTAIN", "reason": "Screen was occluded"}

        mock = MockInputBackend()
        adapter = VisualActionAdapter(input_backend=mock, vision_engine=MockVerifier())
        goal = VisualGoalSpec(criterion=VisualGoalCriterion.CONTAINER_PRESENT, target="State")
        target = make_target(expected_outcome=goal)

        res = adapter.execute_target(target)
        assert res.success is False
        assert res.status == VisualActionResultStatus.VERIFICATION_UNCERTAIN
        assert "UNCERTAIN" in res.reason

    # --------------------------------------------------------------------------
    # Category J: InteractionSkills System Integration
    # --------------------------------------------------------------------------

    def test_22_interaction_skills_safe_click(self) -> None:
        """InteractionSkills executes safe visual click through BaseSystemSkill standard pipeline."""
        mock = MockInputBackend()
        skill = InteractionSkills(input_backend=mock)
        target = make_target(action_type=VisualActionType.CLICK, name="OK Button")

        cmd = {
            "operation": "visual_click",
            "target": "OK Button",
            "parameters": {"target": target},
        }
        res = skill.execute(cmd)
        assert res.success is True
        assert res.operation == "visual_click"
        assert mock.event_count("click") == 1

    def test_23_interaction_skills_confirmation_required_flow(self) -> None:
        """InteractionSkills enforces confirmation gateway for destructive visual targets."""
        mock = MockInputBackend()
        sec = SystemSecurityPolicy()
        conf = SystemConfirmationManager()
        skill = InteractionSkills(
            input_backend=mock,
            security_policy=sec,
            confirmation_manager=conf,
        )

        target = make_target(
            action_type=VisualActionType.CLICK,
            name="Format Drive",
            safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
            requires_confirmation=True,
        )
        cmd = {
            "operation": "visual_click",
            "target": "Format Drive",
            "parameters": {"target": target},
        }

        # First call without confirmation token raises ConfirmationRequiredError
        with pytest.raises(ConfirmationRequiredError) as exc_info:
            skill.execute(cmd)

        token = exc_info.value.confirmation_id
        assert bool(token)
        assert mock.event_count() == 0  # Input not dispatched

        # Approve token through gateway
        conf.resolve_confirmation(token, approved=True)

        # Second call with confirmed token succeeds
        cmd["confirmation_id"] = token
        res = skill.execute(cmd)
        assert res.success is True
        assert mock.event_count("click") == 1

    def test_24_interaction_skills_tampered_parameters_fails(self) -> None:
        """Altering target coordinates or parameters after confirmation invalidates token."""
        mock = MockInputBackend()
        sec = SystemSecurityPolicy()
        conf = SystemConfirmationManager()
        skill = InteractionSkills(
            input_backend=mock,
            security_policy=sec,
            confirmation_manager=conf,
        )

        target = make_target(
            name="Delete File",
            point=Point(100, 100),
            safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
            requires_confirmation=True,
        )
        cmd = {
            "operation": "visual_click",
            "target": "Delete File",
            "parameters": {"target": target},
        }

        with pytest.raises(ConfirmationRequiredError) as exc_info:
            skill.execute(cmd)

        token = exc_info.value.confirmation_id
        conf.resolve_confirmation(token, approved=True)

        # Tamper: modify coordinate point in parameters
        tampered_target = make_target(
            name="Delete File",
            point=Point(999, 999),  # TAMPERED
            safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
            requires_confirmation=True,
        )
        cmd["confirmation_id"] = token
        cmd["parameters"]["target"] = tampered_target

        # Token parameter hash mismatch raises ConfirmationRequiredError or rejection
        with pytest.raises(Exception):
            skill.execute(cmd)

        assert mock.event_count() == 0  # Zero dispatch!
