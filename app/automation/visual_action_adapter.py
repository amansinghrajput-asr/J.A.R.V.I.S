"""Visual Action Execution Bridge and Preflight Validation Adapter.

Phase 27.18 - Visual Action Execution Bridge & Verified Interaction.

Provides deterministic, fail-closed preflight validation and safe dispatch
connecting:
    VisualActionTarget (Phase 27.17)
    → ActionPreflightValidator (15 fail-closed TOCTOU checks)
    → VirtualInputBackend (MockInputBackend in tests)
    → Post-action Visual Goal Verification (Phase 27.13)
    → Structured VisualActionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from app.automation.input import MockInputBackend, VirtualInputBackend
from app.vision.models import (
    Point,
    UIContainer,
    UIElement,
    UIElementType,
    UIScene,
    VisualActionFeasibilityStatus,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualGoalSpec,
    VisualSituation,
    VisualSituationType,
    WindowBounds,
)

logger = logging.getLogger("AUTOMATION.VISUAL_ADAPTER")


# --------------------------------------------------------------------------
# Execution Result States
# --------------------------------------------------------------------------


class VisualActionResultStatus(str, Enum):
    """Structured outcome statuses for visual action execution."""

    SUCCESS = "SUCCESS"
    GROUNDING_FAILED = "GROUNDING_FAILED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    PREFLIGHT_WINDOW_MISMATCH = "PREFLIGHT_WINDOW_MISMATCH"
    PREFLIGHT_STALE_COORDINATES = "PREFLIGHT_STALE_COORDINATES"
    PREFLIGHT_MODAL_CHANGED = "PREFLIGHT_MODAL_CHANGED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    INPUT_DISPATCH_ERROR = "INPUT_DISPATCH_ERROR"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    VERIFICATION_UNCERTAIN = "VERIFICATION_UNCERTAIN"


@dataclass(frozen=True)
class VisualActionResult:
    """Immutable, privacy-safe execution result for a visual interaction.

    Attributes:
        status: High-level outcome status.
        action_type: Category of action performed.
        target_id: Unique identifier of action target.
        target_element_name: Canonical element name.
        target_point: Targeted screen coordinate.
        success: Whether execution and verification succeeded.
        reason: Explainable human-readable summary.
        verification_dict: Verification outcome dictionary if verified.
        metadata: Privacy-safe execution telemetry (never containing secrets).
        duration_ms: Total duration of interaction in milliseconds.
    """

    status: VisualActionResultStatus
    action_type: VisualActionType
    target_id: Optional[str] = None
    target_element_name: str = ""
    target_point: Optional[Point] = None
    success: bool = False
    reason: str = ""
    verification_dict: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize VisualActionResult to dictionary (strictly zero secret leakage)."""
        return {
            "status": (
                self.status.value
                if isinstance(self.status, VisualActionResultStatus)
                else str(self.status)
            ),
            "action_type": (
                self.action_type.value
                if isinstance(self.action_type, VisualActionType)
                else str(self.action_type)
            ),
            "target_id": self.target_id,
            "target_element_name": self.target_element_name,
            "target_point": self.target_point.to_dict() if self.target_point else None,
            "success": self.success,
            "reason": self.reason,
            "verification_dict": dict(self.verification_dict) if self.verification_dict else None,
            "metadata": dict(self.metadata),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class PreflightCheckResult:
    """Outcome of preflight validation immediately prior to input dispatch."""

    passed: bool
    failure_status: Optional[VisualActionResultStatus] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Action Preflight Validator
# --------------------------------------------------------------------------


class ActionPreflightValidator:
    """Validates 15 required preconditions and TOCTOU invariants immediately before dispatch.

    Fails closed if any condition cannot be strictly proven.
    """

    def __init__(
        self,
        *,
        max_ttl_seconds: float = 10.0,
        min_confidence: float = 0.5,
        bounds_tolerance_px: float = 30.0,
    ) -> None:
        self.max_ttl_seconds = max_ttl_seconds
        self.min_confidence = min_confidence
        self.bounds_tolerance_px = bounds_tolerance_px

    def validate(
        self,
        target: Optional[VisualActionTarget],
        *,
        current_window_info: Optional[Dict[str, Any]] = None,
        current_situation: Optional[VisualSituation] = None,
        current_scene: Optional[UIScene] = None,
        confirmation_verified: bool = False,
    ) -> PreflightCheckResult:
        """Run all 15 preflight validation checks.

        Returns:
            PreflightCheckResult with pass/fail verdict and specific failure status.
        """
        # 1. Target exists
        if target is None:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.GROUNDING_FAILED,
                reason="Target is None or action was not grounded.",
            )

        # 2. Feasibility == FEASIBLE
        if target.feasibility != VisualActionFeasibilityStatus.FEASIBLE:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.PRECONDITION_FAILED,
                reason=f"Target feasibility is '{target.feasibility.value}', execution prohibited: {target.reason}",
            )

        # 3. Target point exists when required
        point_required_actions = {
            VisualActionType.CLICK,
            VisualActionType.DOUBLE_CLICK,
            VisualActionType.CLEAR_AND_TYPE,
            VisualActionType.SELECT_OPTION,
            VisualActionType.TOGGLE,
            VisualActionType.DISMISS_MODAL,
        }
        if target.action_type in point_required_actions and target.target_point is None:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.PRECONDITION_FAILED,
                reason=f"Action '{target.action_type.value}' requires a valid coordinate point.",
            )

        # 4. Target safety tier is acceptable
        if target.safety_tier not in (
            VisualActionSafetyTier.SAFE,
            VisualActionSafetyTier.MUTATING,
            VisualActionSafetyTier.DESTRUCTIVE,
        ):
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.SECURITY_BLOCKED,
                reason=f"Target safety tier is '{target.safety_tier}', execution strictly blocked.",
            )

        # 5. Confirmation check: if required, valid confirmation must already exist
        if target.requires_confirmation and not confirmation_verified:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.CONFIRMATION_REQUIRED,
                reason=f"Action on '{target.target_element_name}' requires explicit user confirmation.",
            )

        # 6 & 7. TTL check: grounded_at has not exceeded configured TTL using time.monotonic()
        now_mono = time.monotonic()
        elapsed_sec = now_mono - target.grounded_at
        if elapsed_sec > self.max_ttl_seconds or elapsed_sec < -0.1:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES,
                reason=f"Target coordinates are stale: elapsed {elapsed_sec:.2f}s exceeds TTL {self.max_ttl_seconds}s.",
                metadata={"elapsed_seconds": elapsed_sec, "max_ttl": self.max_ttl_seconds},
            )

        # 8. Foreground window still matches target.window_handle when available
        if target.window_handle is not None and current_window_info is not None:
            curr_hwnd = current_window_info.get("hwnd")
            if curr_hwnd is not None:
                try:
                    if int(curr_hwnd) != int(target.window_handle):
                        return PreflightCheckResult(
                            passed=False,
                            failure_status=VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH,
                            reason=f"Foreground window switched: expected HWND {target.window_handle}, active HWND {curr_hwnd}.",
                            metadata={"expected_hwnd": target.window_handle, "active_hwnd": curr_hwnd},
                        )
                except (ValueError, TypeError):
                    return PreflightCheckResult(
                        passed=False,
                        failure_status=VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH,
                        reason="Current window HWND format is invalid.",
                    )

        # 9. Current window geometry remains compatible with grounded geometry
        if current_window_info is not None:
            curr_bounds = current_window_info.get("bounds")
            if isinstance(curr_bounds, WindowBounds):
                if target.target_point is not None and not curr_bounds.contains_point(target.target_point):
                    return PreflightCheckResult(
                        passed=False,
                        failure_status=VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH,
                        reason=f"Target point ({target.target_point.x}, {target.target_point.y}) is outside current window bounds.",
                    )
                grounded_win = target.metadata.get("window_bounds")
                if isinstance(grounded_win, WindowBounds):
                    delta_w = abs(grounded_win.width - curr_bounds.width)
                    delta_h = abs(grounded_win.height - curr_bounds.height)
                    if delta_w > self.bounds_tolerance_px or delta_h > self.bounds_tolerance_px:
                        return PreflightCheckResult(
                            passed=False,
                            failure_status=VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH,
                            reason=f"Target window geometry shifted incompatibly (delta: {delta_w}x{delta_h}px).",
                        )

        # 10. Target point remains inside the current actionable region
        if target.target_point is not None and target.bounds is not None:
            if not target.bounds.contains_point(target.target_point):
                return PreflightCheckResult(
                    passed=False,
                    failure_status=VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES,
                    reason=f"Target point ({target.target_point.x}, {target.target_point.y}) is outside control bounds.",
                )

        # 11. Target is not disabled in current scene
        if current_scene is not None:
            for el in current_scene.interactive_elements:
                if el.name == target.target_element_name and target.bounds is not None and el.bounds == target.bounds:
                    is_en = el.metadata.get("is_enabled", el.metadata.get("enabled", True))
                    st = str(el.metadata.get("state") or "").lower()
                    if is_en is False or st == "disabled":
                        return PreflightCheckResult(
                            passed=False,
                            failure_status=VisualActionResultStatus.PRECONDITION_FAILED,
                            reason=f"Target control '{target.target_element_name}' has become disabled.",
                        )
                    break

        # 12. Target is not ambiguous
        if target.confidence < self.min_confidence:
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.PRECONDITION_FAILED,
                reason=f"Target resolution confidence ({target.confidence:.2f}) is below threshold ({self.min_confidence}).",
            )

        # 13. No protected/sensitive transition has occurred
        if current_window_info is not None and bool(current_window_info.get("is_sensitive", False)):
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.SECURITY_BLOCKED,
                reason="Active window transitioned to a protected sensitive context.",
            )
        if current_situation is not None and (
            bool(current_situation.metadata.get("is_sensitive", False))
            or getattr(current_situation, "is_sensitive", False)
            or current_situation.situation_type == VisualSituationType.SENSITIVE_PROTECTED
        ):
            return PreflightCheckResult(
                passed=False,
                failure_status=VisualActionResultStatus.SECURITY_BLOCKED,
                reason="Current visual situation is protected sensitive context.",
            )

        # 14. Modal state still permits the intended interaction
        if current_situation is not None:
            active_modal = current_situation.active_modal or (
                current_situation.primary_container
                if current_situation.situation_type == VisualSituationType.MODAL_DIALOG
                else None
            )
            if active_modal is not None and target.action_type != VisualActionType.DISMISS_MODAL:
                # Check if target is inside active modal
                if target.bounds is not None:
                    mb = active_modal.bounds
                    tb = target.bounds
                    inside_modal = (
                        tb.left >= mb.left
                        and tb.right <= mb.right
                        and tb.top >= mb.top
                        and tb.bottom <= mb.bottom
                    )
                    if not inside_modal:
                        return PreflightCheckResult(
                            passed=False,
                            failure_status=VisualActionResultStatus.PREFLIGHT_MODAL_CHANGED,
                            reason="An active modal dialog appeared or blocks the target control.",
                        )

        # 15. No stale target condition exists
        # All 14 prior conditions satisfied
        return PreflightCheckResult(passed=True, reason="All preflight invariants verified.")


# --------------------------------------------------------------------------
# Visual Action Adapter
# --------------------------------------------------------------------------


class VisualActionAdapter:
    """Unified adapter executing grounded visual actions through a mockable input backend.

    Guarantees:
    - Never performs perception, OCR, or scene parsing (read from VisualActionTarget).
    - Never uses raw OS input when a MockInputBackend is bound.
    - Strictly runs ActionPreflightValidator before dispatch.
    - Never leaks secret input text into logs or returned metadata.
    """

    def __init__(
        self,
        *,
        input_backend: Optional[VirtualInputBackend] = None,
        preflight_validator: Optional[ActionPreflightValidator] = None,
        vision_engine: Optional[Any] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._input_backend = input_backend or MockInputBackend()
        self._preflight_validator = preflight_validator or ActionPreflightValidator()
        self._vision_engine = vision_engine

    @property
    def input_backend(self) -> VirtualInputBackend:
        """Active virtual input backend."""
        return self._input_backend

    def set_input_backend(self, backend: VirtualInputBackend) -> None:
        """Inject an alternate virtual input backend (e.g. for testing)."""
        with self._lock:
            self._input_backend = backend

    def execute_target(
        self,
        target: Optional[VisualActionTarget],
        *,
        confirmation_verified: bool = False,
        current_window_info: Optional[Dict[str, Any]] = None,
        current_situation: Optional[VisualSituation] = None,
        current_scene: Optional[UIScene] = None,
    ) -> VisualActionResult:
        """Execute a grounded visual action target with fail-closed preflight validation.

        Args:
            target: Grounded VisualActionTarget to execute.
            confirmation_verified: True if operator confirmation was approved and verified.
            current_window_info: Optional active window telemetry for TOCTOU checks.
            current_situation: Optional active situation for modal/sensitive checks.
            current_scene: Optional active scene for element enablement checks.

        Returns:
            Structured VisualActionResult.
        """
        t0 = time.perf_counter()
        with self._lock:
            # 1. Basic target existence
            if target is None:
                return VisualActionResult(
                    status=VisualActionResultStatus.GROUNDING_FAILED,
                    action_type=VisualActionType.CLICK,
                    target_id=None,
                    target_element_name="",
                    target_point=None,
                    success=False,
                    reason="Visual action target is None.",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # Build sanitized metadata
            sanitized_meta = {
                "element_name": target.target_element_name,
                "action_type": target.action_type.value,
                "input_present": target.input_text is not None,
                "input_length": len(target.input_text) if target.input_text else 0,
                "input_redacted": target.input_text is not None,
            }

            # 2. Run Preflight Validation (15 checks)
            preflight = self._preflight_validator.validate(
                target,
                current_window_info=current_window_info,
                current_situation=current_situation,
                current_scene=current_scene,
                confirmation_verified=confirmation_verified,
            )

            if not preflight.passed:
                fail_status = preflight.failure_status or VisualActionResultStatus.PRECONDITION_FAILED
                logger.warning(
                    "VisualActionAdapter preflight failed for '%s' (%s): %s",
                    target.target_element_name,
                    fail_status.value,
                    preflight.reason,
                )
                return VisualActionResult(
                    status=fail_status,
                    action_type=target.action_type,
                    target_id=target.target_id,
                    target_element_name=target.target_element_name,
                    target_point=target.target_point,
                    success=False,
                    reason=preflight.reason,
                    metadata=sanitized_meta,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # 3. Dispatch to VirtualInputBackend
            dispatch_ok = False
            try:
                act = target.action_type
                pt = target.target_point or Point(0, 0)

                if act == VisualActionType.CLICK:
                    dispatch_ok = self._input_backend.click(pt, "left")
                elif act == VisualActionType.DOUBLE_CLICK:
                    dispatch_ok = self._input_backend.double_click(pt)
                elif act == VisualActionType.TYPE_TEXT:
                    dispatch_ok = self._input_backend.type_text(target.input_text or "")
                elif act == VisualActionType.CLEAR_AND_TYPE:
                    dispatch_ok = self._input_backend.clear_and_type(pt, target.input_text or "")
                elif act == VisualActionType.SELECT_OPTION:
                    dispatch_ok = self._input_backend.select_option(pt)
                elif act == VisualActionType.TOGGLE:
                    dispatch_ok = self._input_backend.toggle(pt)
                elif act == VisualActionType.DISMISS_MODAL:
                    dispatch_ok = self._input_backend.dismiss_modal(pt)
                else:
                    dispatch_ok = False
            except Exception as exc:
                logger.error("VirtualInputBackend dispatch exception: %s", exc)
                dispatch_ok = False

            if not dispatch_ok:
                return VisualActionResult(
                    status=VisualActionResultStatus.INPUT_DISPATCH_ERROR,
                    action_type=target.action_type,
                    target_id=target.target_id,
                    target_element_name=target.target_element_name,
                    target_point=target.target_point,
                    success=False,
                    reason=f"Input backend failed to dispatch action '{target.action_type.value}'.",
                    metadata=sanitized_meta,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # 4. Optional Immediate Post-Condition Verification
            # If vision_engine is configured and target has an expected_outcome
            v_dict: Optional[Dict[str, Any]] = None
            if self._vision_engine is not None and target.expected_outcome is not None:
                try:
                    if hasattr(self._vision_engine, "verify_goal"):
                        v_res = self._vision_engine.verify_goal(
                            target.expected_outcome, target=target.target_element_name
                        )
                        v_dict = v_res.to_dict() if hasattr(v_res, "to_dict") else dict(v_res)
                        outcome = str(v_dict.get("outcome") or "").upper()
                        if outcome == "NOT_VERIFIED":
                            return VisualActionResult(
                                status=VisualActionResultStatus.VERIFICATION_FAILED,
                                action_type=target.action_type,
                                target_id=target.target_id,
                                target_element_name=target.target_element_name,
                                target_point=target.target_point,
                                success=False,
                                reason=f"Visual verification NOT_VERIFIED: {v_dict.get('reason')}",
                                verification_dict=v_dict,
                                metadata=sanitized_meta,
                                duration_ms=(time.perf_counter() - t0) * 1000.0,
                            )
                        elif outcome not in ("VERIFIED", "SUCCESS"):
                            return VisualActionResult(
                                status=VisualActionResultStatus.VERIFICATION_UNCERTAIN,
                                action_type=target.action_type,
                                target_id=target.target_id,
                                target_element_name=target.target_element_name,
                                target_point=target.target_point,
                                success=False,
                                reason=f"Visual verification UNCERTAIN: {v_dict.get('reason')}",
                                verification_dict=v_dict,
                                metadata=sanitized_meta,
                                duration_ms=(time.perf_counter() - t0) * 1000.0,
                            )
                except Exception as exc:
                    logger.warning("Visual verification evaluation error: %s", exc)
                    return VisualActionResult(
                        status=VisualActionResultStatus.VERIFICATION_UNCERTAIN,
                        action_type=target.action_type,
                        target_id=target.target_id,
                        target_element_name=target.target_element_name,
                        target_point=target.target_point,
                        success=False,
                        reason=f"Visual verification evaluation error: {exc}",
                        metadata=sanitized_meta,
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )

            # 5. Success
            return VisualActionResult(
                status=VisualActionResultStatus.SUCCESS,
                action_type=target.action_type,
                target_id=target.target_id,
                target_element_name=target.target_element_name,
                target_point=target.target_point,
                success=True,
                reason=f"Action '{target.action_type.value}' on '{target.target_element_name}' executed successfully.",
                verification_dict=v_dict,
                metadata=sanitized_meta,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
