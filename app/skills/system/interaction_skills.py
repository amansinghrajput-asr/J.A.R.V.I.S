"""Visual Desktop Interaction System Skill for J.A.R.V.I.S. Phase 27.18.

Provides secure, verified desktop visual interaction operations:
- visual_click
- visual_double_click
- visual_type
- visual_clear_and_type
- visual_select
- visual_toggle
- visual_dismiss_modal
- visual_interact

Enforces full integration with:
- SystemSecurityPolicy (SAFE, CONFIRMATION_REQUIRED, RESTRICTED)
- SystemConfirmationManager (parameter-bound confirmation tokens)
- ActionPreflightValidator (15 fail-closed TOCTOU checks)
- VirtualInputBackend (MockInputBackend by default / in tests)
- Post-action visual goal verification
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from app.automation.input import MockInputBackend, VirtualInputBackend
from app.automation.visual_action_adapter import (
    ActionPreflightValidator,
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)
from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.vision.models import (
    Point,
    VisualActionFeasibilityStatus,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    WindowBounds,
)

logger = get_logger("SYSTEM.INTERACTION")


class InteractionSkills(BaseSystemSkill):
    """System skill exposing scoped, verified visual desktop interaction operations."""

    name: str = "interaction"
    description: str = "Verified desktop visual action execution bridge."
    priority: int = 70
    tags: list[str] = ["system", "vision", "interaction", "input"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
        settings: Optional[Settings] = None,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        action_adapter: Optional[VisualActionAdapter] = None,
        input_backend: Optional[VirtualInputBackend] = None,
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=settings,
            logger=get_logger("SKILL.INTERACTION"),
            container=container,
            event_bus=event_bus,
        )
        self._input_backend = input_backend or MockInputBackend()
        self._action_adapter = action_adapter or VisualActionAdapter(
            input_backend=self._input_backend,
            preflight_validator=ActionPreflightValidator(),
        )

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given visual interaction command."""
        op, _, _, _ = self.parse_command(command)
        return op in (
            "visual_click",
            "visual_double_click",
            "visual_type",
            "visual_clear_and_type",
            "visual_select",
            "visual_toggle",
            "visual_dismiss_modal",
            "visual_interact",
        )

    @property
    def action_adapter(self) -> VisualActionAdapter:
        """Active VisualActionAdapter instance."""
        return self._action_adapter

    @property
    def input_backend(self) -> VirtualInputBackend:
        """Active VirtualInputBackend instance."""
        return self._action_adapter.input_backend

    def set_input_backend(self, backend: VirtualInputBackend) -> None:
        """Inject an alternate input backend."""
        self._input_backend = backend
        self._action_adapter.set_input_backend(backend)

    # --------------------------------------------------------------------------
    # Target Parsing Helper
    # --------------------------------------------------------------------------

    def _extract_target(self, params: Dict[str, Any]) -> Optional[VisualActionTarget]:
        """Safely resolve VisualActionTarget from parameters or dictionaries."""
        raw_target = params.get("target") or params.get("action_target")
        if isinstance(raw_target, VisualActionTarget):
            return raw_target
        if isinstance(raw_target, dict):
            try:
                # Reconstruct Point and WindowBounds if dicts
                pt = None
                raw_pt = raw_target.get("target_point")
                if isinstance(raw_pt, Point):
                    pt = raw_pt
                elif isinstance(raw_pt, dict):
                    pt = Point(raw_pt.get("x", 0), raw_pt.get("y", 0))

                b = None
                raw_b = raw_target.get("bounds")
                if isinstance(raw_b, WindowBounds):
                    b = raw_b
                elif isinstance(raw_b, dict):
                    b = WindowBounds(
                        raw_b.get("left", 0),
                        raw_b.get("top", 0),
                        raw_b.get("right", 0),
                        raw_b.get("bottom", 0),
                    )

                act_type_str = str(raw_target.get("action_type") or "CLICK").upper()
                try:
                    act_type = VisualActionType(act_type_str)
                except ValueError:
                    act_type = VisualActionType.CLICK

                safety_str = str(raw_target.get("safety_tier") or "SAFE").upper()
                try:
                    safety_tier = VisualActionSafetyTier(safety_str)
                except ValueError:
                    safety_tier = VisualActionSafetyTier.MUTATING

                feasibility_str = str(raw_target.get("feasibility") or "FEASIBLE").upper()
                try:
                    feasibility = VisualActionFeasibilityStatus(feasibility_str)
                except ValueError:
                    feasibility = VisualActionFeasibilityStatus.FEASIBLE

                return VisualActionTarget(
                    target_id=str(raw_target.get("target_id") or "target_0"),
                    action_type=act_type,
                    target_element_name=str(raw_target.get("target_element_name") or "control"),
                    target_point=pt,
                    bounds=b,
                    safety_tier=safety_tier,
                    feasibility=feasibility,
                    requires_confirmation=bool(raw_target.get("requires_confirmation", False)),
                    confidence=float(raw_target.get("confidence", 1.0)),
                    reason=str(raw_target.get("reason", "")),
                    expected_outcome=raw_target.get("expected_outcome"),
                    metadata=dict(raw_target.get("metadata") or {}),
                    window_handle=raw_target.get("window_handle"),
                    grounded_at=float(raw_target.get("grounded_at", time.monotonic())),
                    input_text=raw_target.get("input_text"),
                )
            except Exception as exc:
                self._logger.warning("Failed to deserialize VisualActionTarget: %s", exc)
                return None
        return None

    # --------------------------------------------------------------------------
    # Command Dispatcher
    # --------------------------------------------------------------------------

    def execute(self, command: Any) -> SystemSkillResult:
        """Execute interaction command with security validation and preflight checks."""
        op, target_arg, params, conf_id = self.parse_command(command)

        if not op:
            raise SkillExecutionError("No valid operation specified in command payload.")

        # 1. Resolve Target
        action_target = self._extract_target(params)

        # 2. Check Security Policy
        tier, reason = self.security_policy.validate_operation(op, target_arg, params)

        # Elevate tier if target specifies DESTRUCTIVE or MUTATING
        if action_target is not None:
            if action_target.safety_tier == VisualActionSafetyTier.DESTRUCTIVE:
                tier = SystemSafetyTier.CONFIRMATION_REQUIRED
                reason = reason or f"Destructive visual action on '{action_target.target_element_name}' requires confirmation."
            elif action_target.feasibility == VisualActionFeasibilityStatus.UNCERTAIN:
                raise SecurityPolicyViolationError(
                    f"Visual action on '{action_target.target_element_name}' has UNCERTAIN feasibility status and cannot be executed."
                )
            elif action_target.feasibility == VisualActionFeasibilityStatus.SENSITIVE_PROTECTED:
                raise SecurityPolicyViolationError(
                    f"Visual action on '{action_target.target_element_name}' is SENSITIVE_PROTECTED and strictly prohibited."
                )
            elif action_target.safety_tier == VisualActionSafetyTier.SAFE and not action_target.requires_confirmation:
                if tier == SystemSafetyTier.CONFIRMATION_REQUIRED and "unrecognized operation" in (reason or "").lower():
                    tier = SystemSafetyTier.SAFE

        if tier == SystemSafetyTier.RESTRICTED:
            raise SecurityPolicyViolationError(
                reason or f"Operation '{op}' is RESTRICTED by system security policy."
            )

        # 3. Confirmation Handling
        confirmation_verified = False
        requires_conf = (
            tier == SystemSafetyTier.CONFIRMATION_REQUIRED
            or (action_target is not None and action_target.requires_confirmation)
            or bool(conf_id)
        )

        if requires_conf:
            if not conf_id:
                # Bind exact operation and parameters for hash verification
                desc = reason or f"Confirmation required for visual action '{op}'."
                issued_id = self.confirmation_manager.request_confirmation(
                    operation=op,
                    target=target_arg or (action_target.target_element_name if action_target else None),
                    parameters=params,
                    description=desc,
                )
                if self.confirmation_manager.is_approved(issued_id):
                    self.confirmation_manager.verify_and_consume(
                        confirmation_id=issued_id,
                        operation=op,
                        target=target_arg or (action_target.target_element_name if action_target else None),
                        parameters=params,
                    )
                    confirmation_verified = True
                else:
                    raise ConfirmationRequiredError(
                        f"Action '{op}' requires explicit confirmation. Token: '{issued_id}'.",
                        confirmation_id=issued_id,
                        operation=op,
                        target=target_arg,
                    )
            else:
                # Validate provided confirmation token
                self.confirmation_manager.verify_and_consume(
                    confirmation_id=conf_id,
                    operation=op,
                    target=target_arg or (action_target.target_element_name if action_target else None),
                    parameters=params,
                )
                confirmation_verified = True

        # 4. Dispatch to VisualActionAdapter
        current_win = params.get("current_window_info")
        current_sit = params.get("current_situation")
        current_scn = params.get("current_scene")

        result = self._action_adapter.execute_target(
            action_target,
            confirmation_verified=confirmation_verified,
            current_window_info=current_win,
            current_situation=current_sit,
            current_scene=current_scn,
        )

        # 5. Return SystemSkillResult
        return SystemSkillResult(
            operation=op,
            success=result.success,
            data=result.to_dict(),
            error=result.reason if not result.success else None,
            duration_ms=result.duration_ms,
            metadata={"target": target_arg or (action_target.target_element_name if action_target else None)},
        )
