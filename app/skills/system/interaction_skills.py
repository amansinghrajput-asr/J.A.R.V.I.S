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

from app.ai.planner.models import GROUNDED_PARAM_KEYS, purge_physical_state
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
        vision_skills: Optional[Any] = None,
        adapter: Optional[VisualActionAdapter] = None,
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
        effective_adapter = action_adapter or adapter
        self._action_adapter = effective_adapter or VisualActionAdapter(
            input_backend=self._input_backend,
            preflight_validator=ActionPreflightValidator(),
        )
        self._vision_skills = vision_skills
        if getattr(self._action_adapter, "_vision_engine", None) is None and vision_skills is not None:
            self._action_adapter._vision_engine = vision_skills

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

    @property
    def vision_skills(self) -> Optional[Any]:
        """Active VisionSkills instance for authorized screen observation and grounding."""
        if self._vision_skills is not None:
            return self._vision_skills
        if self.container is not None and hasattr(self.container, "exists") and hasattr(self.container, "resolve"):
            if self.container.exists("vision"):
                return self.container.resolve("vision")
            if self.container.exists("vision_skills"):
                return self.container.resolve("vision_skills")
        try:
            from app.skills.system.vision_skills import VisionSkills
            self._vision_skills = VisionSkills(container=self.container)
            return self._vision_skills
        except Exception:
            return None

    def _ground_semantic_target(
        self, op: str, target_query: str, params: Dict[str, Any], force_fresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Ground a natural semantic target query string into a VisualActionTarget."""
        vs = self.vision_skills
        if vs is None:
            self._logger.warning("Could not resolve VisionSkills for grounding '%s'.", target_query)
            return None

        if getattr(self._action_adapter, "_vision_engine", None) is None:
            self._action_adapter._vision_engine = vs

        op_map = {
            "visual_click": "CLICK",
            "visual_double_click": "DOUBLE_CLICK",
            "visual_type": "TYPE_TEXT",
            "visual_clear_and_type": "CLEAR_AND_TYPE",
            "visual_select": "SELECT_OPTION",
            "visual_toggle": "TOGGLE",
            "visual_dismiss_modal": "DISMISS_MODAL",
            "visual_interact": "CLICK",
        }
        act_type_str = op_map.get(op, "CLICK")

        grounding_params = dict(params)
        grounding_params["intent"] = target_query
        grounding_params["target"] = target_query
        grounding_params["action_type"] = act_type_str
        if force_fresh:
            grounding_params["force_fresh"] = True
            grounding_params["bypass_cache"] = True
            grounding_params.pop("cached_observation", None)
            grounding_params.pop("observation", None)
            grounding_params.pop("current_window_info", None)
            grounding_params.pop("current_situation", None)
            grounding_params.pop("current_scene", None)
        input_text = params.get("input_text") or params.get("text")
        if input_text:
            grounding_params["input_text"] = input_text

        try:
            skill_res = vs.execute({
                "operation": "ground_visual_action",
                "target": target_query,
                "parameters": grounding_params,
            })
            data = skill_res.data if hasattr(skill_res, "data") else (skill_res or {})
            if not isinstance(data, dict):
                return None

            raw_target = data.get("target") or data.get("action_target")
            target_obj = self._extract_target({"target": raw_target})
            if target_obj is None and isinstance(raw_target, VisualActionTarget):
                target_obj = raw_target

            # Bind input_text if specified in params and not yet on target_obj
            if target_obj is not None and input_text and not target_obj.input_text:
                target_obj = VisualActionTarget(
                    target_id=target_obj.target_id,
                    action_type=target_obj.action_type,
                    target_element_name=target_obj.target_element_name,
                    target_point=target_obj.target_point,
                    bounds=target_obj.bounds,
                    safety_tier=target_obj.safety_tier,
                    feasibility=target_obj.feasibility,
                    requires_confirmation=target_obj.requires_confirmation,
                    confidence=target_obj.confidence,
                    reason=target_obj.reason,
                    expected_outcome=target_obj.expected_outcome or data.get("expected_visual_goal"),
                    metadata=dict(target_obj.metadata),
                    window_handle=target_obj.window_handle,
                    grounded_at=target_obj.grounded_at,
                    input_text=input_text,
                )
            elif target_obj is not None and target_obj.expected_outcome is None:
                evg = data.get("expected_visual_goal") or data.get("expected_outcome")
                if evg is not None:
                    target_obj.expected_outcome = evg

            win_info = data.get("window_info") or data.get("current_window_info")
            sit_info = data.get("situation") or data.get("current_situation")
            scn_info = data.get("scene") or data.get("current_scene")

            return {
                "target": target_obj,
                "status": data.get("status"),
                "reason": data.get("summary") or data.get("reason"),
                "current_window_info": win_info,
                "current_situation": sit_info,
                "current_scene": scn_info,
            }
        except Exception as exc:
            self._logger.warning("Visual action grounding failed for '%s': %s", target_query, exc)
            return None

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

        if op and not op.startswith("visual_"):
            op = f"visual_{op}"

        # Check if this execution is a recovery attempt
        is_recovery = bool(
            params.get("is_recovery")
            or params.get("wave", 1) > 1
            or params.get("attempt", 1) > 1
            or (isinstance(command, dict) and (command.get("is_recovery") or command.get("wave", 1) > 1 or command.get("attempt", 1) > 1))
        )

        # 1. Resolve Target
        action_target = self._extract_target(params)
        if action_target is None and isinstance(command, dict):
            action_target = self._extract_target(command)

        has_workflow = bool(
            params.get("workflow_context")
            or (isinstance(command, dict) and command.get("workflow_context"))
        )
        stale_target_leak = bool(
            action_target is not None
            and target_arg
            and action_target.target_element_name
            and action_target.target_element_name.lower().strip() != target_arg.lower().strip()
        )

        # In recovery wave: ALWAYS discard pre-existing / stale VisualActionTarget & old confirmation
        if is_recovery:
            if action_target is not None:
                if not target_arg and action_target.target_element_name:
                    target_arg = action_target.target_element_name
                action_target = None
            conf_id = None
            params.pop("confirmation_id", None)
            params.pop("confirmation_token", None)
            purge_physical_state(params)
        elif has_workflow or stale_target_leak or (self._vision_skills is not None and target_arg):
            # In multi-step workflow context, or when target mismatch indicates leakage,
            # or when an explicit vision engine is injected to resolve semantic target queries:
            # discard pre-existing physical targets so fresh grounding is always performed
            if action_target is not None:
                if not target_arg and action_target.target_element_name:
                    target_arg = action_target.target_element_name
                action_target = None
            purge_physical_state(params)
        else:
            # If no explicit VisualActionTarget is provided (semantic execution),
            # purge any residual physical state from params to prevent leakage
            if action_target is None:
                purge_physical_state(params)

        current_win = params.get("current_window_info") if not is_recovery else None
        current_sit = params.get("current_situation") if not is_recovery else None
        current_scn = params.get("current_scene") if not is_recovery else None

        # Ground semantic target string with fresh observation
        if action_target is None and (target_arg or params.get("target") or params.get("element")):
            target_query = str(target_arg or params.get("target") or params.get("element") or "").strip()
            if target_query:
                ground_result = self._ground_semantic_target(op, target_query, params, force_fresh=True)
                if ground_result is not None:
                    action_target = ground_result.get("target")
                    if ground_result.get("current_window_info"):
                        current_win = ground_result.get("current_window_info")
                    if ground_result.get("current_situation"):
                        current_sit = ground_result.get("current_situation")
                    if ground_result.get("current_scene"):
                        current_scn = ground_result.get("current_scene")

                    ground_status = str(ground_result.get("status") or "").upper()
                    if ground_status == "SENSITIVE_PROTECTED":
                        raise SecurityPolicyViolationError(
                            f"Visual action on '{target_query}' is SENSITIVE_PROTECTED and strictly prohibited."
                        )
                    if action_target is None:
                        fail_reason = ground_result.get("reason") or f"Could not locate '{target_query}' on the screen."
                        return SystemSkillResult(
                            operation=op,
                            success=False,
                            data={
                                "status": VisualActionResultStatus.GROUNDING_FAILED.value,
                                "action_type": op,
                                "target_element_name": target_query,
                                "success": False,
                                "reason": fail_reason,
                            },
                            error=fail_reason,
                            duration_ms=0.0,
                            metadata={"target": target_query},
                        )

        # Check semantic application continuity constraints
        expected_app = params.get("expected_app")
        expected_proc = params.get("expected_process")
        wf_dict = params.get("workflow_context")
        if isinstance(wf_dict, dict):
            expected_app = expected_app or wf_dict.get("expected_app")
            expected_proc = expected_proc or wf_dict.get("expected_process")

        if (expected_app or expected_proc) and current_win:
            win_proc = str(current_win.get("process_name") or "").lower()
            win_title = str(current_win.get("window_title") or "").lower()
            mismatch = False
            if expected_proc:
                ep = expected_proc.lower()
                ep_base = ep[:-4] if ep.endswith(".exe") else ep
                if ep not in win_proc and ep_base not in win_proc and ep_base not in win_title:
                    mismatch = True
            elif expected_app:
                ea = expected_app.lower()
                if ea not in win_proc and ea not in win_title:
                    mismatch = True

            if mismatch:
                # Do NOT automatically refocus. Fail closed with PREFLIGHT_WINDOW_MISMATCH
                fail_reason = f"Active window mismatch: expected '{expected_proc or expected_app}', but found '{win_proc or win_title}'."
                self._logger.warning("Application continuity violation: %s", fail_reason)
                return SystemSkillResult(
                    operation=op,
                    success=False,
                    data={
                        "status": "PREFLIGHT_WINDOW_MISMATCH",
                        "action_type": op,
                        "target_element_name": target_arg or "",
                        "success": False,
                        "reason": fail_reason,
                        "metadata": {"preflight_status": "WINDOW_MISMATCH"},
                    },
                    error=fail_reason,
                    duration_ms=0.0,
                    metadata={"target": target_arg or ""},
                )

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
        current_win = current_win or params.get("current_window_info")
        current_sit = current_sit or params.get("current_situation")
        current_scn = current_scn or params.get("current_scene")

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
