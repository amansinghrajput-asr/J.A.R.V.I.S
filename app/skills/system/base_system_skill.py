"""Base System Skill foundation for J.A.R.V.I.S. Phase 22.

Provides the foundational contract, security gate evaluation, confirmation resolution,
timing telemetry, structured result formatting, and event emission for all Phase 22 PC skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import logging
import time
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union

from app.ai.planner.events import (
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillFailed,
    SystemSkillPolicyRejected,
    SystemSkillStarted,
)
from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import BaseSkill, SkillError, SkillExecutionError
from app.skills.system.security import (
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityError,
    SystemSecurityPolicy,
)

DEFAULT_SYSTEM_SKILL_PRIORITY: Final[int] = 60


@dataclass
class SystemSkillResult:
    """Standardized output structure for all system skill operations."""

    operation: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "operation": self.operation,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }

    @property
    def message(self) -> str:
        """Produce a concise, human-readable, safe text summary suitable for speech/TTS."""
        if not self.success:
            if self.error:
                err_str = str(self.error).strip()
                lower_err = err_str.lower()
                if "captureblocked" in lower_err or "blocked" in lower_err or "security policy" in lower_err:
                    return "Screen capture was blocked by vision privacy policy."
                return err_str
            return f"Operation '{self.operation}' failed."

        # Handle operation-specific formatting when data is a dict
        if isinstance(self.data, dict):
            op = (self.operation or "").strip().lower()

            if op == "capture_screen":
                return "Screen captured successfully."

            if op == "read_screen_text":
                text = self.data.get("text")
                if text is not None and isinstance(text, str) and text.strip():
                    return text.strip()
                return "No text was detected on your screen."

            if op == "explain_active_window":
                summary = self.data.get("summary")
                if summary is not None and isinstance(summary, str) and summary.strip():
                    return summary.strip()
                return "Active window analyzed successfully."

            if op == "diagnose_screen_error":
                if self.data.get("error_found"):
                    title = self.data.get("error_title") or "Error detected"
                    cause = self.data.get("root_cause")
                    fix = self.data.get("recommended_fix")
                    parts = [str(title).strip()]
                    if cause and str(cause).strip() and str(cause).strip().lower() not in ("n/a", "none"):
                        parts.append(f"Root cause: {str(cause).strip()}")
                    if fix and str(fix).strip() and str(fix).strip().lower() not in ("none", "n/a"):
                        parts.append(f"Recommended fix: {str(fix).strip()}")
                    msg = ". ".join(parts)
                    return msg if msg.endswith(".") else msg + "."
                return "No errors were detected on your screen."

            if op == "ask_screen":
                ans = self.data.get("answer") or self.data.get("response") or self.data.get("summary")
                if ans is not None and isinstance(ans, str) and ans.strip():
                    return ans.strip()
                return "I couldn't determine that from the screen."

            if op == "verify_screen_state":
                verified = self.data.get("verified")
                status = str(self.data.get("status") or "").strip().lower()
                reason = self.data.get("reason")
                if verified is True or status == "verified":
                    if reason and isinstance(reason, str) and reason.strip():
                        r = reason.strip()
                        return r if r.lower().startswith("verified") else f"The condition was verified. {r}"
                    return "The condition was verified."
                elif verified is False or status == "not_verified":
                    if reason and isinstance(reason, str) and reason.strip():
                        r = reason.strip()
                        return r if r.lower().startswith("not verified") else f"The condition was not verified. {r}"
                    return "The condition was not verified."
                else:
                    if reason and isinstance(reason, str) and reason.strip() and reason.strip().lower() != "uncertain":
                        return reason.strip()
                    return "I couldn't determine that from the screen."

            if op == "locate_element":
                summary = self.data.get("summary")
                if summary and isinstance(summary, str) and summary.strip():
                    return summary.strip()
                if self.data.get("is_found") and self.data.get("element"):
                    el = self.data["element"]
                    name = el.get("name", "element") if isinstance(el, dict) else getattr(el, "name", "element")
                    return f"Found '{name}' on the screen."
                return "I couldn't locate that element on the screen."

            if op == "detect_screen_change":
                explanation = self.data.get("explanation") or self.data.get("summary")
                if explanation and isinstance(explanation, str) and explanation.strip():
                    return explanation.strip()
                if self.data.get("meaningful_change_detected"):
                    return "Visual changes were detected on the screen."
                return "No meaningful changes were detected on your screen."

            if op == "map_ui_scene":
                summary = self.data.get("summary")
                if summary and isinstance(summary, str) and summary.strip():
                    return summary.strip()
                count = len(self.data.get("interactive_elements", []))
                if count > 0:
                    return f"Mapped {count} interactive control{'s' if count > 1 else ''} on the screen."
                return "Screen layout analyzed successfully."

            # General dict fallback for other system skills (AppSkills, WindowSkills, FileSkills, etc.)
            for key in ("message", "summary", "text", "response", "content", "output"):
                val = self.data.get(key)
                if val is not None and isinstance(val, str) and val.strip():
                    return val.strip()

        elif isinstance(self.data, str) and self.data.strip():
            return self.data.strip()

        return f"Operation '{self.operation}' completed successfully."



class BaseSystemSkill(BaseSkill):
    """Abstract foundational skill for all Phase 22 Windows and PC automation skills.

    Integrates:
    - BaseSkill contract and lifecycle hooks
    - Automatic SystemSecurityPolicy validation (SAFE / CONFIRMATION_REQUIRED / RESTRICTED)
    - SystemConfirmationManager authorization workflows
    - Execution metrics and telemetry
    - Standardized event publishing on PlannerEventBus
    """

    priority: int = DEFAULT_SYSTEM_SKILL_PRIORITY
    version: str = "1.0.0"
    tags: list[str] = ["system", "pc", "automation"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        tags: Optional[list[str]] = None,
        permissions: Optional[Union[set[str], list[str]]] = None,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, PlannerEventBus]] = None,
    ) -> None:
        """Initialize the BaseSystemSkill.

        Args:
            name: Override skill name.
            description: Override skill description.
            version: Semantic version.
            enabled: Skill active status.
            priority: Execution priority.
            tags: Discovery tags.
            permissions: Required permissions.
            security_policy: Injected SystemSecurityPolicy.
            confirmation_manager: Injected SystemConfirmationManager.
            config: Injected Settings.
            logger: Injected Logger.
            container: Injected ServiceContainer.
            event_bus: Injected EventBus or PlannerEventBus.
        """
        super().__init__(
            name=name,
            description=description,
            version=version,
            enabled=enabled,
            priority=priority,
            tags=tags,
            permissions=permissions,
            config=config,
            logger=logger,
            container=container,
            event_bus=event_bus if isinstance(event_bus, EventBus) else None,
        )

        self._planner_event_bus: Optional[PlannerEventBus] = (
            event_bus if isinstance(event_bus, PlannerEventBus) else None
        )
        self._security_policy = security_policy
        self._confirmation_manager = confirmation_manager

    @property
    def security_policy(self) -> SystemSecurityPolicy:
        """Retrieve the active security policy, resolving from container if needed."""
        if self._security_policy is not None:
            return self._security_policy
        if self._container is not None and self._container.exists("system_security_policy"):
            return self._container.resolve("system_security_policy")
        # Default isolated instance
        self._security_policy = SystemSecurityPolicy(event_bus=self.planner_event_bus)
        return self._security_policy

    @property
    def confirmation_manager(self) -> SystemConfirmationManager:
        """Retrieve the active confirmation manager, resolving from container if needed."""
        if self._confirmation_manager is not None:
            return self._confirmation_manager
        if self._container is not None and self._container.exists("system_confirmation_manager"):
            return self._container.resolve("system_confirmation_manager")
        # Default isolated instance
        self._confirmation_manager = SystemConfirmationManager(event_bus=self.planner_event_bus)
        return self._confirmation_manager

    @property
    def planner_event_bus(self) -> Optional[PlannerEventBus]:
        """Retrieve the PlannerEventBus instance if available."""
        if self._planner_event_bus is not None:
            return self._planner_event_bus
        if self._container is not None and self._container.exists("planner_event_bus"):
            return self._container.resolve("planner_event_bus")
        return None

    def bind_system_services(
        self,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        planner_event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Bind system-specific security and confirmation dependencies."""
        if security_policy is not None:
            self._security_policy = security_policy
        if confirmation_manager is not None:
            self._confirmation_manager = confirmation_manager
        if planner_event_bus is not None:
            self._planner_event_bus = planner_event_bus

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Normalize arbitrary command inputs into structured components.

        Returns:
            Tuple of (operation, target, parameters, confirmation_id).
        """
        if isinstance(command, dict):
            op = str(command.get("operation") or command.get("action") or "").strip().lower()
            target = command.get("target")
            if target is not None:
                target = str(target).strip()
            params = dict(command.get("parameters") or command.get("params") or {})
            conf_id = command.get("confirmation_id")
            if conf_id is not None:
                conf_id = str(conf_id).strip()
            return op, target, params, conf_id

        # Fallback for plain string
        text = str(command or "").strip()
        tokens = text.split()
        op = tokens[0].lower() if tokens else ""
        target = tokens[1] if len(tokens) > 1 else None
        return op, target, {}, None

    def execute(self, command: Any) -> SystemSkillResult:
        """Execute system command with security validation, confirmation, and timing.

        Args:
            command: Command dict or string.

        Returns:
            SystemSkillResult containing outcome and timing metadata.

        Raises:
            SecurityPolicyViolationError: If operation is RESTRICTED.
            ConfirmationRequiredError: If operation requires confirmation and none is verified.
            ConfirmationRejectedError: If confirmation token is invalid or rejected.
            SkillExecutionError: If execution fails.
        """
        op, target, params, conf_id = self.parse_command(command)

        if not op:
            raise SkillExecutionError("No valid operation specified in command payload.")

        # 1. Security Policy Classification & Validation
        tier, reason = self.security_policy.validate_operation(op, target, params)

        if tier == SystemSafetyTier.RESTRICTED:
            bus = self.planner_event_bus
            if bus is not None:
                bus.publish(
                    SystemSkillPolicyRejected(
                        skill_name=self.name,
                        operation=op,
                        target=target,
                        reason=reason or f"Operation '{op}' is RESTRICTED by system security policy.",
                    )
                )
            raise SecurityPolicyViolationError(
                reason or f"Operation '{op}' is RESTRICTED by system security policy."
            )

        # 2. Confirmation Handling
        if tier == SystemSafetyTier.CONFIRMATION_REQUIRED or conf_id:
            if not conf_id:
                # Issue new confirmation request
                issued_id = self.confirmation_manager.request_confirmation(
                    operation=op,
                    target=target,
                    parameters=params,
                    description=reason or f"Confirmation required for '{op}'.",
                )
                # Check if it was auto-resolved (e.g. via test handler or HITL gateway)
                if self.confirmation_manager.is_approved(issued_id):
                    self.confirmation_manager.verify_and_consume(
                        confirmation_id=issued_id,
                        operation=op,
                        target=target,
                        parameters=params,
                    )
                else:
                    raise ConfirmationRequiredError(
                        f"Action '{op}' requires explicit confirmation. Token: '{issued_id}'.",
                        confirmation_id=issued_id,
                        operation=op,
                        target=target,
                    )
            else:
                # Caller provided an existing confirmation_id -> verify and consume it
                self.confirmation_manager.verify_and_consume(
                    confirmation_id=conf_id,
                    operation=op,
                    target=target,
                    parameters=params,
                )

        # 3. Execution with Telemetry & Event Broadcast
        bus = self.planner_event_bus
        if bus is not None:
            bus.publish(
                SystemSkillStarted(
                    skill_name=self.name,
                    operation=op,
                    target=target,
                    parameters=params,
                )
            )

        exec_id = str(params.get("execution_id") or params.get("task_id") or "")
        meta: Dict[str, Any] = {}
        for k in ("task_id", "plan_id", "trajectory_id"):
            if k in params:
                meta[k] = params[k]

        start_time = time.perf_counter()
        try:
            output = self._execute_operation(op, target, params)
            duration_s = time.perf_counter() - start_time
            duration_ms = duration_s * 1000.0

            if bus is not None:
                bus.publish(
                    SystemSkillCompleted(
                        execution_id=exec_id,
                        skill_name=self.name,
                        operation=op,
                        target=target,
                        duration=duration_s,
                        success=True,
                        result=output,
                        metadata=meta,
                    )
                )

            return SystemSkillResult(
                operation=op,
                success=True,
                data=output,
                duration_ms=duration_ms,
                metadata={"skill": self.name, "target": target},
            )
        except Exception as exc:
            duration_s = time.perf_counter() - start_time
            duration_ms = duration_s * 1000.0
            err_msg = str(exc)

            if bus is not None:
                bus.publish(
                    SystemSkillFailed(
                        execution_id=exec_id,
                        skill_name=self.name,
                        operation=op,
                        target=target,
                        error=err_msg,
                        duration=duration_s,
                        metadata=meta,
                    )
                )

            self.logger.error(
                "SystemSkill '%s' failed operation '%s': %s", self.name, op, exc, exc_info=True
            )
            if isinstance(exc, (SystemSecurityError, SkillError)):
                raise
            raise SkillExecutionError(f"System skill execution failed: {err_msg}") from exc

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher to be implemented by derived skill classes.

        Subclasses inspect `operation` and dispatch to concrete handler functions.
        """
        raise NotImplementedError(
            f"Skill '{self.name}' has not implemented operation '{operation}'."
        )


__all__ = [
    "BaseSystemSkill",
    "DEFAULT_SYSTEM_SKILL_PRIORITY",
    "SystemSkillResult",
]
