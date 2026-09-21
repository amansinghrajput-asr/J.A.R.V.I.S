"""Phase 22: System Skills Foundation Package for J.A.R.V.I.S.

Provides modular, secure, and observable PC and Windows automation skills,
integrated with the existing Planner, Executor, and Metacognition subsystems.
"""

from __future__ import annotations

from typing import Optional

from app.core.container import ServiceContainer
from app.skills.system.base_system_skill import (
    BaseSystemSkill,
    DEFAULT_SYSTEM_SKILL_PRIORITY,
    SystemSkillResult,
)
from app.skills.system.security import (
    BLOCKED_SHELL_COMMANDS,
    BLOCKED_SHELL_PATTERNS,
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    PROTECTED_WINDOWS_DIRS,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityError,
    SystemSecurityPolicy,
    WINDOWS_RESERVED_DEVICE_NAMES,
    is_critical_process,
    is_shell_command,
    validate_path,
)
from app.skills.system.app_skills import (
    AppResolver,
    AppSkills,
    DEFAULT_SAFE_APP_ALIASES,
    ProcessManager,
)
from app.skills.system.file_skills import (
    DEFAULT_MAX_DIR_ENTRIES,
    DEFAULT_MAX_READ_BYTES,
    DEFAULT_MAX_RECURSIVE_DEPTH,
    DEFAULT_MAX_RECURSIVE_ITEMS,
    FileSkills,
    MAX_ALLOWED_DIR_ENTRIES,
)
from app.skills.system.system_info_skills import SystemInfoSkills
from app.skills.system.system_control_skills import SystemControlSkills
from app.skills.system.window_skills import WindowSkills
from app.skills.system.browser_skills import BrowserSkills
from app.skills.system.vision_skills import VisionSkills
from app.skills.system.interaction_skills import InteractionSkills


def register_system_foundation(
    container: ServiceContainer,
    *,
    security_policy: Optional[SystemSecurityPolicy] = None,
    confirmation_manager: Optional[SystemConfirmationManager] = None,
    planner_event_bus: Optional[PlannerEventBus] = None,
    allow_override: bool = True,
) -> None:
    """Register PlannerEventBus, SystemSecurityPolicy, and SystemConfirmationManager in ServiceContainer.

    Ensures a single shared PlannerEventBus singleton connects security, confirmation,
    executor, and presentation layers without creating competing event systems.

    Args:
        container: Target ServiceContainer instance.
        security_policy: Optional custom SystemSecurityPolicy instance.
        confirmation_manager: Optional custom SystemConfirmationManager instance.
        planner_event_bus: Optional custom PlannerEventBus instance.
        allow_override: Whether to permit re-registration if already present.
    """
    from app.ai.planner.events import PlannerEventBus

    # 1. Resolve or register shared PlannerEventBus singleton
    if planner_event_bus is not None:
        peb = planner_event_bus
        container.register_singleton("planner_event_bus", peb, allow_override=allow_override)
    elif container.exists("planner_event_bus"):
        peb = container.resolve("planner_event_bus")
    else:
        peb = PlannerEventBus()
        container.register_singleton("planner_event_bus", peb, allow_override=allow_override)

    # 2. Resolve or register SystemSecurityPolicy bound to shared PlannerEventBus
    if security_policy is not None:
        sec = security_policy
        container.register_singleton("system_security_policy", sec, allow_override=allow_override)
    elif container.exists("system_security_policy"):
        sec = container.resolve("system_security_policy")
        if getattr(sec, "_event_bus", None) is None:
            sec._event_bus = peb
    else:
        sec = SystemSecurityPolicy(event_bus=peb)
        container.register_singleton("system_security_policy", sec, allow_override=allow_override)

    # 3. Resolve or register SystemConfirmationManager bound to shared PlannerEventBus
    if confirmation_manager is not None:
        conf = confirmation_manager
        container.register_singleton("system_confirmation_manager", conf, allow_override=allow_override)
    elif container.exists("system_confirmation_manager"):
        conf = container.resolve("system_confirmation_manager")
        if getattr(conf, "_event_bus", None) is None:
            conf._event_bus = peb
    else:
        conf = SystemConfirmationManager(event_bus=peb)
        container.register_singleton("system_confirmation_manager", conf, allow_override=allow_override)


__all__ = [
    "AppResolver",
    "AppSkills",
    "BLOCKED_SHELL_COMMANDS",
    "BLOCKED_SHELL_PATTERNS",
    "BaseSystemSkill",
    "BrowserSkills",
    "ConfirmationRejectedError",
    "ConfirmationRequiredError",
    "ConfirmationTimeoutError",
    "DEFAULT_MAX_DIR_ENTRIES",
    "DEFAULT_MAX_READ_BYTES",
    "DEFAULT_MAX_RECURSIVE_DEPTH",
    "DEFAULT_MAX_RECURSIVE_ITEMS",
    "DEFAULT_SAFE_APP_ALIASES",
    "DEFAULT_SYSTEM_SKILL_PRIORITY",
    "FileSkills",
    "InteractionSkills",
    "MAX_ALLOWED_DIR_ENTRIES",
    "PROTECTED_WINDOWS_DIRS",
    "ProcessManager",
    "SecurityPolicyViolationError",
    "SystemConfirmationManager",
    "SystemControlSkills",
    "SystemInfoSkills",
    "SystemSafetyTier",
    "SystemSecurityError",
    "SystemSecurityPolicy",
    "SystemSkillResult",
    "WINDOWS_RESERVED_DEVICE_NAMES",
    "WindowSkills",
    "VisionSkills",
    "is_critical_process",
    "is_shell_command",
    "register_system_foundation",
    "validate_path",
]
