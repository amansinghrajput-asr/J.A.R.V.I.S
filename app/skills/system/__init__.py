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


def register_system_foundation(
    container: ServiceContainer,
    *,
    security_policy: Optional[SystemSecurityPolicy] = None,
    confirmation_manager: Optional[SystemConfirmationManager] = None,
    allow_override: bool = True,
) -> None:
    """Register SystemSecurityPolicy and SystemConfirmationManager in ServiceContainer.

    Args:
        container: Target ServiceContainer instance.
        security_policy: Optional custom SystemSecurityPolicy instance.
        confirmation_manager: Optional custom SystemConfirmationManager instance.
        allow_override: Whether to permit re-registration if already present.
    """
    sec = security_policy or SystemSecurityPolicy()
    conf = confirmation_manager or SystemConfirmationManager()

    container.register_singleton("system_security_policy", sec, allow_override=allow_override)
    container.register_singleton("system_confirmation_manager", conf, allow_override=allow_override)


__all__ = [
    "AppResolver",
    "AppSkills",
    "BLOCKED_SHELL_COMMANDS",
    "BLOCKED_SHELL_PATTERNS",
    "BaseSystemSkill",
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
    "is_critical_process",
    "is_shell_command",
    "register_system_foundation",
    "validate_path",
]
