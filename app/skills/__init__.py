"""Skill Framework for J.A.R.V.I.S.

Provides the extensible BaseSkill specification, exception hierarchy,
and the centralized, thread-safe SkillManager for skill lifecycle management,
dynamic command routing, event dispatch, and dependency injection.
"""

from __future__ import annotations

from app.skills.ai_skill import (
    AISkill,
    DEFAULT_AI_SKILL_DESCRIPTION,
    DEFAULT_AI_SKILL_NAME,
    DEFAULT_AI_SKILL_PRIORITY,
    ai_skill,
)
from app.skills.base import (
    DEFAULT_SKILL_PRIORITY,
    DEFAULT_SKILL_VERSION,
    BaseSkill,
    InvalidSkillError,
    SkillAlreadyRegisteredError,
    SkillDiscoveryError,
    SkillError,
    SkillExecutionError,
    SkillInitializationError,
    SkillNotFoundError,
)
from app.skills.manager import (
    SkillManager,
    skill_manager,
)
from app.skills.system_skill import (
    DEFAULT_SYSTEM_SKILL_DESCRIPTION,
    DEFAULT_SYSTEM_SKILL_NAME,
    DEFAULT_SYSTEM_SKILL_PRIORITY,
    SystemSkill,
    system_skill,
)

__all__ = [
    "AISkill",
    "DEFAULT_AI_SKILL_DESCRIPTION",
    "DEFAULT_AI_SKILL_NAME",
    "DEFAULT_AI_SKILL_PRIORITY",
    "DEFAULT_SKILL_PRIORITY",
    "DEFAULT_SKILL_VERSION",
    "DEFAULT_SYSTEM_SKILL_DESCRIPTION",
    "DEFAULT_SYSTEM_SKILL_NAME",
    "DEFAULT_SYSTEM_SKILL_PRIORITY",
    "BaseSkill",
    "InvalidSkillError",
    "SkillAlreadyRegisteredError",
    "SkillDiscoveryError",
    "SkillError",
    "SkillExecutionError",
    "SkillInitializationError",
    "SkillManager",
    "SkillNotFoundError",
    "SystemSkill",
    "ai_skill",
    "skill_manager",
    "system_skill",
]
