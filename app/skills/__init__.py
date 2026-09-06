"""Skill Framework for J.A.R.V.I.S.

Provides the extensible BaseSkill specification, exception hierarchy,
and the centralized, thread-safe SkillManager for skill lifecycle management,
dynamic command routing, event dispatch, and dependency injection.
"""

from __future__ import annotations

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

__all__ = [
    "DEFAULT_SKILL_PRIORITY",
    "DEFAULT_SKILL_VERSION",
    "BaseSkill",
    "InvalidSkillError",
    "SkillAlreadyRegisteredError",
    "SkillDiscoveryError",
    "SkillError",
    "SkillExecutionError",
    "SkillInitializationError",
    "SkillManager",
    "SkillNotFoundError",
    "skill_manager",
]
