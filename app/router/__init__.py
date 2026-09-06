"""Command Router Subsystem for J.A.R.V.I.S.

Provides command normalization, alias expansion, preprocessor chains,
middleware pipelines, intent classification, and prioritized routing to the SkillManager.
"""

from __future__ import annotations

from app.router.intent import Intent
from app.router.router import (
    CommandPreprocessError,
    CommandRouter,
    InvalidCommandError,
    MiddlewareError,
    RouterError,
    RoutingError,
    command_router,
)

__all__ = [
    "Intent",
    "CommandRouter",
    "command_router",
    "RouterError",
    "RoutingError",
    "CommandPreprocessError",
    "MiddlewareError",
    "InvalidCommandError",
]
