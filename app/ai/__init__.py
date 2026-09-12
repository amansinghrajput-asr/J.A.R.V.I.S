"""AI Subsystem for J.A.R.V.I.S."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AIManager",
    "ai_manager",
    "ProviderRouter",
    "provider_router",
    "GeminiProvider",
    "PromptBuilder",
    "DEFAULT_SYSTEM_PROMPT",
    "AIResponse",
    "ChatMessage",
    "GenerationConfig",
    "Role",
    "AIError",
    "AIConfigError",
    "AIProviderError",
    "AIRateLimitError",
    "AIAuthenticationError",
    "AITimeoutError",
    "PromptError",
]

_LAZY_EXPORTS = {
    "AIManager": ("app.ai.manager", "AIManager"),
    "ai_manager": ("app.ai.manager", "ai_manager"),
    "ProviderRouter": ("app.ai.provider_router", "ProviderRouter"),
    "provider_router": ("app.ai.provider_router", "provider_router"),
    "GeminiProvider": ("app.ai.provider", "GeminiProvider"),
    "PromptBuilder": ("app.ai.prompt", "PromptBuilder"),
    "DEFAULT_SYSTEM_PROMPT": ("app.ai.prompt", "DEFAULT_SYSTEM_PROMPT"),
    "AIResponse": ("app.ai.models", "AIResponse"),
    "ChatMessage": ("app.ai.models", "ChatMessage"),
    "GenerationConfig": ("app.ai.models", "GenerationConfig"),
    "Role": ("app.ai.models", "Role"),
    "AIError": ("app.ai.models", "AIError"),
    "AIConfigError": ("app.ai.models", "AIConfigError"),
    "AIProviderError": ("app.ai.models", "AIProviderError"),
    "AIRateLimitError": ("app.ai.models", "AIRateLimitError"),
    "AIAuthenticationError": ("app.ai.models", "AIAuthenticationError"),
    "AITimeoutError": ("app.ai.models", "AITimeoutError"),
    "PromptError": ("app.ai.models", "PromptError"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
