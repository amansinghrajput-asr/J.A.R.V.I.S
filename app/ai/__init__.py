"""AI Subsystem for J.A.R.V.I.S."""

from __future__ import annotations

from app.ai.manager import AIManager, ai_manager
from app.ai.models import (
    AIAuthenticationError,
    AIConfigError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AIResponse,
    AITimeoutError,
    ChatMessage,
    GenerationConfig,
    PromptError,
    Role,
)
from app.ai.prompt import DEFAULT_SYSTEM_PROMPT, PromptBuilder
from app.ai.provider import GeminiProvider
from app.ai.provider_router import ProviderRouter, provider_router

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
