"""Data models and exception hierarchy for the J.A.R.V.I.S AI Subsystem."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Optional

from app.core.constants import DEFAULT_GEMINI_MODEL
from app.core.container import JarvisException

DEFAULT_MAX_TOKENS: Final[int] = 2048
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 0.95
DEFAULT_TOP_K: Final[int] = 40


# --------------------------------------------------------------------------
# Exception Hierarchy
# --------------------------------------------------------------------------

class AIError(JarvisException):
    """Base exception for all AI subsystem errors."""


class AIConfigError(AIError):
    """Raised when AI configuration or API credentials are missing or invalid."""


class AIProviderError(AIError):
    """Raised when an error occurs during model provider communication or response generation."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class AIRateLimitError(AIProviderError):
    """Raised when an AI provider rate limit or quota has been reached (e.g. HTTP 429)."""


class AIAuthenticationError(AIProviderError):
    """Raised when API credentials are rejected or unauthorized (e.g. HTTP 401/403)."""


class AITimeoutError(AIProviderError):
    """Raised when a request to the AI provider exceeds the timeout duration."""


class PromptError(AIError):
    """Raised when prompt construction or template rendering fails."""


# --------------------------------------------------------------------------
# Enums and Typed Models
# --------------------------------------------------------------------------

class Role(str, Enum):
    """Chat message roles recognized across AI and memory systems."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatMessage:
    """Individual conversational message turn."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chat message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters and token limits for content generation."""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_output_tokens: int = DEFAULT_MAX_TOKENS
    stop_sequences: list[str] = field(default_factory=list)

    def to_gemini_dict(self) -> dict[str, Any]:
        """Format configuration matching the Google Gemini API schema."""
        cfg: dict[str, Any] = {
            "temperature": self.temperature,
            "topP": self.top_p,
            "topK": self.top_k,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.stop_sequences:
            cfg["stopSequences"] = list(self.stop_sequences)
        return cfg


@dataclass(frozen=True)
class AIResponse:
    """Standardized response payload from AI generation."""

    content: str
    model: str = DEFAULT_GEMINI_MODEL
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration: float = 0.0
    finish_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize response object to a dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration": self.duration,
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }
