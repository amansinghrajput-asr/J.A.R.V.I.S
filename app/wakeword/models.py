"""Wake Word Subsystem Data Models and Exceptions for J.A.R.V.I.S."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.constants import DEFAULT_WAKE_PHRASES
from app.core.container import JarvisException


class WakeWordError(JarvisException):
    """Base exception for all wake word subsystem errors."""


class InvalidInputError(WakeWordError):
    """Raised when an invalid input is passed to the wake word detector."""


@dataclass(frozen=True)
class WakeWordResult:
    """Encapsulates the result of a wake word detection evaluation.

    Attributes:
        detected: True if a configured wake phrase was detected in the input, False otherwise.
        matched_phrase: The exact configured wake phrase that matched (e.g., 'Hey Jarvis'),
            or None if no wake word was detected.
        normalized_text: The normalized form of the input text evaluated by the detector.
        timestamp: Unix epoch timestamp in seconds marking when detection occurred.
    """

    detected: bool
    matched_phrase: Optional[str] = None
    normalized_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def __bool__(self) -> bool:
        """Allow direct boolean evaluation of the detection outcome."""
        return self.detected

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary."""
        return {
            "detected": self.detected,
            "matched_phrase": self.matched_phrase,
            "normalized_text": self.normalized_text,
            "timestamp": self.timestamp,
        }


__all__ = [
    "DEFAULT_WAKE_PHRASES",
    "InvalidInputError",
    "WakeWordError",
    "WakeWordResult",
]
