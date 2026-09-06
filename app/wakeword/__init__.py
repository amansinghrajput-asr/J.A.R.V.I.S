"""Wake Word Subsystem for J.A.R.V.I.S.

Provides text-based wake word detection, normalization, event notifications,
and service container integration.
"""

from __future__ import annotations

from app.wakeword.detector import (
    EVENT_SOURCE_WAKEWORD,
    EVENT_WAKEWORD_DETECTED,
    EVENT_WAKEWORD_IGNORED,
    WakeWordDetector,
    normalize_text,
    wake_word_detector,
)
from app.wakeword.models import (
    DEFAULT_WAKE_PHRASES,
    InvalidInputError,
    WakeWordError,
    WakeWordResult,
)

__all__ = [
    "DEFAULT_WAKE_PHRASES",
    "EVENT_SOURCE_WAKEWORD",
    "EVENT_WAKEWORD_DETECTED",
    "EVENT_WAKEWORD_IGNORED",
    "InvalidInputError",
    "WakeWordDetector",
    "WakeWordError",
    "WakeWordResult",
    "normalize_text",
    "wake_word_detector",
]
