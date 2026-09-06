"""Wake Word Subsystem for J.A.R.V.I.S.

Provides production wake word detection using openwakeWord, text-based
evaluation fallback, normalization, EventBus lifecycle events, and
service container integration.
"""

from __future__ import annotations

from app.core.constants import DEFAULT_WAKE_WORD
from app.wakeword.detector import (
    EVENT_SOURCE_WAKEWORD,
    EVENT_WAKEWORD_DETECTED,
    EVENT_WAKEWORD_FAILED,
    EVENT_WAKEWORD_IGNORED,
    EVENT_WAKEWORD_STARTED,
    EVENT_WAKEWORD_STOPPED,
    WakeWordDetector,
    normalize_text,
    wake_word_detector,
)
from app.wakeword.engine import (
    CHUNK_SIZE_SAMPLES,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_DETECTION_THRESHOLD,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    WakeWordEngine,
    wake_word_engine,
)
from app.wakeword.models import (
    AudioStreamError,
    DEFAULT_WAKE_PHRASES,
    InvalidInputError,
    ModelLoadError,
    WakeWordError,
    WakeWordResult,
)

__all__ = [
    "AudioStreamError",
    "CHUNK_SIZE_SAMPLES",
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_DETECTION_THRESHOLD",
    "DEFAULT_WAKE_PHRASES",
    "DEFAULT_WAKE_WORD",
    "EVENT_SOURCE_WAKEWORD",
    "EVENT_WAKEWORD_DETECTED",
    "EVENT_WAKEWORD_FAILED",
    "EVENT_WAKEWORD_IGNORED",
    "EVENT_WAKEWORD_STARTED",
    "EVENT_WAKEWORD_STOPPED",
    "InvalidInputError",
    "ModelLoadError",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "WakeWordDetector",
    "WakeWordEngine",
    "WakeWordError",
    "WakeWordResult",
    "normalize_text",
    "wake_word_detector",
    "wake_word_engine",
]
