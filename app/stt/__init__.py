"""Speech-to-Text (STT) Subsystem for J.A.R.V.I.S.

Provides faster-whisper powered offline audio transcription with lazy model loading,
model caching, thread safety, async-readiness, and event lifecycle notifications.
"""

from __future__ import annotations

from app.stt.models import (
    AudioFileNotFoundError,
    InvalidAudioFormatError,
    ModelLoadError,
    STTError,
    TranscriptionError,
    TranscriptionResult,
)
from app.stt.transcriber import (
    EVENT_SOURCE_STT,
    EVENT_STT_COMPLETED,
    EVENT_STT_FAILED,
    EVENT_STT_STARTED,
    SpeechToText,
    speech_to_text,
)

__all__ = [
    "AudioFileNotFoundError",
    "EVENT_SOURCE_STT",
    "EVENT_STT_COMPLETED",
    "EVENT_STT_FAILED",
    "EVENT_STT_STARTED",
    "InvalidAudioFormatError",
    "ModelLoadError",
    "STTError",
    "SpeechToText",
    "TranscriptionError",
    "TranscriptionResult",
    "speech_to_text",
]
