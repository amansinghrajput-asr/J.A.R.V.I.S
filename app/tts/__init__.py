"""Text-to-Speech (TTS) Subsystem for J.A.R.V.I.S.

Provides Microsoft Edge TTS powered speech synthesis with configurable voice,
rate, volume, pitch, temporary audio file management, lazy initialization,
thread-safety, async-readiness, and event bus lifecycle notifications.
"""

from __future__ import annotations

from app.tts.models import (
    AudioFileError,
    ConfigurationError,
    EmptyTextError,
    SpeechResult,
    SynthesisError,
    TTSError,
    VoiceNotFoundError,
)
from app.tts.speaker import (
    EVENT_SOURCE_TTS,
    EVENT_TTS_COMPLETED,
    EVENT_TTS_FAILED,
    EVENT_TTS_STARTED,
    TextToSpeech,
    text_to_speech,
)

__all__ = [
    "AudioFileError",
    "ConfigurationError",
    "EmptyTextError",
    "EVENT_SOURCE_TTS",
    "EVENT_TTS_COMPLETED",
    "EVENT_TTS_FAILED",
    "EVENT_TTS_STARTED",
    "SpeechResult",
    "SynthesisError",
    "TTSError",
    "TextToSpeech",
    "VoiceNotFoundError",
    "text_to_speech",
]
