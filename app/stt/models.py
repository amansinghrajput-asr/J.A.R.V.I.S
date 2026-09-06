"""Speech-to-Text (STT) Data Models and Exception Hierarchy for J.A.R.V.I.S."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.container import JarvisException


class STTError(JarvisException):
    """Base exception for all Speech-to-Text subsystem errors."""


class AudioFileNotFoundError(STTError):
    """Raised when an audio file cannot be found at the specified path."""


class InvalidAudioFormatError(STTError):
    """Raised when an unsupported audio file format is provided (e.g. non-.wav)."""


class ModelLoadError(STTError):
    """Raised when the faster-whisper STT model fails to load."""


class TranscriptionError(STTError):
    """Raised when an error occurs during audio transcription."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Encapsulates the result of a Speech-to-Text audio transcription.

    Attributes:
        text: Transcribed text output (stripped of leading/trailing whitespace).
        language: Detected or specified language code (e.g., 'en', 'hi').
        language_probability: Confidence score for the language detection (0.0 to 1.0).
        duration: Audio duration in seconds.
        segments: Tuple of segment dictionaries containing timestamps and text chunks.
        model_name: Identifier of the Whisper model used (e.g., 'base', 'tiny').
        execution_time: Wall-clock time in seconds taken to process transcription.
        timestamp: Unix epoch timestamp in seconds marking when transcription completed.
    """

    text: str
    language: str
    language_probability: float = 1.0
    duration: float = 0.0
    segments: tuple[dict[str, Any], ...] = ()
    model_name: str = "base"
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __bool__(self) -> bool:
        """Allow direct boolean evaluation indicating if transcription yielded non-empty text."""
        return bool(self.text and self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary representation."""
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": list(self.segments),
            "model_name": self.model_name,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


__all__ = [
    "AudioFileNotFoundError",
    "InvalidAudioFormatError",
    "ModelLoadError",
    "STTError",
    "TranscriptionError",
    "TranscriptionResult",
]
