"""Text-to-Speech (TTS) Data Models and Exception Hierarchy for J.A.R.V.I.S."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from app.core.container import JarvisException


class TTSError(JarvisException):
    """Base exception for all Text-to-Speech subsystem errors."""


class EmptyTextError(TTSError):
    """Raised when text provided for speech synthesis is empty or whitespace-only."""


class VoiceNotFoundError(TTSError):
    """Raised when the specified TTS voice is invalid or not available."""


class SynthesisError(TTSError):
    """Raised when speech synthesis fails during generation or network communication."""


class AudioFileError(TTSError):
    """Raised when audio file creation, saving, or accessing fails."""


class ConfigurationError(TTSError):
    """Raised when invalid synthesis parameters (voice, rate, pitch, volume) are provided."""


@dataclass(frozen=True)
class SpeechResult:
    """Encapsulates the result of a Text-to-Speech synthesis operation.

    Attributes:
        text: Original text synthesized into speech.
        audio_path: Absolute Path to the generated audio file.
        voice: Voice model identifier used for synthesis (e.g. 'en-GB-RyanNeural').
        rate: Configured speaking rate (e.g. '+0%', '+10%').
        volume: Configured volume level (e.g. '+0%', '-10%').
        pitch: Configured pitch offset (e.g. '+0Hz', '+5Hz').
        duration: Audio duration in seconds (estimated or calculated).
        file_size_bytes: Size of the synthesized audio file in bytes.
        execution_time: Wall-clock time in seconds taken to perform synthesis.
        timestamp: Unix epoch timestamp in seconds marking when synthesis completed.
    """

    text: str
    audio_path: Path
    voice: str
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"
    duration: float = 0.0
    file_size_bytes: int = 0
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Normalize audio_path to a Path object."""
        if not isinstance(self.audio_path, Path):
            object.__setattr__(self, "audio_path", Path(self.audio_path).resolve())

    def __bool__(self) -> bool:
        """Allow direct boolean evaluation indicating if speech was successfully generated."""
        return bool(self.text.strip()) and self.file_size_bytes > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary representation."""
        return {
            "text": self.text,
            "audio_path": str(self.audio_path),
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume,
            "pitch": self.pitch,
            "duration": self.duration,
            "file_size_bytes": self.file_size_bytes,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


__all__ = [
    "AudioFileError",
    "ConfigurationError",
    "EmptyTextError",
    "SpeechResult",
    "SynthesisError",
    "TTSError",
    "VoiceNotFoundError",
]
