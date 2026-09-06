"""Voice Pipeline Subsystem Data Models, State Definitions, and Audio Abstractions.

Defines the VoiceState enumeration, VoicePipelineResult dataclass, exception
hierarchy, and the placeholder audio provider interface for J.A.R.V.I.S.
"""

from __future__ import annotations

import collections
import inspect
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from app.core.container import JarvisException
from app.stt.models import TranscriptionResult
from app.tts.models import SpeechResult
from app.wakeword.models import WakeWordResult


class VoiceState(str, Enum):
    """Lifecycle states of the J.A.R.V.I.S Voice Pipeline state machine."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

    def __str__(self) -> str:
        return self.value


# ------------------------------------------------------------------------------
# Exception Hierarchy
# ------------------------------------------------------------------------------


class VoicePipelineError(JarvisException):
    """Base exception for all voice pipeline errors."""


class AudioProviderError(VoicePipelineError):
    """Raised when an error occurs in the audio provider layer."""


class MicrophoneError(AudioProviderError):
    """Raised when an error occurs in the microphone capture layer."""


class RecordingError(MicrophoneError):
    """Raised when an error occurs during audio recording execution."""


class InvalidStateError(VoicePipelineError):
    """Raised when an invalid state transition is requested."""


class PipelineNotRunningError(VoicePipelineError):
    """Raised when an operation requires a running pipeline but it is stopped."""


class PipelineExecutionError(VoicePipelineError):
    """Raised when an error occurs during voice pipeline orchestration."""


# ------------------------------------------------------------------------------
# Audio Provider Interfaces
# ------------------------------------------------------------------------------


class AudioProvider(ABC):
    """Abstract interface for audio capture and microphone providers.

    Decouples physical microphone hardware from high-level pipeline orchestration.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize and activate the audio source."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and release the audio source."""

    @abstractmethod
    def is_active(self) -> bool:
        """Check if the audio source is currently active."""

    @abstractmethod
    def capture_audio(self) -> Optional[Path]:
        """Synchronously capture an audio sample and return its file path.

        Returns:
            Path to recorded audio (.wav) file, or None if no audio was captured.
        """

    async def capture_audio_async(self) -> Optional[Path]:
        """Asynchronously capture an audio sample.

        Default implementation wraps synchronous `capture_audio()`.
        """
        return self.capture_audio()


class PlaceholderAudioProvider(AudioProvider):
    """Thread-safe placeholder audio provider for testing and decoupled execution.

    Maintains an in-memory queue of pre-recorded audio file paths or simulated
    samples without requiring microphone hardware.
    """

    def __init__(self, audio_files: Optional[list[Union[str, Path]]] = None) -> None:
        """Initialize the placeholder audio provider.

        Args:
            audio_files: Optional initial list of audio file paths to queue.
        """
        self._lock = threading.RLock()
        self._active: bool = False
        self._queue: collections.deque[Path] = collections.deque()

        if audio_files:
            for audio_path in audio_files:
                self.enqueue_audio(audio_path)

    def start(self) -> None:
        """Activate the placeholder audio provider."""
        with self._lock:
            self._active = True

    def stop(self) -> None:
        """Deactivate the placeholder audio provider."""
        with self._lock:
            self._active = False

    def is_active(self) -> bool:
        """Return whether the provider is currently active."""
        with self._lock:
            return self._active

    def enqueue_audio(self, audio_path: Union[str, Path]) -> None:
        """Enqueue an audio file path to be returned by next capture calls.

        Args:
            audio_path: Path string or Path object pointing to an audio file.
        """
        with self._lock:
            self._queue.append(Path(audio_path).resolve())

    def clear(self) -> None:
        """Clear all pending audio paths from the queue."""
        with self._lock:
            self._queue.clear()

    def pending_count(self) -> int:
        """Return the number of audio samples currently queued."""
        with self._lock:
            return len(self._queue)

    def capture_audio(self) -> Optional[Path]:
        """Pop the next available queued audio path.

        Returns:
            Path to next queued audio file if active and available, else None.
        """
        with self._lock:
            if not self._active:
                return None
            if self._queue:
                return self._queue.popleft()
            return None


# ------------------------------------------------------------------------------
# Pipeline Execution Result
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VoicePipelineResult:
    """Encapsulates the complete result of a voice pipeline execution cycle.

    Attributes:
        session_id: Unique correlation identifier for this pipeline turn.
        state: Final voice state at the end of the cycle.
        audio_input: Optional path to the audio file processed.
        transcription: Optional result from Speech-to-Text transcription.
        wake_word_result: Optional result from wake word detection.
        command_text: Extracted command text forwarded to router, if any.
        command_result: Execution result returned by the Command Router / Skill.
        speech_result: Speech synthesis result returned by Text-to-Speech.
        success: Boolean flag indicating overall cycle success.
        error: Diagnostic error message if the cycle failed.
        duration: Total execution duration in seconds.
        timestamp: Epoch timestamp marking when the cycle completed.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: VoiceState = VoiceState.IDLE
    audio_input: Optional[Path] = None
    transcription: Optional[TranscriptionResult] = None
    wake_word_result: Optional[WakeWordResult] = None
    command_text: Optional[str] = None
    command_result: Optional[Any] = None
    speech_result: Optional[SpeechResult] = None
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Normalize audio_input to a resolved Path if provided."""
        if self.audio_input is not None:
            object.__setattr__(self, "audio_input", Path(self.audio_input).resolve())

    def __bool__(self) -> bool:
        """Boolean evaluation based on cycle success."""
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary representation."""
        return {
            "session_id": self.session_id,
            "state": self.state.value if isinstance(self.state, VoiceState) else str(self.state),
            "audio_input": str(self.audio_input) if self.audio_input else None,
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "wake_word_result": self.wake_word_result.to_dict() if self.wake_word_result else None,
            "command_text": self.command_text,
            "command_result": self.command_result,
            "speech_result": self.speech_result.to_dict() if self.speech_result else None,
            "success": self.success,
            "error": self.error,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


# ------------------------------------------------------------------------------
# Voice Conversation Engine Result
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceConversationResult:
    """Encapsulates the complete result of a single voice interaction cycle.

    Flow:
        record audio -> transcribe -> route -> receive response -> speak response -> return result

    Attributes:
        session_id: Unique correlation identifier for this interaction turn.
        audio_path: Path to the recorded/processed audio WAV file.
        transcription: Transcription outcome (TranscriptionResult or string).
        text: Transcribed user speech text.
        command: Command string routed to the CommandRouter.
        response: Execution result object from the CommandRouter / Skill.
        response_text: User-facing speakable text extracted from response.
        speech_result: Synthesized speech result from TextToSpeech.
        success: Whether the interaction completed successfully without fatal error.
        error: Diagnostic error message if any phase failed.
        duration: Total interaction elapsed duration in seconds.
        timestamp: Epoch timestamp marking interaction completion.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_path: Optional[Path] = None
    transcription: Optional[Union[TranscriptionResult, str]] = None
    text: str = ""
    command: str = ""
    response: Optional[Any] = None
    response_text: str = ""
    speech_result: Optional[SpeechResult] = None
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Normalize audio_path to a resolved Path if provided."""
        if self.audio_path is not None:
            object.__setattr__(self, "audio_path", Path(self.audio_path).resolve())

    @property
    def audio_input(self) -> Optional[Path]:
        """Backward-compatible alias for audio_path."""
        return self.audio_path

    @property
    def audio_output(self) -> Optional[Path]:
        """Path to synthesized audio response if available."""
        if self.speech_result and hasattr(self.speech_result, "audio_path"):
            return self.speech_result.audio_path
        return None

    @property
    def command_text(self) -> str:
        """Backward-compatible alias for command."""
        return self.command

    @property
    def command_result(self) -> Optional[Any]:
        """Backward-compatible alias for response."""
        return self.response

    def __bool__(self) -> bool:
        """Boolean evaluation based on interaction success."""
        return self.success

    def __str__(self) -> str:
        """String representation showing the assistant response or user text."""
        return self.response_text or self.text or (self.error or "")

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary representation."""
        transcription_dict: Any = None
        if hasattr(self.transcription, "to_dict"):
            transcription_dict = self.transcription.to_dict()
        elif self.transcription is not None:
            transcription_dict = str(self.transcription)

        return {
            "session_id": self.session_id,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "audio_input": str(self.audio_path) if self.audio_path else None,
            "transcription": transcription_dict,
            "text": self.text,
            "command": self.command,
            "command_text": self.command,
            "response": self.response,
            "command_result": self.response,
            "response_text": self.response_text,
            "speech_result": (
                self.speech_result.to_dict()
                if hasattr(self.speech_result, "to_dict")
                else (str(self.speech_result) if self.speech_result else None)
            ),
            "audio_output": str(self.audio_output) if self.audio_output else None,
            "success": self.success,
            "error": self.error,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


__all__ = [
    "AudioProvider",
    "AudioProviderError",
    "InvalidStateError",
    "MicrophoneError",
    "PlaceholderAudioProvider",
    "PipelineExecutionError",
    "PipelineNotRunningError",
    "RecordingError",
    "VoiceConversationResult",
    "VoicePipelineError",
    "VoicePipelineResult",
    "VoiceState",
]
