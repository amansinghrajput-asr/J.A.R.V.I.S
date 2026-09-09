"""Microphone Audio Capture Subsystem for J.A.R.V.I.S.

Provides the production audio acquisition layer implementing AudioProvider:
coordinates hardware/simulated microphone capture, writes standardized 16kHz
mono 16-bit PCM WAV audio files (Whisper-compatible), and manages thread-safe
lifecycle events and Service Container registration.
"""

from __future__ import annotations

import sounddevice as sd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Final, Optional, Union
import uuid
import wave

import sounddevice as sd
import soundfile as sf

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.voice.models import (
    AudioProvider,
    AudioProviderError,
    MicrophoneError,
    RecordingError,
)

# Lifecycle Event Constants
EVENT_MICROPHONE_STARTED: Final[str] = "microphone.started"
EVENT_MICROPHONE_STOPPED: Final[str] = "microphone.stopped"
EVENT_MICROPHONE_RECORDING: Final[str] = "microphone.recording"
EVENT_MICROPHONE_RECORDED: Final[str] = "microphone.recorded"
EVENT_SOURCE_MICROPHONE: Final[str] = "microphone"

# Audio Standard Constants (Whisper and STT standard)
DEFAULT_SAMPLE_RATE: Final[int] = 16000  # 16 kHz
DEFAULT_CHANNELS: Final[int] = 1         # Mono
DEFAULT_SAMPLE_WIDTH: Final[int] = 2     # 16-bit PCM (2 bytes)
DEFAULT_CHUNK_SIZE: Final[int] = 1024
DEFAULT_RECORD_DURATION: Final[float] = 6.0


class MicrophoneRecorder(AudioProvider):
    """Production audio recorder capturing microphone input to standard WAV files.

    Inherits from AudioProvider to seamlessly integrate with VoicePipeline.
    Features:
    - Standardized 16kHz, mono, 16-bit PCM WAV format.
    - Thread-safe start, stop, and active state management.
    - Hardware-ready with graceful simulated audio frame generation for headless/CI.
    - Context manager interface support.
    - Synchronous and asynchronous capture methods.
    - EventBus lifecycle notifications.
    - Registers as 'microphone' in ServiceContainer.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        record_duration: float = DEFAULT_RECORD_DURATION,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
        audio_source_callback: Optional[Callable[[int], bytes]] = None,
    ) -> None:
        """Initialize the MicrophoneRecorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000).
            channels: Number of audio channels (default: 1 for mono).
            sample_width: Bytes per sample (default: 2 for 16-bit PCM).
            chunk_size: Buffer chunk size in frames.
            record_duration: Default recording duration in seconds.
            output_dir: Directory where temporary recordings are stored.
            config: Optional Settings instance. Defaults to container or global settings.
            logger: Optional Logger instance. Defaults to 'MICROPHONE' logger.
            container_instance: Optional ServiceContainer. Defaults to global container.
            event_bus_instance: Optional EventBus. Defaults to container or global event bus.
            auto_register_in_container: If True, registers 'microphone' in ServiceContainer.
            audio_source_callback: Optional callback(num_frames) -> bytes returning raw PCM data.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if sample_width <= 0:
            raise ValueError(f"sample_width must be positive, got {sample_width}")
        if record_duration <= 0:
            raise ValueError(f"record_duration must be positive, got {record_duration}")

        self._lock = threading.RLock()
        self._active: bool = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MicrophoneRecorder")

        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._chunk_size = chunk_size
        self._record_duration = float(record_duration)
        self._audio_source_callback = audio_source_callback

        # 1. Dependency Resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency Resolution: Logger
        self._logger = logger if logger is not None else get_logger("MICROPHONE")

        # 3. Dependency Resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Dependency Resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = event_bus

        # 5. Output Directory Configuration
        if output_dir is not None:
            self._output_dir = Path(output_dir).resolve()
        elif hasattr(self._config, "data_dir") and self._config.data_dir:
            self._output_dir = Path(self._config.data_dir) / "audio"
        else:
            self._output_dir = Path(tempfile.gettempdir()) / "jarvis_audio"

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._logger.warning(f"Could not create audio output directory '{self._output_dir}': {exc}")

        # 6. Service Container Registration
        if auto_register_in_container:
            try:
                self._container.register_singleton("microphone", self, allow_override=True)
                self._logger.debug("Registered 'microphone' singleton in Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register MicrophoneRecorder in container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Sampling rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def sample_width(self) -> int:
        """Sample width in bytes."""
        return self._sample_width

    @property
    def record_duration(self) -> float:
        """Default recording duration in seconds."""
        with self._lock:
            return self._record_duration

    @record_duration.setter
    def record_duration(self, val: float) -> None:
        """Update the default recording duration in seconds."""
        if val <= 0:
            raise ValueError(f"record_duration must be positive, got {val}")
        with self._lock:
            self._record_duration = float(val)

    @property
    def output_dir(self) -> Path:
        """Directory for audio recordings."""
        return self._output_dir

    # --------------------------------------------------------------------------
    # Lifecycle Control
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Activate the microphone recorder.

        Emits 'microphone.started' event on the EventBus.
        """
        with self._lock:
            if self._active:
                return
            self._active = True

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_STARTED,
                payload={
                    "sample_rate": self._sample_rate,
                    "channels": self._channels,
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.info("MicrophoneRecorder started and listening.")

    def stop(self) -> None:
        """Deactivate the microphone recorder.

        Emits 'microphone.stopped' event on the EventBus.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_STOPPED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.info("MicrophoneRecorder stopped.")

    def is_active(self) -> bool:
        """Check whether the microphone recorder is active."""
        with self._lock:
            return self._active

    def __enter__(self) -> MicrophoneRecorder:
        """Context manager entry activates the recorder."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit stops the recorder."""
        self.stop()

    # --------------------------------------------------------------------------
    # Audio Frame Generation & Capture
    # --------------------------------------------------------------------------

    def _generate_pcm_frames(self, num_frames: int) -> bytes:
        """Record audio from the system microphone."""

        if self._audio_source_callback is not None:
            try:
             data = self._audio_source_callback(num_frames)
             if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
            except Exception as exc:
                self._logger.warning(f"Error in audio_source_callback: {exc}")

        recording = sd.rec(
            num_frames,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        )

        sd.wait()

        return recording.tobytes()

    def record(
        self,
        duration: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Record audio for the specified duration and save to a .wav file.

        Args:
            duration: Duration to record in seconds. Defaults to self.record_duration.
            output_path: Optional target file path. If None, generates in output_dir.

        Returns:
            Resolved Path to the recorded .wav file.

        Raises:
            MicrophoneError: If recording fails or duration is invalid.
        """
        actual_duration = float(duration) if duration is not None else self._record_duration
        if actual_duration <= 0:
            raise MicrophoneError(f"Recording duration must be positive, got {actual_duration}")

        if output_path is not None:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            unique_name = f"mic_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            out_file = self._output_dir / unique_name

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_RECORDING,
                payload={"duration": actual_duration, "output_path": str(out_file)},
                source=EVENT_SOURCE_MICROPHONE,
            )

        try:
            recording = sd.rec(
             int(actual_duration * self._sample_rate),
             samplerate=self._sample_rate,
             channels=self._channels,
             dtype="int16",
            )

            sd.wait()

            sf.write(
            str(out_file),
            recording,
            self._sample_rate,
            subtype="PCM_16",
            )

        except Exception as exc:
            err_msg = f"Failed to record audio to '{out_file}': {exc}"
            self._logger.error(err_msg, exc_info=True)
            raise RecordingError(err_msg) from exc

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_RECORDED,
                payload={
                    "output_path": str(out_file),
                    "duration": actual_duration,
                    "file_size": out_file.stat().st_size if out_file.exists() else 0,
                },
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.debug(f"Audio recorded successfully to '{out_file}' ({actual_duration}s).")
        return out_file

    def capture_audio(self, duration: Optional[float] = None) -> Optional[Path]:
        """Capture an audio sample if recorder is active.

        Implements AudioProvider.capture_audio contract.

        Returns:
            Path to recorded audio (.wav) file, or None if recorder is not active.
        """
        with self._lock:
            if not self._active:
                self._logger.debug("MicrophoneRecorder.capture_audio called while inactive; returning None.")
                return None

        try:
            return self.record(duration=duration)
        except Exception as exc:
            self._logger.error(f"Error capturing audio from microphone: {exc}")
            raise AudioProviderError(f"Microphone audio capture failed: {exc}") from exc

    async def capture_audio_async(self, duration: Optional[float] = None) -> Optional[Path]:
        """Asynchronously capture an audio sample if recorder is active.

        Runs capture_audio() in a background thread pool.

        Returns:
            Path to recorded audio (.wav) file, or None if recorder is not active.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.capture_audio, duration)

    def close(self) -> None:
        """Release recorder resources and shutdown executor."""
        self.stop()
        self._executor.shutdown(wait=False)


# ------------------------------------------------------------------------------
# Default Singleton Instance
# ------------------------------------------------------------------------------
microphone_recorder = MicrophoneRecorder(auto_register_in_container=True)
microphone = microphone_recorder

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_RECORD_DURATION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_WIDTH",
    "EVENT_MICROPHONE_RECORDED",
    "EVENT_MICROPHONE_RECORDING",
    "EVENT_MICROPHONE_STARTED",
    "EVENT_MICROPHONE_STOPPED",
    "EVENT_SOURCE_MICROPHONE",
    "MicrophoneRecorder",
    "microphone",
    "microphone_recorder",
]
