"""Text-to-Speech (TTS) Subsystem for J.A.R.V.I.S.

Provides Microsoft Edge TTS powered speech synthesis with configurable voices,
rate, volume, pitch, temporary audio file caching, lazy initialization,
thread-safe execution, async-readiness, and event bus lifecycle notifications.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Coroutine, Final, Optional, TypeVar, Union

from app.core.config import Settings, settings
from app.core.constants import (
    DEFAULT_TTS_PITCH,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOICE_EN,
    DEFAULT_TTS_VOLUME,
)
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.tts.models import (
    AudioFileError,
    ConfigurationError,
    EmptyTextError,
    SpeechResult,
    SynthesisError,
    TTSError,
    VoiceNotFoundError,
)

T = TypeVar("T")

# Event Constants
EVENT_TTS_STARTED: Final[str] = "tts.started"
EVENT_TTS_COMPLETED: Final[str] = "tts.completed"
EVENT_TTS_FAILED: Final[str] = "tts.failed"
EVENT_SOURCE_TTS: Final[str] = "tts"


def format_percentage(val: Union[str, int, float, None], default: str = "+0%") -> str:
    """Format and validate a percentage value for rate and volume.

    Accepts strings (e.g. '+10%', '-5%', '20%'), integers/floats (e.g. 10, -5, 0),
    or None (returns default).

    Args:
        val: Input percentage representation.
        default: Default fallback if val is None.

    Returns:
        Standardized string in the format '+X%' or '-X%'.

    Raises:
        ConfigurationError: If val cannot be parsed as a valid percentage.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        int_val = int(round(val))
        return f"+{int_val}%" if int_val >= 0 else f"{int_val}%"

    s = str(val).strip()
    if not s:
        return default
    if not s.endswith("%"):
        s = f"{s}%"
    if not (s.startswith("+") or s.startswith("-")):
        s = f"+{s}"

    # Verify numeric component contains only digits
    num_part = s[1:-1]
    if not num_part.isdigit():
        raise ConfigurationError(
            f"Invalid percentage value '{val}'. Expected format like '+10%' or '-5%'."
        )
    return s


def format_pitch(val: Union[str, int, float, None], default: str = "+0Hz") -> str:
    """Format and validate a pitch offset value in Hz.

    Accepts strings (e.g. '+5Hz', '-10Hz', '15Hz'), integers/floats (e.g. 5, -10, 0),
    or None (returns default).

    Args:
        val: Input pitch representation.
        default: Default fallback if val is None.

    Returns:
        Standardized string in the format '+XHz' or '-XHz'.

    Raises:
        ConfigurationError: If val cannot be parsed as a valid pitch in Hz.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        int_val = int(round(val))
        return f"+{int_val}Hz" if int_val >= 0 else f"{int_val}Hz"

    s = str(val).strip()
    if not s:
        return default
    if not s.lower().endswith("hz"):
        s = f"{s}Hz"
    else:
        s = f"{s[:-2]}Hz"
    if not (s.startswith("+") or s.startswith("-")):
        s = f"+{s}"

    # Verify numeric component contains only digits
    num_part = s[1:-2]
    if not num_part.isdigit():
        raise ConfigurationError(
            f"Invalid pitch value '{val}'. Expected format like '+5Hz' or '-10Hz'."
        )
    return s


class TextToSpeech:
    """Text-to-Speech synthesizer using Microsoft edge-tts.

    Features:
    - Lazy initialization of temporary workspace and dependencies.
    - Thread-safe configuration changes and synthesis execution.
    - Configurable voice, speaking rate, volume, and pitch.
    - Automatic saving to temporary audio files or specified target destinations.
    - Synchronous and asynchronous synthesis APIs.
    - Lifecycle event notifications (`tts.started`, `tts.completed`, `tts.failed`).
    - Service Container, Logger, and Config integration.
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        rate: Optional[Union[str, int, float]] = None,
        volume: Optional[Union[str, int, float]] = None,
        pitch: Optional[Union[str, int, float]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        temp_dir: Optional[Union[str, Path]] = None,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the TextToSpeech subsystem.

        Args:
            voice: Optional TTS voice identifier (e.g. 'en-GB-RyanNeural').
            rate: Optional speaking rate offset (e.g. '+0%', '+10%').
            volume: Optional volume offset (e.g. '+0%', '-10%').
            pitch: Optional pitch offset (e.g. '+0Hz', '+5Hz').
            config: Optional Settings instance. Defaults to global settings.
            logger: Optional Logger instance. Defaults to 'TTS' logger.
            container_instance: Optional ServiceContainer. Defaults to global container.
            event_bus_instance: Optional EventBus. Defaults to global event bus.
            temp_dir: Optional directory path for temporary audio storage.
            auto_register_in_container: If True, registers singleton in Service Container.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = (
            container_instance if container_instance is not None else container
        )

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("TTS")

        # 3. Dependency resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Dependency resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = event_bus

        # 5. Configurable Voice Parameters
        if voice is not None and str(voice).strip():
            self._voice = str(voice).strip()
        elif hasattr(self._config, "tts_voice") and self._config.tts_voice:
            self._voice = self._config.tts_voice.strip()
        else:
            self._voice = DEFAULT_TTS_VOICE_EN

        default_cfg_rate = getattr(self._config, "tts_rate", DEFAULT_TTS_RATE)
        default_cfg_vol = getattr(self._config, "tts_volume", DEFAULT_TTS_VOLUME)
        default_cfg_pitch = getattr(self._config, "tts_pitch", DEFAULT_TTS_PITCH)

        self._rate = format_percentage(rate, default=default_cfg_rate)
        self._volume = format_percentage(volume, default=default_cfg_vol)
        self._pitch = format_pitch(pitch, default=default_cfg_pitch)

        # 6. Temporary audio storage configuration
        if temp_dir is not None:
            self._temp_dir = Path(temp_dir).resolve()
        elif hasattr(self._config, "data_dir") and self._config.data_dir:
            self._temp_dir = Path(self._config.data_dir) / "tts"
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "jarvis_tts"

        # 7. Lazy initialization state
        self._initialized: bool = False

        # 8. Service Container registration
        if auto_register_in_container:
            try:
                self._container.register_singleton("tts", self, allow_override=True)
                self._container.register_singleton(
                    "text_to_speech", self, allow_override=True
                )
                self._logger.debug("Registered 'tts' singleton in Service Container.")
            except Exception as exc:
                self._logger.warning(
                    f"Could not register TextToSpeech in container: {exc}"
                )

    # --------------------------------------------------------------------------
    # Properties & Setters
    # --------------------------------------------------------------------------

    @property
    def voice(self) -> str:
        """Currently active TTS voice model."""
        with self._lock:
            return self._voice

    @property
    def rate(self) -> str:
        """Currently active speaking rate offset."""
        with self._lock:
            return self._rate

    @property
    def volume(self) -> str:
        """Currently active volume offset."""
        with self._lock:
            return self._volume

    @property
    def pitch(self) -> str:
        """Currently active pitch offset."""
        with self._lock:
            return self._pitch

    @property
    def temp_dir(self) -> Path:
        """Active directory path for temporary audio storage."""
        with self._lock:
            return self._temp_dir

    @property
    def is_initialized(self) -> bool:
        """Check whether lazy initialization has been performed."""
        with self._lock:
            return self._initialized

    @property
    def config(self) -> Settings:
        """Active Settings instance."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Active Logger instance."""
        return self._logger

    @property
    def event_bus(self) -> EventBus:
        """Active EventBus instance."""
        return self._event_bus

    @property
    def container(self) -> ServiceContainer:
        """Active ServiceContainer instance."""
        return self._container

    def set_voice(self, voice: str) -> None:
        """Change the active TTS voice model.

        Args:
            voice: Voice identifier (e.g. 'en-US-GuyNeural').

        Raises:
            ConfigurationError: If voice is empty or not a string.
        """
        if not isinstance(voice, str) or not voice.strip():
            raise ConfigurationError("Voice must be a non-empty string.")

        clean_voice = voice.strip()
        with self._lock:
            if clean_voice != self._voice:
                self._logger.info(
                    f"Switching TTS voice from '{self._voice}' to '{clean_voice}'"
                )
                self._voice = clean_voice

    def set_rate(self, rate: Union[str, int, float]) -> None:
        """Change the active speaking rate offset.

        Args:
            rate: Speaking rate offset (e.g. '+10%', '-5%', 20).

        Raises:
            ConfigurationError: If rate format is invalid.
        """
        formatted = format_percentage(rate)
        with self._lock:
            if formatted != self._rate:
                self._logger.info(
                    f"Switching TTS rate from '{self._rate}' to '{formatted}'"
                )
                self._rate = formatted

    def set_volume(self, volume: Union[str, int, float]) -> None:
        """Change the active volume offset.

        Args:
            volume: Volume offset (e.g. '+10%', '-10%', 0).

        Raises:
            ConfigurationError: If volume format is invalid.
        """
        formatted = format_percentage(volume)
        with self._lock:
            if formatted != self._volume:
                self._logger.info(
                    f"Switching TTS volume from '{self._volume}' to '{formatted}'"
                )
                self._volume = formatted

    def set_pitch(self, pitch: Union[str, int, float]) -> None:
        """Change the active pitch offset.

        Args:
            pitch: Pitch offset in Hz (e.g. '+5Hz', '-10Hz', 5).

        Raises:
            ConfigurationError: If pitch format is invalid.
        """
        formatted = format_pitch(pitch)
        with self._lock:
            if formatted != self._pitch:
                self._logger.info(
                    f"Switching TTS pitch from '{self._pitch}' to '{formatted}'"
                )
                self._pitch = formatted

    # --------------------------------------------------------------------------
    # Lazy Initialization
    # --------------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the TTS subsystem and ensure temporary directories exist.

        Thread-safe and idempotent.
        """
        with self._lock:
            if self._initialized:
                return

            try:
                self._temp_dir.mkdir(parents=True, exist_ok=True)
                self._initialized = True
                self._logger.debug(
                    f"TextToSpeech initialized [voice='{self._voice}', "
                    f"rate='{self._rate}', vol='{self._volume}', "
                    f"pitch='{self._pitch}', dir='{self._temp_dir}']"
                )
            except Exception as exc:
                raise AudioFileError(
                    f"Failed to initialize TTS storage directory '{self._temp_dir}': {exc}"
                ) from exc

    def ensure_initialized(self) -> None:
        """Ensure that the subsystem is initialized before executing operations."""
        if not self._initialized:
            self.initialize()

    # --------------------------------------------------------------------------
    # Duration Calculation
    # --------------------------------------------------------------------------

    def _estimate_or_probe_duration(
        self, audio_path: Path, text: str, file_size: int
    ) -> float:
        """Estimate audio playback duration based on file size and word count."""
        # edge-tts MP3 stream bitrates average ~48kbps (~6000 bytes/second)
        if file_size > 0:
            return round(max(0.5, file_size / 6000.0), 2)
        words = len(text.split())
        # Average conversational speaking rate ~150 words/min = 2.5 words/sec
        return round(max(0.5, words / 2.5), 2)

    # --------------------------------------------------------------------------
    # Synchronous Execution Helper
    # --------------------------------------------------------------------------

    @staticmethod
    def _run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
        """Execute a coroutine synchronously without deadlocking existing event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    # --------------------------------------------------------------------------
    # Speech Synthesis APIs
    # --------------------------------------------------------------------------

    async def synthesize_async(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        voice: Optional[str] = None,
        rate: Optional[Union[str, int, float]] = None,
        volume: Optional[Union[str, int, float]] = None,
        pitch: Optional[Union[str, int, float]] = None,
        *,
        publish_event: bool = True,
    ) -> SpeechResult:
        """Asynchronously synthesize text to an audio file using edge-tts.

        Args:
            text: Text string to convert to speech.
            output_path: Optional destination Path or str for the .mp3 file.
                If None, saves to a unique temporary file in temp_dir.
            voice: Optional voice override for this synthesis call.
            rate: Optional speaking rate override (e.g. '+10%').
            volume: Optional volume override (e.g. '-10%').
            pitch: Optional pitch override (e.g. '+5Hz').
            publish_event: If True, emits lifecycle events on the Event Bus.

        Returns:
            SpeechResult encapsulating generated audio path and metadata.

        Raises:
            EmptyTextError: If text is empty or whitespace-only.
            ConfigurationError: If any parameter format is invalid.
            AudioFileError: If file path creation or writing fails.
            SynthesisError: If edge-tts synthesis communication fails.
        """
        # 1. Validate text input
        if text is None or not isinstance(text, str) or not text.strip():
            raise EmptyTextError("Text for speech synthesis must be a non-empty string.")

        clean_text = text.strip()

        # 2. Ensure subsystem is initialized
        self.ensure_initialized()

        # 3. Resolve parameters under thread lock
        with self._lock:
            target_voice = voice.strip() if voice and voice.strip() else self._voice
            target_rate = (
                format_percentage(rate, default=self._rate)
                if rate is not None
                else self._rate
            )
            target_volume = (
                format_percentage(volume, default=self._volume)
                if volume is not None
                else self._volume
            )
            target_pitch = (
                format_pitch(pitch, default=self._pitch)
                if pitch is not None
                else self._pitch
            )
            active_temp_dir = self._temp_dir

        # 4. Resolve output audio destination
        is_temporary = output_path is None
        try:
            if is_temporary:
                fd, raw_temp = tempfile.mkstemp(
                    suffix=".mp3", prefix="tts_", dir=str(active_temp_dir)
                )
                os.close(fd)
                target_output_path = Path(raw_temp).resolve()
            else:
                target_output_path = Path(output_path).resolve()
                target_output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise AudioFileError(f"Failed to prepare audio output path: {exc}") from exc

        # 5. Publish tts.started event
        if publish_event and self._event_bus is not None:
            await self._event_bus.publish_async(
                event=EVENT_TTS_STARTED,
                payload={
                    "text": clean_text,
                    "voice": target_voice,
                    "rate": target_rate,
                    "volume": target_volume,
                    "pitch": target_pitch,
                    "output_path": str(target_output_path),
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_TTS,
            )

        # 6. Perform speech synthesis
        start_time = time.perf_counter()
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=target_voice,
                rate=target_rate,
                volume=target_volume,
                pitch=target_pitch,
            )
            await communicate.save(str(target_output_path))
            execution_time = time.perf_counter() - start_time

            # Validate generated file
            if (
                not target_output_path.exists()
                or target_output_path.stat().st_size == 0
            ):
                raise SynthesisError(
                    f"Generated audio file is missing or empty: {target_output_path}"
                )

            file_size = target_output_path.stat().st_size
            duration = self._estimate_or_probe_duration(
                target_output_path, clean_text, file_size
            )

            result = SpeechResult(
                text=clean_text,
                audio_path=target_output_path,
                voice=target_voice,
                rate=target_rate,
                volume=target_volume,
                pitch=target_pitch,
                duration=duration,
                file_size_bytes=file_size,
                execution_time=execution_time,
                timestamp=time.time(),
            )

            preview = clean_text[:40] + ("..." if len(clean_text) > 40 else "")
            self._logger.info(
                f"TTS synthesis completed in {execution_time:.2f}s: "
                f"'{preview}' -> {target_output_path.name} "
                f"({file_size} bytes, voice={target_voice})"
            )

            # 7. Publish tts.completed event
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_TTS_COMPLETED,
                    payload={
                        "text": clean_text,
                        "voice": target_voice,
                        "audio_path": str(target_output_path),
                        "duration": duration,
                        "file_size_bytes": file_size,
                        "execution_time": execution_time,
                        "result": result,
                        "timestamp": time.time(),
                    },
                    source=EVENT_SOURCE_TTS,
                )

            return result

        except Exception as exc:
            # Clean up empty temp file if failed
            if (
                is_temporary
                and target_output_path.exists()
                and target_output_path.stat().st_size == 0
            ):
                try:
                    target_output_path.unlink(missing_ok=True)
                except Exception:
                    pass

            # 8. Publish tts.failed event
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_TTS_FAILED,
                    payload={
                        "text": clean_text,
                        "voice": target_voice,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timestamp": time.time(),
                    },
                    source=EVENT_SOURCE_TTS,
                )

            if isinstance(
                exc,
                (
                    EmptyTextError,
                    ConfigurationError,
                    VoiceNotFoundError,
                    AudioFileError,
                    SynthesisError,
                ),
            ):
                raise
            raise SynthesisError(
                f"TTS synthesis failed for text '{clean_text[:30]}': {exc}"
            ) from exc

    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        voice: Optional[str] = None,
        rate: Optional[Union[str, int, float]] = None,
        volume: Optional[Union[str, int, float]] = None,
        pitch: Optional[Union[str, int, float]] = None,
        *,
        publish_event: bool = True,
    ) -> SpeechResult:
        """Synchronously synthesize text to an audio file using edge-tts.

        Args:
            text: Text string to convert to speech.
            output_path: Optional destination Path or str for the .mp3 file.
                If None, saves to a unique temporary file in temp_dir.
            voice: Optional voice override for this synthesis call.
            rate: Optional speaking rate override (e.g. '+10%').
            volume: Optional volume override (e.g. '-10%').
            pitch: Optional pitch override (e.g. '+5Hz').
            publish_event: If True, emits lifecycle events on the Event Bus.

        Returns:
            SpeechResult encapsulating generated audio path and metadata.

        Raises:
            EmptyTextError: If text is empty or whitespace-only.
            ConfigurationError: If any parameter format is invalid.
            AudioFileError: If file path creation or writing fails.
            SynthesisError: If edge-tts synthesis communication fails.
        """
        return self._run_coroutine(
            self.synthesize_async(
                text=text,
                output_path=output_path,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                publish_event=publish_event,
            )
        )

    # --------------------------------------------------------------------------
    # Voices Discovery & Cleanup Utilities
    # --------------------------------------------------------------------------

    async def list_available_voices(self) -> list[dict[str, Any]]:
        """Asynchronously query Microsoft Edge TTS for all available voices.

        Returns:
            List of voice metadata dictionaries.
        """
        try:
            import edge_tts

            voices: list[dict[str, Any]] = await edge_tts.list_voices()
            return voices
        except Exception as exc:
            self._logger.warning(f"Failed to list voices from edge-tts: {exc}")
            return []

    def get_available_voices(self) -> list[dict[str, Any]]:
        """Synchronously query Microsoft Edge TTS for all available voices.

        Returns:
            List of voice metadata dictionaries.
        """
        return self._run_coroutine(self.list_available_voices())

    def cleanup_temp_files(self) -> int:
        """Remove any temporary audio files generated in the active temp directory.

        Returns:
            Number of files removed.
        """
        with self._lock:
            if not self._temp_dir.exists():
                return 0
            count = 0
            for file_path in self._temp_dir.glob("tts_*.mp3"):
                try:
                    file_path.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
            return count


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
text_to_speech: Final[TextToSpeech] = TextToSpeech()

__all__ = [
    "EVENT_SOURCE_TTS",
    "EVENT_TTS_COMPLETED",
    "EVENT_TTS_FAILED",
    "EVENT_TTS_STARTED",
    "TextToSpeech",
    "format_percentage",
    "format_pitch",
    "text_to_speech",
]
