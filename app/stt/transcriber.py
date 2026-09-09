"""Speech-to-Text (STT) Subsystem for J.A.R.V.I.S.

Provides CPU-only, thread-safe, async-ready audio file transcription powered by
faster-whisper, with lazy model loading, in-memory model caching, lifecycle events,
and seamless integration with Config, Logger, Event Bus, and Service Container.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Final, Optional, Union

from app.core.config import Settings, settings
from app.core.constants import (
    DEFAULT_STT_COMPUTE_TYPE,
    DEFAULT_STT_DEVICE,
    DEFAULT_STT_MODEL,
)
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.stt.models import (
    AudioFileNotFoundError,
    InvalidAudioFormatError,
    ModelLoadError,
    STTError,
    TranscriptionError,
    TranscriptionResult,
)

# Event Constants
EVENT_STT_STARTED: Final[str] = "stt.started"
EVENT_STT_COMPLETED: Final[str] = "stt.completed"
EVENT_STT_FAILED: Final[str] = "stt.failed"
EVENT_SOURCE_STT: Final[str] = "stt"

# Supported file extensions (initially .wav only)
SUPPORTED_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav"})


class SpeechToText:
    """Speech-to-Text transcriber using faster-whisper.

    Features:
    - Lazy loading of Whisper models on first use.
    - Thread-safe model caching and execution.
    - Enforced CPU mode (`device='cpu'`, `compute_type='int8'`).
    - Synchronous and asynchronous transcription APIs.
    - File validation restricted to `.wav` files initially.
    - Event Bus notifications for lifecycle (`stt.started`, `stt.completed`, `stt.failed`).
    - Service Container and Logger integration.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        compute_type: str = DEFAULT_STT_COMPUTE_TYPE,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the SpeechToText subsystem.

        Args:
            model_name: Optional Whisper model name (e.g. 'base', 'tiny', 'small').
                If None, resolved from active configuration or defaults to 'base'.
            config: Optional Settings instance. If None, resolved from container or global settings.
            logger: Optional Logger instance. If None, creates 'STT' logger.
            container_instance: Optional ServiceContainer. If None, uses global container.
            event_bus_instance: Optional EventBus. If None, resolves from container or global event bus.
            compute_type: Computation quantization type. Defaults to 'int8' for CPU efficiency.
            auto_register_in_container: If True, registers this instance into Service Container.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = (
            container_instance if container_instance is not None else container
        )

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("STT")

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

        # 5. Model configuration & caching
        if model_name is not None and model_name.strip():
            self._model_name = model_name.strip()
        elif hasattr(self._config, "stt_model") and self._config.stt_model:
            self._model_name = self._config.stt_model.strip()
        else:
            self._model_name = DEFAULT_STT_MODEL

        # CPU mode only
        self._device: Final[str] = DEFAULT_STT_DEVICE  # "cpu"
        self._compute_type: str = compute_type

        # In-memory model cache: maps model_name -> WhisperModel instance
        self._model_cache: dict[str, Any] = {}

        # 6. Auto-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("stt", self, allow_override=True)
                self._container.register_singleton(
                    "speech_to_text", self, allow_override=True
                )
                self._logger.debug("Registered 'stt' singleton in Service Container.")
            except Exception as exc:
                self._logger.warning(
                    f"Could not register SpeechToText in container: {exc}"
                )

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Currently active model name."""
        with self._lock:
            return self._model_name

    @property
    def device(self) -> str:
        """Active compute device (always 'cpu')."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Active computation precision type."""
        with self._lock:
            return self._compute_type

    @property
    def is_model_loaded(self) -> bool:
        """Check if the currently active model is loaded in memory."""
        with self._lock:
            return self._model_name in self._model_cache

    @property
    def config(self) -> Settings:
        """Retrieve active configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Retrieve active logger."""
        return self._logger

    @property
    def event_bus(self) -> EventBus:
        """Retrieve active event bus."""
        return self._event_bus

    @property
    def container(self) -> ServiceContainer:
        """Retrieve active service container."""
        return self._container

    # --------------------------------------------------------------------------
    # Model Management
    # --------------------------------------------------------------------------

    def set_model(self, model_name: str) -> None:
        """Change the active Whisper model.

        Args:
            model_name: Identifier for the model (e.g. 'tiny', 'base', 'small').

        Raises:
            ValueError: If model_name is empty or not a string.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

        clean_name = model_name.strip()
        with self._lock:
            if clean_name != self._model_name:
                self._logger.info(
                    f"Switching STT model from '{self._model_name}' to '{clean_name}'"
                )
                self._model_name = clean_name

    def clear_cache(self) -> None:
        """Clear all loaded models from memory cache."""
        with self._lock:
            self._model_cache.clear()
            self._logger.debug("Cleared STT model cache.")

    def load_model(self, model_name: Optional[str] = None) -> Any:
        """Load and cache the WhisperModel instance for the specified or active model.

        Thread-safe and lazily invoked on first transcription.

        Args:
            model_name: Optional model to load. If None, uses active `self.model_name`.

        Returns:
            The loaded `WhisperModel` instance.

        Raises:
            ModelLoadError: If loading fails due to import errors, missing files, or network issues.
        """
        target_model = model_name.strip() if model_name and model_name.strip() else self.model_name

        with self._lock:
            if target_model in self._model_cache:
                return self._model_cache[target_model]

            self._logger.info(
                f"Loading faster-whisper model '{target_model}' on {self._device} ({self._compute_type})..."
            )

            try:
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]

                download_root: Optional[str] = None
                if hasattr(self._config, "models_dir") and self._config.models_dir:
                    download_root = str(self._config.models_dir)

                model = WhisperModel(
                    target_model,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=download_root,
                )
                self._model_cache[target_model] = model
                self._logger.info(
                    f"Successfully loaded and cached STT model '{target_model}'."
                )
                return model
            except Exception as exc:
                err_msg = (
                    f"Failed to load faster-whisper model '{target_model}' "
                    f"on {self._device}: {exc}"
                )
                self._logger.error(err_msg, exc_info=True)
                raise ModelLoadError(err_msg) from exc

    # --------------------------------------------------------------------------
    # File Validation
    # --------------------------------------------------------------------------

    def validate_audio_file(self, audio_path: Union[str, Path]) -> Path:
        """Validate that the audio file exists and has a supported .wav extension.

        Args:
            audio_path: Filepath to the audio file.

        Returns:
            Resolved Path object.

        Raises:
            AudioFileNotFoundError: If the file does not exist or is not a regular file.
            InvalidAudioFormatError: If the file format is not .wav.
        """
        if audio_path is None:
            raise AudioFileNotFoundError("Audio path cannot be None.")

        path = Path(audio_path).resolve()

        if not path.exists() or not path.is_file():
            raise AudioFileNotFoundError(f"Audio file not found: '{audio_path}'")

        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise InvalidAudioFormatError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Only {sorted(SUPPORTED_AUDIO_EXTENSIONS)} files are supported."
            )

        return path

    # --------------------------------------------------------------------------
    # Transcription Core
    # --------------------------------------------------------------------------

    def _execute_transcription(
        self,
        audio_file: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Internal synchronous execution of transcription logic."""
        start_time = time.perf_counter()
        target_model_name = self.model_name

        model = self.load_model(target_model_name)

        target_lang = language.strip().lower() if language and language.strip() else None
        if target_lang is None and hasattr(self._config, "language"):
            # If language is configured as 'en' or 'hi', pass to whisper
            cfg_lang = self._config.language.lower()
            if cfg_lang in ("en", "hi"):
                target_lang = cfg_lang

        try:
            
            segments_gen, info = model.transcribe(
            str(audio_file),
            language=target_lang,
            beam_size=10,
            best_of=10,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
            "min_silence_duration_ms": 500,
                },
            )
            segment_list: list[dict[str, Any]] = []
            text_chunks: list[str] = []

            for seg in segments_gen:
                segment_dict = {
                    "id": getattr(seg, "id", 0),
                    "seek": getattr(seg, "seek", 0),
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", "").strip(),
                    "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                    "compression_ratio": getattr(seg, "compression_ratio", 0.0),
                    "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                }
                segment_list.append(segment_dict)
                if seg.text:
                    text_chunks.append(seg.text.strip())

            full_text = " ".join(text_chunks).strip()
            execution_time = time.perf_counter() - start_time

            detected_lang = getattr(info, "language", target_lang or "en")
            lang_prob = getattr(info, "language_probability", 1.0)
            duration = getattr(info, "duration", 0.0)

            return TranscriptionResult(
                text=full_text,
                language=str(detected_lang),
                language_probability=float(lang_prob),
                duration=float(duration),
                segments=tuple(segment_list),
                model_name=target_model_name,
                execution_time=execution_time,
                timestamp=time.time(),
            )
        except Exception as exc:
            err_msg = f"Transcription failed for '{audio_file}': {exc}"
            self._logger.error(err_msg, exc_info=True)
            raise TranscriptionError(err_msg) from exc

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        *,
        publish_event: bool = True,
    ) -> TranscriptionResult:
        """Synchronously transcribe a .wav audio file.

        Args:
            audio_path: Path to the .wav audio file.
            language: Optional language code hint (e.g. 'en', 'hi').
            publish_event: If True, publishes lifecycle events on the Event Bus.

        Returns:
            TranscriptionResult containing transcribed text and metadata.

        Raises:
            AudioFileNotFoundError: If the audio file does not exist.
            InvalidAudioFormatError: If the file is not a .wav file.
            ModelLoadError: If the STT model fails to load.
            TranscriptionError: If transcription processing fails.
        """
        path_str = str(audio_path)

        # 1. Validate audio file
        audio_file = self.validate_audio_file(audio_path)

        # 2. Publish stt.started event
        if publish_event and self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_STT_STARTED,
                payload={
                    "audio_path": path_str,
                    "model_name": self.model_name,
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_STT,
            )

        # 3. Execute transcription under thread lock
        try:
            with self._lock:
                result = self._execute_transcription(audio_file, language=language)

            self._logger.info(
                f"STT completed in {result.execution_time:.2f}s: '{result.text}' "
                f"[lang={result.language}, dur={result.duration:.1f}s]"
            )

            # 4. Publish stt.completed event
            if publish_event and self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_STT_COMPLETED,
                    payload={
                        "audio_path": path_str,
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "execution_time": result.execution_time,
                        "result": result,
                    },
                    source=EVENT_SOURCE_STT,
                )

            return result

        except Exception as exc:
            # 5. Publish stt.failed event
            if publish_event and self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_STT_FAILED,
                    payload={
                        "audio_path": path_str,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timestamp": time.time(),
                    },
                    source=EVENT_SOURCE_STT,
                )
            raise

    async def transcribe_async(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        *,
        publish_event: bool = True,
    ) -> TranscriptionResult:
        """Asynchronously transcribe a .wav audio file without blocking the event loop.

        Args:
            audio_path: Path to the .wav audio file.
            language: Optional language code hint (e.g. 'en', 'hi').
            publish_event: If True, publishes lifecycle events asynchronously on Event Bus.

        Returns:
            TranscriptionResult containing transcribed text and metadata.

        Raises:
            AudioFileNotFoundError: If the audio file does not exist.
            InvalidAudioFormatError: If the file is not a .wav file.
            ModelLoadError: If the STT model fails to load.
            TranscriptionError: If transcription processing fails.
        """
        path_str = str(audio_path)

        # 1. Validate audio file
        audio_file = self.validate_audio_file(audio_path)

        # 2. Publish stt.started asynchronously
        if publish_event and self._event_bus is not None:
            await self._event_bus.publish_async(
                event=EVENT_STT_STARTED,
                payload={
                    "audio_path": path_str,
                    "model_name": self.model_name,
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_STT,
            )

        # 3. Offload CPU-bound transcription to background thread worker
        try:
            def _run() -> TranscriptionResult:
                with self._lock:
                    return self._execute_transcription(audio_file, language=language)

            result = await asyncio.to_thread(_run)

            self._logger.info(
                f"STT completed (async) in {result.execution_time:.2f}s: '{result.text}'"
            )

            # 4. Publish stt.completed asynchronously
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_STT_COMPLETED,
                    payload={
                        "audio_path": path_str,
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "execution_time": result.execution_time,
                        "result": result,
                    },
                    source=EVENT_SOURCE_STT,
                )

            return result

        except Exception as exc:
            # 5. Publish stt.failed asynchronously
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_STT_FAILED,
                    payload={
                        "audio_path": path_str,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timestamp": time.time(),
                    },
                    source=EVENT_SOURCE_STT,
                )
            raise


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
speech_to_text: Final[SpeechToText] = SpeechToText()

__all__ = [
    "EVENT_SOURCE_STT",
    "EVENT_STT_COMPLETED",
    "EVENT_STT_FAILED",
    "EVENT_STT_STARTED",
    "SpeechToText",
    "speech_to_text",
]
