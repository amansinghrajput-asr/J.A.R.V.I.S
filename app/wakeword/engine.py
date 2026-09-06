"""Production Wake Word Engine for J.A.R.V.I.S using openWakeWord.

Provides continuous, background wake-word detection using openwakeword ONNX models.
Features:
- Default wake word: "Jarvis" (mapped to 'hey_jarvis_v0.1.onnx').
- Continuous listening in a background daemon thread.
- Thread-safe start, stop, pause, and resume lifecycle.
- Automatic EventBus event notifications:
    - wakeword.started
    - wakeword.detected
    - wakeword.stopped
    - wakeword.failed
- Automatic integration with ServiceContainer.
- Automatic integration with VoiceConversationEngine:
    When "Jarvis" is detected, executes `voice_engine.listen_once()` automatically.
- Direct audio feeding support for tests and custom audio pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Final, Generator, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np

from app.core.config import Settings, settings as default_settings
from app.core.constants import DEFAULT_WAKE_WORD
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.voice.engine import VoiceConversationEngine

from app.wakeword.detector import (
    EVENT_SOURCE_WAKEWORD,
    EVENT_WAKEWORD_DETECTED,
    EVENT_WAKEWORD_FAILED,
    EVENT_WAKEWORD_STARTED,
    EVENT_WAKEWORD_STOPPED,
)
from app.wakeword.models import (
    AudioStreamError,
    ModelLoadError,
    WakeWordError,
    WakeWordResult,
)

# Audio Standards for openwakeword
CHUNK_SIZE_SAMPLES: Final[int] = 1280  # 80ms of 16kHz audio
SAMPLE_RATE: Final[int] = 16000
SAMPLE_WIDTH: Final[int] = 2  # 16-bit PCM
DEFAULT_DETECTION_THRESHOLD: Final[float] = 0.5
DEFAULT_COOLDOWN_SECONDS: Final[float] = 2.0


def _resolve_model_path(wake_word: str) -> str:
    """Resolve the ONNX model path for a given wake word name.

    Args:
        wake_word: Wake word identifier (e.g., 'Jarvis', 'hey_jarvis').

    Returns:
        Path or model name identifier for openwakeword.
    """
    normalized = wake_word.strip().lower().replace(" ", "_")

    # If it's a file path that exists, return it
    p = Path(wake_word)
    if p.is_file():
        return str(p.resolve())

    try:
        import openwakeword
        from openwakeword.utils import download_models

        # Check installed openwakeword resources directory
        resource_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        
        # Target candidates for 'jarvis' or 'hey_jarvis'
        candidates = []
        if "jarvis" in normalized:
            candidates.extend([
                resource_dir / "hey_jarvis_v0.1.onnx",
                resource_dir / "hey_jarvis.onnx",
            ])
        candidates.append(resource_dir / f"{normalized}.onnx")
        candidates.append(resource_dir / f"{normalized}_v0.1.onnx")

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

        # If not found locally, attempt to download
        model_to_download = "hey_jarvis" if "jarvis" in normalized else normalized
        download_models(model_names=[model_to_download], target_directory=str(resource_dir))

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

    except Exception:
        pass

    # Fallback to model name
    if "jarvis" in normalized:
        return "hey_jarvis_v0.1.onnx"
    return f"{normalized}.onnx"


class WakeWordEngine:
    """Production Wake Word Engine powered by openWakeWord.

    Runs in a dedicated background thread, listens for "Jarvis" (or custom wake words),
    publishes lifecycle events on the EventBus, and automatically triggers
    `VoiceConversationEngine.listen_once()` upon detection.
    """

    def __init__(
        self,
        wake_word: str = DEFAULT_WAKE_WORD,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        model_path: Optional[str] = None,
        model_instance: Optional[Any] = None,
        voice_engine: Optional[VoiceConversationEngine] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        audio_source_callback: Optional[Callable[[], Optional[Union[bytes, np.ndarray]]]] = None,
        *,
        auto_listen: bool = True,
        auto_register_in_container: bool = True,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        lazy_load_model: bool = False,
    ) -> None:
        """Initialize the WakeWordEngine.

        Args:
            wake_word: Target wake word phrase or identifier. Defaults to 'Jarvis'.
            threshold: Detection confidence threshold (0.0 to 1.0). Defaults to 0.5.
            model_path: Optional explicit path to openwakeword ONNX model file.
            model_instance: Optional pre-loaded openwakeword.Model instance (used for testing).
            voice_engine: Optional VoiceConversationEngine instance to trigger on detection.
            config: Optional Settings instance. Defaults to container/global settings.
            logger: Optional Logger instance. Defaults to 'WAKEWORD_ENGINE' logger.
            container_instance: Optional ServiceContainer instance. Defaults to global container.
            event_bus_instance: Optional EventBus instance. Defaults to container/global event bus.
            audio_source_callback: Optional callable providing audio chunks in streaming loop.
            auto_listen: Whether to automatically call `voice_engine.listen_once()` on detection.
            auto_register_in_container: Whether to register self into ServiceContainer.
            cooldown_seconds: Cooldown in seconds after a detection before re-triggering.
            lazy_load_model: If True, delays loading the model until start() or first feed_audio().
        """
        self._lock = threading.RLock()
        self._wake_word = wake_word.strip() if wake_word else DEFAULT_WAKE_WORD
        self._threshold = float(threshold)
        self._cooldown_seconds = float(cooldown_seconds)
        self._last_detection_time: float = 0.0
        self._auto_listen = bool(auto_listen)
        self._audio_source_callback = audio_source_callback

        # Threading state
        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=100)

        # 1. Dependency Resolution: Service Container
        self._container = (
            container_instance if container_instance is not None else default_container
        )

        # 2. Dependency Resolution: Logger
        self._logger = logger if logger is not None else get_logger("WAKEWORD_ENGINE")

        # 3. Dependency Resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = default_settings

        # 4. Dependency Resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = default_event_bus

        # 5. Dependency Resolution: VoiceConversationEngine
        if voice_engine is not None:
            self._voice_engine: Optional[VoiceConversationEngine] = voice_engine
        elif self._container.exists("voice_engine"):
            self._voice_engine = self._container.resolve("voice_engine")
        elif self._container.exists("voice_conversation_engine"):
            self._voice_engine = self._container.resolve("voice_conversation_engine")
        else:
            self._voice_engine = None

        # 6. Initialize openwakeword Model
        self._model_path = model_path if model_path else _resolve_model_path(self._wake_word)
        self._model = model_instance
        self._model_key: Optional[str] = None

        if self._model is not None:
            if hasattr(self._model, "models") and isinstance(self._model.models, dict):
                self._model_key = next(iter(self._model.models.keys()), None)
        elif not lazy_load_model:
            self._init_model()

        # 7. Service Container Self-Registration
        if auto_register_in_container:
            for alias in ("wakeword_engine", "wake_word_engine", "wakeword_listener", "wake_word"):
                try:
                    self._container.register_singleton(alias, self, allow_override=True)
                except Exception as exc:
                    self._logger.warning("Could not register '%s' into container: %s", alias, exc)

    def _init_model(self) -> None:
        """Initialize openwakeword Model with ONNX runtime."""
        if self._model is not None:
            return
        try:
            import openwakeword
            from openwakeword.model import Model

            self._logger.debug("Loading openwakeword model from '%s'...", self._model_path)
            self._model = Model(
                wakeword_models=[self._model_path],
                inference_framework="onnx",
            )
            if hasattr(self._model, "models") and self._model.models:
                self._model_key = next(iter(self._model.models.keys()))
            else:
                self._model_key = Path(self._model_path).name

            self._logger.info(
                "openWakeWord model '%s' loaded successfully for wake word '%s'.",
                self._model_key,
                self._wake_word,
            )
        except Exception as exc:
            err_msg = f"Failed to initialize openWakeWord model '{self._model_path}': {exc}"
            self._logger.error(err_msg, exc_info=True)
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_FAILED,
                    payload={"error": err_msg, "stage": "model_init", "timestamp": time.time()},
                    source=EVENT_SOURCE_WAKEWORD,
                )
            raise ModelLoadError(err_msg) from exc

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def wake_word(self) -> str:
        """Return the target wake word string."""
        return self._wake_word

    @property
    def threshold(self) -> float:
        """Return the detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        """Set detection threshold (0.0 to 1.0)."""
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {val}")
        self._threshold = float(val)

    @property
    def is_running(self) -> bool:
        """Check whether the background listening thread is active."""
        with self._lock:
            return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check whether detection is temporarily paused."""
        with self._lock:
            return self._is_paused

    @property
    def model(self) -> Any:
        """Return the underlying openwakeword Model instance."""
        if self._model is None:
            self._init_model()
        return self._model

    @property
    def voice_engine(self) -> Optional[VoiceConversationEngine]:
        """Return the associated VoiceConversationEngine instance."""
        if self._voice_engine is None:
            if self._container.exists("voice_engine"):
                self._voice_engine = self._container.resolve("voice_engine")
            elif self._container.exists("voice_conversation_engine"):
                self._voice_engine = self._container.resolve("voice_conversation_engine")
        return self._voice_engine

    @voice_engine.setter
    def voice_engine(self, engine: Optional[VoiceConversationEngine]) -> None:
        """Set or update the associated VoiceConversationEngine."""
        self._voice_engine = engine

    @property
    def container(self) -> ServiceContainer:
        """Return the active ServiceContainer instance."""
        return self._container

    @property
    def event_bus(self) -> EventBus:
        """Return the active EventBus instance."""
        return self._event_bus

    @property
    def logger(self) -> logging.Logger:
        """Return the active Logger instance."""
        return self._logger

    # --------------------------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Start the continuous wake-word listening loop in a background thread."""
        with self._lock:
            if self._is_running:
                return
            if self._model is None:
                self._init_model()
            self._is_running = True
            self._is_paused = False
            self._stop_event.clear()

        # Emit wakeword.started event
        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_WAKEWORD_STARTED,
                payload={
                    "wake_word": self._wake_word,
                    "threshold": self._threshold,
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_WAKEWORD,
            )

        self._thread = threading.Thread(
            target=self._run_loop,
            name="WakeWordListener",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("WakeWordEngine started listening for '%s'.", self._wake_word)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the background wake-word listening thread cleanly."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        self._stop_event.set()

        try:
            self._audio_queue.put_nowait(None)
        except Exception:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

        # Emit wakeword.stopped event
        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_WAKEWORD_STOPPED,
                payload={"wake_word": self._wake_word, "timestamp": time.time()},
                source=EVENT_SOURCE_WAKEWORD,
            )

        self._logger.info("WakeWordEngine stopped listening.")

    def pause(self) -> None:
        """Temporarily pause wake-word detection (e.g. while assistant is speaking)."""
        with self._lock:
            self._is_paused = True
            self._logger.debug("WakeWordEngine paused.")

    def resume(self) -> None:
        """Resume wake-word detection."""
        with self._lock:
            self._is_paused = False
            self._last_detection_time = time.time()
            self._logger.debug("WakeWordEngine resumed.")

    def close(self) -> None:
        """Release all allocated resources and stop the engine."""
        self.stop()

    def __enter__(self) -> WakeWordEngine:
        """Context manager entry activates the listener."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit stops the listener."""
        self.stop()

    # --------------------------------------------------------------------------
    # Audio Feed & Prediction Processing
    # --------------------------------------------------------------------------

    def feed_audio(
        self,
        chunk: Union[bytes, np.ndarray],
        *,
        trigger_conversation: Optional[bool] = None,
    ) -> WakeWordResult:
        """Feed an audio chunk directly to the model for prediction.

        Accepts 16-bit 16kHz PCM audio bytes or a numpy int16 array.
        Evaluates model score, emits `wakeword.detected` on match, and optionally
        triggers `VoiceConversationEngine.listen_once()`.

        Args:
            chunk: Audio samples as raw bytes or 1D int16 numpy array.
            trigger_conversation: Whether to trigger conversation on match.
                If None, uses self._auto_listen.

        Returns:
            WakeWordResult indicating detection status, confidence score, and matched phrase.
        """
        if self._is_paused:
            return WakeWordResult(detected=False, score=0.0, matched_phrase=None)

        if self._model is None:
            self._init_model()

        # Convert chunk to numpy array
        if isinstance(chunk, (bytes, bytearray)):
            audio_data = np.frombuffer(chunk, dtype=np.int16)
        elif isinstance(chunk, np.ndarray):
            audio_data = chunk.astype(np.int16)
        else:
            raise AudioStreamError(f"Unsupported audio data type: {type(chunk)}")

        # Check cooldown to prevent duplicate triggers
        now = time.time()
        if now - self._last_detection_time < self._cooldown_seconds:
            return WakeWordResult(detected=False, score=0.0, matched_phrase=None)

        try:
            predictions = self._model.predict(audio_data)
        except Exception as exc:
            err_msg = f"Inference error in openwakeword: {exc}"
            self._logger.error(err_msg, exc_info=True)
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_FAILED,
                    payload={"error": err_msg, "stage": "prediction", "timestamp": now},
                    source=EVENT_SOURCE_WAKEWORD,
                )
            return WakeWordResult(detected=False, score=0.0, matched_phrase=None)

        # Evaluate score
        matched_model_key = None
        max_score = 0.0

        if isinstance(predictions, dict):
            for model_name, score in predictions.items():
                if score > max_score:
                    max_score = float(score)
                    matched_model_key = model_name

        detected = max_score >= self._threshold

        if detected:
            self._last_detection_time = now
            result = WakeWordResult(
                detected=True,
                matched_phrase=self._wake_word,
                normalized_text=self._wake_word.lower(),
                timestamp=now,
                score=max_score,
                model_name=matched_model_key,
            )

            self._logger.info(
                "Wake word '%s' detected! (model=%s, score=%.3f, threshold=%.2f)",
                self._wake_word,
                matched_model_key,
                max_score,
                self._threshold,
            )

            # Emit wakeword.detected event
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_DETECTED,
                    payload={
                        "wake_word": self._wake_word,
                        "score": max_score,
                        "model_name": matched_model_key,
                        "timestamp": now,
                        "result": result,
                    },
                    source=EVENT_SOURCE_WAKEWORD,
                )

            # Automatically trigger conversation if enabled
            should_trigger = (
                trigger_conversation if trigger_conversation is not None else self._auto_listen
            )
            if should_trigger:
                self._trigger_voice_engine()

            return result

        return WakeWordResult(
            detected=False,
            matched_phrase=None,
            normalized_text="",
            timestamp=now,
            score=max_score,
            model_name=matched_model_key,
        )

    def queue_audio(self, chunk: Union[bytes, np.ndarray]) -> None:
        """Enqueue an audio chunk to be processed by the background thread."""
        if isinstance(chunk, (bytes, bytearray)):
            audio_data = np.frombuffer(chunk, dtype=np.int16)
        elif isinstance(chunk, np.ndarray):
            audio_data = chunk.astype(np.int16)
        else:
            raise AudioStreamError(f"Unsupported audio data type: {type(chunk)}")

        try:
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            self._logger.warning("Wake word audio queue full, dropping frame.")

    def _trigger_voice_engine(self) -> None:
        """Trigger VoiceConversationEngine.listen_once() in a controlled manner."""
        engine = self.voice_engine
        if engine is None:
            self._logger.warning(
                "Wake word detected but no VoiceConversationEngine is registered/available."
            )
            return

        self._logger.info("Executing VoiceConversationEngine.listen_once() automatically...")
        self.pause()
        try:
            engine.listen_once()
        except Exception as exc:
            err_msg = f"VoiceConversationEngine.listen_once() failed: {exc}"
            self._logger.error(err_msg, exc_info=True)
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_FAILED,
                    payload={"error": err_msg, "stage": "conversation_trigger", "timestamp": time.time()},
                    source=EVENT_SOURCE_WAKEWORD,
                )
        finally:
            self.resume()

    # --------------------------------------------------------------------------
    # Background Listening Thread Loop
    # --------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background listening loop acquiring and evaluating audio chunks."""
        chunk_duration = CHUNK_SIZE_SAMPLES / SAMPLE_RATE

        while not self._stop_event.is_set():
            if self._is_paused:
                time.sleep(0.05)
                continue

            chunk: Optional[Union[bytes, np.ndarray]] = None

            # 1. Try fetching from custom audio_source_callback
            if self._audio_source_callback is not None:
                try:
                    chunk = self._audio_source_callback()
                except Exception as exc:
                    self._logger.warning("Error in audio_source_callback: %s", exc)
                    if self._event_bus is not None:
                        self._event_bus.publish(
                            event=EVENT_WAKEWORD_FAILED,
                            payload={"error": str(exc), "stage": "audio_source", "timestamp": time.time()},
                            source=EVENT_SOURCE_WAKEWORD,
                        )
                    time.sleep(chunk_duration)
                    continue

            # 2. Try fetching from internal queue (fed externally)
            if chunk is None:
                try:
                    chunk = self._audio_queue.get(timeout=chunk_duration)
                except queue.Empty:
                    pass

            # 3. If no audio chunk available, sleep for chunk duration and continue
            if chunk is None:
                time.sleep(chunk_duration)
                continue

            # 4. Feed audio to model
            try:
                self.feed_audio(chunk)
            except Exception as exc:
                self._logger.error("Error processing audio chunk: %s", exc, exc_info=True)


# Default Singleton Instance Export (lazy loaded)
wake_word_engine: Final[WakeWordEngine] = WakeWordEngine(
    auto_register_in_container=False,
    lazy_load_model=True,
)

__all__ = [
    "CHUNK_SIZE_SAMPLES",
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_DETECTION_THRESHOLD",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "WakeWordEngine",
    "wake_word_engine",
]
