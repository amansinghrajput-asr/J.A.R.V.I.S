"""Voice Pipeline Subsystem for J.A.R.V.I.S.

Provides the central voice orchestration layer: coordinates audio input capture,
Speech-to-Text (STT), Wake Word detection, Command Router dispatch, Conversation
Memory context integration, Text-to-Speech (TTS) response synthesis, and system-wide
Event Bus lifecycle notifications within a thread-safe, async-ready state machine.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import string
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Coroutine, Final, Optional, TypeVar, Union

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.memory.manager import MemoryManager, memory_manager
from app.router.router import CommandRouter, command_router
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText, speech_to_text
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech, text_to_speech
from app.voice.models import (
    AudioProvider,
    AudioProviderError,
    InvalidStateError,
    PlaceholderAudioProvider,
    PipelineExecutionError,
    PipelineNotRunningError,
    VoicePipelineError,
    VoicePipelineResult,
    VoiceState,
)
from app.wakeword.detector import WakeWordDetector, wake_word_detector
from app.wakeword.models import WakeWordResult

T = TypeVar("T")

# Lifecycle Event Constants
EVENT_VOICE_STARTED: Final[str] = "voice.started"
EVENT_VOICE_STOPPED: Final[str] = "voice.stopped"
EVENT_VOICE_LISTENING: Final[str] = "voice.listening"
EVENT_VOICE_PROCESSING: Final[str] = "voice.processing"
EVENT_VOICE_RESPONDING: Final[str] = "voice.responding"
EVENT_VOICE_FAILED: Final[str] = "voice.failed"
EVENT_SOURCE_VOICE: Final[str] = "voice_pipeline"

# Allowed State Transitions
_VALID_TRANSITIONS: Final[dict[VoiceState, set[VoiceState]]] = {
    VoiceState.IDLE: {VoiceState.LISTENING, VoiceState.PROCESSING},
    VoiceState.LISTENING: {VoiceState.PROCESSING, VoiceState.IDLE},
    VoiceState.PROCESSING: {VoiceState.SPEAKING, VoiceState.LISTENING, VoiceState.IDLE},
    VoiceState.SPEAKING: {VoiceState.LISTENING, VoiceState.IDLE},
}


class VoicePipeline:
    """Thread-safe, async-ready Voice Pipeline Orchestrator for J.A.R.V.I.S.

    Coordinates the complete voice interaction lifecycle:
    1. Input Acquisition: Captures audio from an AudioProvider.
    2. Speech-to-Text: Converts speech audio to transcribed text via SpeechToText.
    3. Wake Word Detection: Verifies activation triggers via WakeWordDetector.
    4. Memory Storage: Records user utterances in MemoryManager.
    5. Command Routing: Forwards user intent to the CommandRouter.
    6. Speech Synthesis: Converts assistant response to speech via TextToSpeech.
    7. Memory Storage: Records assistant response in MemoryManager.
    8. Event Publishing: Emits structured lifecycle events across the EventBus.
    """

    def __init__(
        self,
        audio_provider: Optional[AudioProvider] = None,
        stt: Optional[SpeechToText] = None,
        wake_word_detector_instance: Optional[WakeWordDetector] = None,
        router: Optional[CommandRouter] = None,
        tts: Optional[TextToSpeech] = None,
        memory: Optional[MemoryManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        require_wake_word: bool = True,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the VoicePipeline subsystem.

        Args:
            audio_provider: Optional AudioProvider instance. Defaults to PlaceholderAudioProvider.
            stt: Optional SpeechToText instance. Resolved from container or default.
            wake_word_detector_instance: Optional WakeWordDetector. Resolved from container or default.
            router: Optional CommandRouter instance. Resolved from container or default.
            tts: Optional TextToSpeech instance. Resolved from container or default.
            memory: Optional MemoryManager instance. Resolved from container or default.
            config: Optional Settings instance. Resolved from container or global settings.
            logger: Optional Logger instance. Defaults to 'VOICE_PIPELINE' logger.
            container_instance: Optional ServiceContainer. Defaults to global container.
            event_bus_instance: Optional EventBus. Defaults to global event bus.
            require_wake_word: Whether wake word is mandatory to trigger command routing.
            auto_register_in_container: If True, registers this pipeline in the Service Container.
        """
        self._lock = threading.RLock()
        self._state: VoiceState = VoiceState.IDLE
        self._is_running: bool = False
        self._require_wake_word: bool = bool(require_wake_word)

        # 1. Dependency resolution: Service Container
        self._container = (
            container_instance if container_instance is not None else container
        )

        # 2. Dependency resolution: Logger
        self._logger = (
            logger if logger is not None else get_logger("VOICE_PIPELINE")
        )

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

        # 5. Dependency resolution: Audio Provider
        if audio_provider is not None:
            self._audio_provider = audio_provider
        elif self._container.exists("audio_provider"):
            self._audio_provider = self._container.resolve("audio_provider")
        elif self._container.exists("microphone"):
            self._audio_provider = self._container.resolve("microphone")
        else:
            self._audio_provider = PlaceholderAudioProvider()

        # 6. Dependency resolution: Speech-to-Text (STT)
        if stt is not None:
            self._stt = stt
        elif self._container.exists("stt"):
            self._stt = self._container.resolve("stt")
        elif self._container.exists("speech_to_text"):
            self._stt = self._container.resolve("speech_to_text")
        else:
            self._stt = speech_to_text

        # 7. Dependency resolution: Wake Word Detector
        if wake_word_detector_instance is not None:
            self._wake_word_detector = wake_word_detector_instance
        elif self._container.exists("wake_word_detector"):
            self._wake_word_detector = self._container.resolve("wake_word_detector")
        elif self._container.exists("wakeword"):
            self._wake_word_detector = self._container.resolve("wakeword")
        else:
            self._wake_word_detector = wake_word_detector

        # 8. Dependency resolution: Command Router
        if router is not None:
            self._router = router
        elif self._container.exists("command_router"):
            self._router = self._container.resolve("command_router")
        elif self._container.exists("router"):
            self._router = self._container.resolve("router")
        else:
            self._router = command_router

        # 9. Dependency resolution: Text-to-Speech (TTS)
        if tts is not None:
            self._tts = tts
        elif self._container.exists("tts"):
            self._tts = self._container.resolve("tts")
        elif self._container.exists("text_to_speech"):
            self._tts = self._container.resolve("text_to_speech")
        else:
            self._tts = text_to_speech

        # 10. Dependency resolution: Memory Manager
        if memory is not None:
            self._memory = memory
        elif self._container.exists("memory_manager"):
            self._memory = self._container.resolve("memory_manager")
        elif self._container.exists("memory"):
            self._memory = self._container.resolve("memory")
        else:
            self._memory = memory_manager

        # 11. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("voice_pipeline", self, allow_override=True)
                self._container.register_singleton("voice", self, allow_override=True)
                self._logger.debug("Registered 'voice_pipeline' singleton in container.")
            except Exception as exc:
                self._logger.warning(f"Could not register VoicePipeline in container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        """Currently active lifecycle state."""
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Flag indicating whether the pipeline is currently started."""
        with self._lock:
            return self._is_running

    @property
    def require_wake_word(self) -> bool:
        """Flag indicating whether wake word detection is mandatory."""
        with self._lock:
            return self._require_wake_word

    @require_wake_word.setter
    def require_wake_word(self, value: bool) -> None:
        """Set whether wake word detection is mandatory."""
        with self._lock:
            self._require_wake_word = bool(value)

    @property
    def audio_provider(self) -> AudioProvider:
        """The active AudioProvider instance."""
        return self._audio_provider

    @property
    def stt(self) -> SpeechToText:
        """The active SpeechToText instance."""
        return self._stt

    @property
    def wake_word_detector(self) -> WakeWordDetector:
        """The active WakeWordDetector instance."""
        return self._wake_word_detector

    @property
    def router(self) -> CommandRouter:
        """The active CommandRouter instance."""
        return self._router

    @property
    def tts(self) -> TextToSpeech:
        """The active TextToSpeech instance."""
        return self._tts

    @property
    def memory(self) -> MemoryManager:
        """The active MemoryManager instance."""
        return self._memory

    @property
    def event_bus(self) -> EventBus:
        """The active EventBus instance."""
        return self._event_bus

    @property
    def container(self) -> ServiceContainer:
        """The active ServiceContainer instance."""
        return self._container

    @property
    def logger(self) -> logging.Logger:
        """The active Logger instance."""
        return self._logger

    # --------------------------------------------------------------------------
    # State Machine & Event Dispatching
    # --------------------------------------------------------------------------

    def _transition_to(
        self,
        new_state: VoiceState,
        *,
        payload: Optional[dict[str, Any]] = None,
        publish_event: bool = True,
    ) -> None:
        """Transition pipeline state machine with validation and event dispatch.

        Args:
            new_state: Target VoiceState.
            payload: Optional contextual payload to include in lifecycle event.
            publish_event: Whether to publish corresponding event to EventBus.

        Raises:
            InvalidStateError: If transition from current state to new_state is disallowed.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return

            valid_targets = _VALID_TRANSITIONS.get(old_state, set())
            if new_state not in valid_targets:
                raise InvalidStateError(
                    f"Invalid voice pipeline state transition from '{old_state.value}' to '{new_state.value}'."
                )

            self._state = new_state
            self._logger.debug(
                f"Voice pipeline state changed: {old_state.value} -> {new_state.value}"
            )

        if not publish_event or self._event_bus is None:
            return

        event_name = self._state_to_event(new_state)
        if event_name:
            event_payload = {
                "previous_state": old_state.value,
                "current_state": new_state.value,
                "timestamp": time.time(),
            }
            if payload:
                event_payload.update(payload)

            self._event_bus.publish(
                event=event_name,
                payload=event_payload,
                source=EVENT_SOURCE_VOICE,
            )

    async def _transition_to_async(
        self,
        new_state: VoiceState,
        *,
        payload: Optional[dict[str, Any]] = None,
        publish_event: bool = True,
    ) -> None:
        """Asynchronously transition pipeline state machine and dispatch event."""
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return

            valid_targets = _VALID_TRANSITIONS.get(old_state, set())
            if new_state not in valid_targets:
                raise InvalidStateError(
                    f"Invalid voice pipeline state transition from '{old_state.value}' to '{new_state.value}'."
                )

            self._state = new_state
            self._logger.debug(
                f"Voice pipeline state changed (async): {old_state.value} -> {new_state.value}"
            )

        if not publish_event or self._event_bus is None:
            return

        event_name = self._state_to_event(new_state)
        if event_name:
            event_payload = {
                "previous_state": old_state.value,
                "current_state": new_state.value,
                "timestamp": time.time(),
            }
            if payload:
                event_payload.update(payload)

            await self._event_bus.publish_async(
                event=event_name,
                payload=event_payload,
                source=EVENT_SOURCE_VOICE,
            )

    @staticmethod
    def _state_to_event(state: VoiceState) -> Optional[str]:
        """Map VoiceState to its primary lifecycle event topic."""
        if state == VoiceState.LISTENING:
            return EVENT_VOICE_LISTENING
        elif state == VoiceState.PROCESSING:
            return EVENT_VOICE_PROCESSING
        elif state == VoiceState.SPEAKING:
            return EVENT_VOICE_RESPONDING
        return None

    def _publish_failure(
        self,
        error: Exception,
        *,
        session_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Publish a voice.failed event synchronously."""
        if self._event_bus is None:
            return
        payload: dict[str, Any] = {
            "session_id": session_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "state": self._state.value,
            "timestamp": time.time(),
        }
        if context:
            payload.update(context)

        self._event_bus.publish(
            event=EVENT_VOICE_FAILED,
            payload=payload,
            source=EVENT_SOURCE_VOICE,
        )

    async def _publish_failure_async(
        self,
        error: Exception,
        *,
        session_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Publish a voice.failed event asynchronously."""
        if self._event_bus is None:
            return
        payload: dict[str, Any] = {
            "session_id": session_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "state": self._state.value,
            "timestamp": time.time(),
        }
        if context:
            payload.update(context)

        await self._event_bus.publish_async(
            event=EVENT_VOICE_FAILED,
            payload=payload,
            source=EVENT_SOURCE_VOICE,
        )

    # --------------------------------------------------------------------------
    # Lifecycle Control (Start / Stop)
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Start the voice pipeline and begin listening for audio input.

        Transitions state to LISTENING and emits 'voice.started' and 'voice.listening'.
        """
        with self._lock:
            if self._is_running:
                return

            self._is_running = True
            try:
                self._audio_provider.start()
            except Exception as exc:
                self._is_running = False
                raise AudioProviderError(f"Failed to start audio provider: {exc}") from exc

            # Publish voice.started event
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_VOICE_STARTED,
                    payload={
                        "timestamp": time.time(),
                        "require_wake_word": self._require_wake_word,
                    },
                    source=EVENT_SOURCE_VOICE,
                )

            self._transition_to(VoiceState.LISTENING)
            self._logger.info("Voice Pipeline started successfully.")

    async def start_async(self) -> None:
        """Asynchronously start the voice pipeline and begin listening."""
        with self._lock:
            if self._is_running:
                return

            self._is_running = True
            try:
                self._audio_provider.start()
            except Exception as exc:
                self._is_running = False
                raise AudioProviderError(f"Failed to start audio provider: {exc}") from exc

        if self._event_bus is not None:
            await self._event_bus.publish_async(
                event=EVENT_VOICE_STARTED,
                payload={
                    "timestamp": time.time(),
                    "require_wake_word": self._require_wake_word,
                },
                source=EVENT_SOURCE_VOICE,
            )

        await self._transition_to_async(VoiceState.LISTENING)
        self._logger.info("Voice Pipeline started (async) successfully.")

    def stop(self) -> None:
        """Stop the voice pipeline and release audio capture resources.

        Transitions state to IDLE and emits 'voice.stopped'.
        """
        with self._lock:
            if not self._is_running and self._state == VoiceState.IDLE:
                return

            self._is_running = False
            try:
                self._audio_provider.stop()
            except Exception as exc:
                self._logger.warning(f"Error stopping audio provider: {exc}")

            self._state = VoiceState.IDLE

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_VOICE_STOPPED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_VOICE,
            )

        self._logger.info("Voice Pipeline stopped.")

    async def stop_async(self) -> None:
        """Asynchronously stop the voice pipeline and release resources."""
        with self._lock:
            if not self._is_running and self._state == VoiceState.IDLE:
                return

            self._is_running = False
            try:
                self._audio_provider.stop()
            except Exception as exc:
                self._logger.warning(f"Error stopping audio provider: {exc}")

            self._state = VoiceState.IDLE

        if self._event_bus is not None:
            await self._event_bus.publish_async(
                event=EVENT_VOICE_STOPPED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_VOICE,
            )

        self._logger.info("Voice Pipeline stopped (async).")

    # --------------------------------------------------------------------------
    # Command Extraction & Response Formatting Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _extract_command(raw_text: str, matched_phrase: Optional[str]) -> str:
        """Extract the command payload by stripping the matched wake phrase.

        Args:
            raw_text: Transcribed full speech text.
            matched_phrase: The exact wake phrase that was detected, if any.

        Returns:
            Clean command string stripped of wake phrase and extraneous punctuation.
        """
        if not raw_text or not raw_text.strip():
            return ""

        cleaned = raw_text.strip()
        if not matched_phrase:
            return cleaned

        # Case-insensitive removal of matched wake phrase at start or within text
        pattern = re.compile(rf"\b{re.escape(matched_phrase)}\b", re.IGNORECASE)
        command_candidate = pattern.sub("", cleaned, count=1).strip()

        # Remove leading punctuation (e.g. commas, colons after 'Hey Jarvis, ...')
        command_candidate = command_candidate.lstrip(string.punctuation + " ").strip()
        return command_candidate

    @staticmethod
    def _format_response_text(raw_response: Any) -> str:
        """Format an execution response from CommandRouter into speakable text.

        Args:
            raw_response: Return value from command execution.

        Returns:
            Non-empty string suitable for TTS synthesis.
        """
        if raw_response is None:
            return ""

        if isinstance(raw_response, str):
            return raw_response.strip()

        if isinstance(raw_response, dict):
            for key in ("response", "message", "text", "output", "result"):
                if key in raw_response and raw_response[key]:
                    return str(raw_response[key]).strip()
            return str(raw_response)

        for attr in ("response", "message", "text", "output"):
            if hasattr(raw_response, attr):
                val = getattr(raw_response, attr)
                if val:
                    return str(val).strip()

        return str(raw_response).strip()

    # --------------------------------------------------------------------------
    # Synchronous Execution Engine
    # --------------------------------------------------------------------------

    def process_audio(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        *,
        require_wake_word: Optional[bool] = None,
    ) -> VoicePipelineResult:
        """Execute a complete voice processing cycle synchronously.

        Workflow:
        1. Acquire audio file path from argument or active audio provider.
        2. Set state to PROCESSING (emits voice.processing).
        3. Transcribe audio to text via SpeechToText.
        4. Detect wake word via WakeWordDetector (if required).
        5. Record user utterance in MemoryManager.
        6. Route command via CommandRouter.
        7. Set state to SPEAKING (emits voice.responding).
        8. Synthesize assistant response via TextToSpeech.
        9. Record assistant response in MemoryManager.
        10. Return to LISTENING (if running) or IDLE.

        Args:
            audio_path: Explicit audio file path to process. If None, captures from audio_provider.
            require_wake_word: Optional override for wake word requirement.

        Returns:
            VoicePipelineResult encapsulating all stage outcomes and metadata.
        """
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())
        req_wake = (
            self._require_wake_word
            if require_wake_word is None
            else bool(require_wake_word)
        )

        # 1. Acquire audio input
        target_audio: Optional[Path] = None
        if audio_path is not None:
            target_audio = Path(audio_path).resolve()
        else:
            try:
                target_audio = self._audio_provider.capture_audio()
            except Exception as exc:
                self._publish_failure(exc, session_id=session_id)
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=None,
                    success=False,
                    error=f"Audio capture failed: {exc}",
                    duration=time.perf_counter() - start_time,
                )

        if target_audio is None:
            self._logger.debug("No audio available to process.")
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=None,
                success=True,
                duration=time.perf_counter() - start_time,
            )

        # 2. Transition state to PROCESSING
        try:
            self._transition_to(
                VoiceState.PROCESSING,
                payload={"session_id": session_id, "audio_path": str(target_audio)},
            )
        except InvalidStateError:
            # Force transition if necessary during manual step
            with self._lock:
                self._state = VoiceState.PROCESSING

        transcription_res: Optional[TranscriptionResult] = None
        wake_res: Optional[WakeWordResult] = None
        command_text: Optional[str] = None
        command_result: Optional[Any] = None
        speech_res: Optional[SpeechResult] = None

        try:
            # 3. Speech-to-Text Transcription
            self._logger.debug(f"Transcribing audio: {target_audio}")
            transcription_res = self._stt.transcribe(target_audio)
            raw_text = transcription_res.text if transcription_res else ""

            if not raw_text or not raw_text.strip():
                self._logger.info("Speech-to-text returned empty transcription.")
                self._recover_state_sync()
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=target_audio,
                    transcription=transcription_res,
                    success=True,
                    duration=time.perf_counter() - start_time,
                )

            # 4. Wake Word Detection
            wake_res = self._wake_word_detector.detect(raw_text)
            if req_wake and not wake_res.detected:
                self._logger.info(
                    f"Wake word not detected in transcript '{raw_text}'. Command ignored."
                )
                self._recover_state_sync()
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=target_audio,
                    transcription=transcription_res,
                    wake_word_result=wake_res,
                    success=True,
                    duration=time.perf_counter() - start_time,
                )

            # Extract command text
            command_text = self._extract_command(raw_text, wake_res.matched_phrase)
            if not command_text:
                # User spoke wake word only (e.g. "Hey Jarvis")
                command_text = "hello"

            # 5. Record User Input in Memory
            if self._memory is not None:
                try:
                    self._memory.add(
                        content=command_text,
                        role="user",
                        source="voice",
                        metadata={"session_id": session_id, "raw_transcription": raw_text},
                    )
                except Exception as exc:
                    self._logger.warning(f"Failed to record user memory: {exc}")

            # 6. Dispatch Command to Command Router
            self._logger.info(f"Forwarding voice command to router: '{command_text}'")
            command_result = self._router.route(
                command=command_text,
                source="voice",
                context={"session_id": session_id},
            )

            # Format speakable response
            response_text = self._format_response_text(command_result)

            # 7. Transition to SPEAKING and Synthesize Response
            if response_text:
                self._transition_to(
                    VoiceState.SPEAKING,
                    payload={"session_id": session_id, "response_text": response_text},
                )

                self._logger.debug(f"Synthesizing speech for response: '{response_text[:40]}...'")
                speech_res = self._tts.synthesize(response_text)

                # 8. Record Assistant Response in Memory
                if self._memory is not None:
                    try:
                        self._memory.add(
                            content=response_text,
                            role="assistant",
                            source="voice",
                            metadata={
                                "session_id": session_id,
                                "speech_audio": str(speech_res.audio_path) if speech_res else None,
                            },
                        )
                    except Exception as exc:
                        self._logger.warning(f"Failed to record assistant memory: {exc}")

            # 9. Return to LISTENING or IDLE
            self._recover_state_sync()

            total_duration = time.perf_counter() - start_time
            self._logger.info(f"Voice cycle {session_id[:8]} completed in {total_duration:.2f}s.")

            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=target_audio,
                transcription=transcription_res,
                wake_word_result=wake_res,
                command_text=command_text,
                command_result=command_result,
                speech_result=speech_res,
                success=True,
                duration=total_duration,
            )

        except Exception as exc:
            self._logger.error(f"Voice processing cycle failed: {exc}", exc_info=True)
            self._publish_failure(
                exc,
                session_id=session_id,
                context={"audio_path": str(target_audio) if target_audio else None},
            )
            self._recover_state_sync()
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=target_audio,
                transcription=transcription_res,
                wake_word_result=wake_res,
                command_text=command_text,
                command_result=command_result,
                speech_result=speech_res,
                success=False,
                error=str(exc),
                duration=time.perf_counter() - start_time,
            )

    def process_command(self, command_text: str) -> VoicePipelineResult:
        """Directly process a textual command through the voice pipeline.

        Bypasses STT and wake word detection while preserving Memory integration,
        Command Router execution, Text-to-Speech synthesis, and lifecycle events.

        Args:
            command_text: Textual user command string.

        Returns:
            VoicePipelineResult encapsulating execution outcomes.
        """
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())

        if not command_text or not command_text.strip():
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                success=False,
                error="Command text cannot be empty.",
                duration=0.0,
            )

        clean_cmd = command_text.strip()
        self._transition_to(
            VoiceState.PROCESSING,
            payload={"session_id": session_id, "command": clean_cmd},
        )

        try:
            # 1. Record User Memory
            if self._memory is not None:
                try:
                    self._memory.add(
                        content=clean_cmd,
                        role="user",
                        source="voice",
                        metadata={"session_id": session_id},
                    )
                except Exception as exc:
                    self._logger.warning(f"Failed to record user memory: {exc}")

            # 2. Forward to Command Router
            command_result = self._router.route(
                command=clean_cmd,
                source="voice",
                context={"session_id": session_id},
            )

            # 3. Format response
            response_text = self._format_response_text(command_result)
            speech_res: Optional[SpeechResult] = None

            # 4. Speak response
            if response_text:
                self._transition_to(
                    VoiceState.SPEAKING,
                    payload={"session_id": session_id, "response_text": response_text},
                )
                speech_res = self._tts.synthesize(response_text)

                if self._memory is not None:
                    try:
                        self._memory.add(
                            content=response_text,
                            role="assistant",
                            source="voice",
                            metadata={"session_id": session_id},
                        )
                    except Exception as exc:
                        self._logger.warning(f"Failed to record assistant memory: {exc}")

            self._recover_state_sync()
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                command_text=clean_cmd,
                command_result=command_result,
                speech_result=speech_res,
                success=True,
                duration=time.perf_counter() - start_time,
            )

        except Exception as exc:
            self._logger.error(f"Voice process_command failed: {exc}", exc_info=True)
            self._publish_failure(exc, session_id=session_id)
            self._recover_state_sync()
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                command_text=clean_cmd,
                success=False,
                error=str(exc),
                duration=time.perf_counter() - start_time,
            )

    def _recover_state_sync(self) -> None:
        """Reset state back to LISTENING if running, else IDLE."""
        with self._lock:
            target = VoiceState.LISTENING if self._is_running else VoiceState.IDLE
            if self._state != target:
                self._state = target
                if target == VoiceState.LISTENING and self._event_bus is not None:
                    self._event_bus.publish(
                        event=EVENT_VOICE_LISTENING,
                        payload={"timestamp": time.time()},
                        source=EVENT_SOURCE_VOICE,
                    )

    # --------------------------------------------------------------------------
    # Asynchronous Execution Engine
    # --------------------------------------------------------------------------

    async def process_audio_async(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        *,
        require_wake_word: Optional[bool] = None,
    ) -> VoicePipelineResult:
        """Asynchronously execute a complete voice processing cycle.

        Args:
            audio_path: Explicit audio file path to process.
            require_wake_word: Optional override for wake word requirement.

        Returns:
            VoicePipelineResult encapsulating all stage outcomes and metadata.
        """
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())
        req_wake = (
            self._require_wake_word
            if require_wake_word is None
            else bool(require_wake_word)
        )

        # 1. Acquire audio input
        target_audio: Optional[Path] = None
        if audio_path is not None:
            target_audio = Path(audio_path).resolve()
        else:
            try:
                target_audio = await self._audio_provider.capture_audio_async()
            except Exception as exc:
                await self._publish_failure_async(exc, session_id=session_id)
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=None,
                    success=False,
                    error=f"Audio capture failed: {exc}",
                    duration=time.perf_counter() - start_time,
                )

        if target_audio is None:
            self._logger.debug("No audio available to process (async).")
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=None,
                success=True,
                duration=time.perf_counter() - start_time,
            )

        # 2. Transition to PROCESSING
        try:
            await self._transition_to_async(
                VoiceState.PROCESSING,
                payload={"session_id": session_id, "audio_path": str(target_audio)},
            )
        except InvalidStateError:
            with self._lock:
                self._state = VoiceState.PROCESSING

        transcription_res: Optional[TranscriptionResult] = None
        wake_res: Optional[WakeWordResult] = None
        command_text: Optional[str] = None
        command_result: Optional[Any] = None
        speech_res: Optional[SpeechResult] = None

        try:
            # 3. Speech-to-Text Transcription
            self._logger.debug(f"Transcribing audio (async): {target_audio}")
            transcription_res = await self._stt.transcribe_async(target_audio)
            raw_text = transcription_res.text if transcription_res else ""

            if not raw_text or not raw_text.strip():
                self._logger.info("Speech-to-text returned empty transcription (async).")
                await self._recover_state_async()
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=target_audio,
                    transcription=transcription_res,
                    success=True,
                    duration=time.perf_counter() - start_time,
                )

            # 4. Wake Word Detection
            wake_res = await self._wake_word_detector.detect_async(raw_text)
            if req_wake and not wake_res.detected:
                self._logger.info(
                    f"Wake word not detected in transcript '{raw_text}'. Command ignored (async)."
                )
                await self._recover_state_async()
                return VoicePipelineResult(
                    session_id=session_id,
                    state=self._state,
                    audio_input=target_audio,
                    transcription=transcription_res,
                    wake_word_result=wake_res,
                    success=True,
                    duration=time.perf_counter() - start_time,
                )

            command_text = self._extract_command(raw_text, wake_res.matched_phrase)
            if not command_text:
                command_text = "hello"

            # 5. Record User Input in Memory
            if self._memory is not None:
                try:
                    self._memory.add(
                        content=command_text,
                        role="user",
                        source="voice",
                        metadata={"session_id": session_id, "raw_transcription": raw_text},
                    )
                except Exception as exc:
                    self._logger.warning(f"Failed to record user memory: {exc}")

            # 6. Dispatch Command to Command Router
            self._logger.info(f"Forwarding voice command to router (async): '{command_text}'")
            route_coro = self._router.route_async(
                command=command_text,
                source="voice",
                context={"session_id": session_id},
            )
            command_result = await route_coro

            response_text = self._format_response_text(command_result)

            # 7. Transition to SPEAKING and Synthesize Response
            if response_text:
                await self._transition_to_async(
                    VoiceState.SPEAKING,
                    payload={"session_id": session_id, "response_text": response_text},
                )

                self._logger.debug(f"Synthesizing speech (async) for response: '{response_text[:40]}...'")
                speech_res = await self._tts.synthesize_async(response_text)

                # 8. Record Assistant Response in Memory
                if self._memory is not None:
                    try:
                        self._memory.add(
                            content=response_text,
                            role="assistant",
                            source="voice",
                            metadata={
                                "session_id": session_id,
                                "speech_audio": str(speech_res.audio_path) if speech_res else None,
                            },
                        )
                    except Exception as exc:
                        self._logger.warning(f"Failed to record assistant memory: {exc}")

            # 9. Return to LISTENING or IDLE
            await self._recover_state_async()

            total_duration = time.perf_counter() - start_time
            self._logger.info(f"Voice cycle {session_id[:8]} completed in {total_duration:.2f}s (async).")

            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=target_audio,
                transcription=transcription_res,
                wake_word_result=wake_res,
                command_text=command_text,
                command_result=command_result,
                speech_result=speech_res,
                success=True,
                duration=total_duration,
            )

        except Exception as exc:
            self._logger.error(f"Voice processing cycle failed (async): {exc}", exc_info=True)
            await self._publish_failure_async(
                exc,
                session_id=session_id,
                context={"audio_path": str(target_audio) if target_audio else None},
            )
            await self._recover_state_async()
            return VoicePipelineResult(
                session_id=session_id,
                state=self._state,
                audio_input=target_audio,
                transcription=transcription_res,
                wake_word_result=wake_res,
                command_text=command_text,
                command_result=command_result,
                speech_result=speech_res,
                success=False,
                error=str(exc),
                duration=time.perf_counter() - start_time,
            )

    async def _recover_state_async(self) -> None:
        """Asynchronously reset state back to LISTENING if running, else IDLE."""
        with self._lock:
            target = VoiceState.LISTENING if self._is_running else VoiceState.IDLE
            if self._state == target:
                return
            self._state = target

        if target == VoiceState.LISTENING and self._event_bus is not None:
            await self._event_bus.publish_async(
                event=EVENT_VOICE_LISTENING,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_VOICE,
            )


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
voice_pipeline: Final[VoicePipeline] = VoicePipeline()

__all__ = [
    "EVENT_SOURCE_VOICE",
    "EVENT_VOICE_FAILED",
    "EVENT_VOICE_LISTENING",
    "EVENT_VOICE_PROCESSING",
    "EVENT_VOICE_RESPONDING",
    "EVENT_VOICE_STARTED",
    "EVENT_VOICE_STOPPED",
    "VoicePipeline",
    "voice_pipeline",
]
