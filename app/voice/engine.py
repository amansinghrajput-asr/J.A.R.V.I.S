"""Voice Conversation Engine for J.A.R.V.I.S.

Provides the complete voice interaction loop integrating existing components:
1. MicrophoneRecorder: Captures microphone input to standard WAV format.
2. SpeechToText: Transcribes speech audio to text.
3. CommandRouter: Routes transcribed command intent to registered skills.
4. TextToSpeech: Synthesizes spoken audio response from skill execution output.

Flow:
    record audio -> transcribe -> route through CommandRouter -> receive response -> speak response -> return result

No STT or TTS logic is duplicated; all speech processing delegates directly to
the existing speech-to-text and text-to-speech subsystems.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Final, Optional, Union
import uuid

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.router.router import CommandRouter, command_router
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText, speech_to_text
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech, text_to_speech
from app.voice.microphone import DEFAULT_RECORD_DURATION, MicrophoneRecorder
from app.voice.models import (
    AudioProviderError,
    PipelineExecutionError,
    VoiceConversationResult,
    VoicePipelineError,
)
from app.voice.player import AudioPlayer, audio_player

# Lifecycle Event Constants
EVENT_ENGINE_STARTED: Final[str] = "voice_engine.started"
EVENT_ENGINE_STOPPED: Final[str] = "voice_engine.stopped"
EVENT_ENGINE_RECORDING: Final[str] = "voice_engine.recording"
EVENT_ENGINE_TRANSCRIBING: Final[str] = "voice_engine.transcribing"
EVENT_ENGINE_ROUTING: Final[str] = "voice_engine.routing"
EVENT_ENGINE_SPEAKING: Final[str] = "voice_engine.speaking"
EVENT_ENGINE_COMPLETED: Final[str] = "voice_engine.completed"
EVENT_ENGINE_FAILED: Final[str] = "voice_engine.failed"
EVENT_SOURCE_VOICE_ENGINE: Final[str] = "voice_engine"


class VoiceConversationEngine:
    """Production Voice Conversation Engine orchestrating voice interactions.

    Integrates existing components without duplication:
    - MicrophoneRecorder (`app.voice.microphone`)
    - SpeechToText (`app.stt`)
    - CommandRouter (`app.router`)
    - TextToSpeech (`app.tts`)
    - AudioPlayer (`app.voice.player`)

    Executes the standard voice interaction cycle:
    record audio -> transcribe -> route -> receive response -> speak response -> return result.
    """

    def __init__(
        self,
        recorder: Optional[MicrophoneRecorder] = None,
        stt: Optional[SpeechToText] = None,
        router: Optional[CommandRouter] = None,
        tts: Optional[TextToSpeech] = None,
        player: Optional[AudioPlayer] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        state_manager: Optional[Any] = None,
        *,
        record_duration: Optional[float] = None,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the VoiceConversationEngine.

        Args:
            recorder: Optional MicrophoneRecorder instance. Resolved from container if None.
            stt: Optional SpeechToText instance. Resolved from container or default if None.
            router: Optional CommandRouter instance. Resolved from container or default if None.
            tts: Optional TextToSpeech instance. Resolved from container or default if None.
            player: Optional AudioPlayer instance. Resolved from container or default if None.
            config: Optional Settings instance. Resolved from container or global settings.
            logger: Optional Logger instance. Defaults to 'VOICE_ENGINE' logger.
            container_instance: Optional ServiceContainer. Defaults to global container.
            event_bus_instance: Optional EventBus. Defaults to container or global event bus.
            record_duration: Default recording duration in seconds. Defaults to 3.0s.
            auto_register_in_container: If True, registers in ServiceContainer.
        """
        self._lock = threading.RLock()
        self._is_running: bool = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="VoiceEngine")

        # 1. Dependency Resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency Resolution: Logger
        self._logger = logger if logger is not None else get_logger("VOICE_ENGINE")

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

        # 5. Dependency Resolution: Microphone Recorder
        if recorder is not None:
            self._recorder = recorder
        elif self._container.exists("microphone"):
            self._recorder = self._container.resolve("microphone")
        elif self._container.exists("microphone_recorder"):
            self._recorder = self._container.resolve("microphone_recorder")
        else:
            self._recorder = MicrophoneRecorder(
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
                auto_register_in_container=False,
            )

        # 6. Dependency Resolution: Speech-to-Text (STT)
        if stt is not None:
            self._stt = stt
        elif self._container.exists("stt"):
            self._stt = self._container.resolve("stt")
        elif self._container.exists("speech_to_text"):
            self._stt = self._container.resolve("speech_to_text")
        else:
            self._stt = speech_to_text

        # 7. Dependency Resolution: Command Router
        if router is not None:
            self._router = router
        elif self._container.exists("command_router"):
            self._router = self._container.resolve("command_router")
        elif self._container.exists("router"):
            self._router = self._container.resolve("router")
        else:
            self._router = command_router

        # 8. Dependency Resolution: Text-to-Speech (TTS)
        if tts is not None:
            self._tts = tts
        elif self._container.exists("tts"):
            self._tts = self._container.resolve("tts")
        elif self._container.exists("text_to_speech"):
            self._tts = self._container.resolve("text_to_speech")
        else:
            self._tts = text_to_speech

        # 9. Dependency Resolution: Audio Player (Speaker Playback)
        if player is not None:
            self._player: Optional[AudioPlayer] = player
        elif self._container.exists("audio_player"):
            self._player = self._container.resolve("audio_player")
        else:
            self._player = audio_player

        # Recording Duration Setting
        if record_duration is not None:
            self._record_duration = float(record_duration)
        elif hasattr(self._recorder, "record_duration"):
            self._record_duration = float(self._recorder.record_duration)
        else:
            self._record_duration = DEFAULT_RECORD_DURATION

        # 10. State Manager Dependency & Amplitude Telemetry
        self._state_manager = state_manager
        self._bind_amplitude_telemetry()
        self._bind_player_amplitude_telemetry()

        # 11. Self-Registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("voice_engine", self, allow_override=True)
                self._container.register_singleton("voice_conversation_engine", self, allow_override=True)
                self._logger.debug("Registered 'voice_engine' singleton in Service Container.")
            except Exception as exc:
                self._logger.warning("Could not register VoiceConversationEngine in container: %s", exc)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def recorder(self) -> MicrophoneRecorder:
        """Return the active MicrophoneRecorder instance."""
        return self._recorder

    @property
    def stt(self) -> SpeechToText:
        """Return the active SpeechToText instance."""
        return self._stt

    @property
    def router(self) -> CommandRouter:
        """Return the active CommandRouter instance."""
        return self._router

    @property
    def tts(self) -> TextToSpeech:
        """Return the active TextToSpeech instance."""
        return self._tts

    @property
    def player(self) -> Optional[AudioPlayer]:
        """Return the active AudioPlayer instance."""
        return self._player

    @property
    def is_running(self) -> bool:
        """Return whether the engine is active."""
        with self._lock:
            return self._is_running

    @property
    def record_duration(self) -> float:
        """Return the default recording duration in seconds."""
        with self._lock:
            return self._record_duration

    @record_duration.setter
    def record_duration(self, value: float) -> None:
        """Update the default recording duration in seconds."""
        if value <= 0:
            raise ValueError(f"record_duration must be positive, got {value}")
        with self._lock:
            self._record_duration = float(value)

    @property
    def container(self) -> ServiceContainer:
        """Return the ServiceContainer instance."""
        return self._container

    @property
    def config(self) -> Settings:
        """Return the application Settings instance."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the Logger instance."""
        return self._logger

    @property
    def event_bus(self) -> EventBus:
        """Return the EventBus instance."""
        return self._event_bus

    # --------------------------------------------------------------------------
    # Telemetry Helpers
    # --------------------------------------------------------------------------

    def _get_state_manager(self) -> Optional[Any]:
        """Resolve AssistantStateManager safely via parameter or container."""
        if self._state_manager is not None:
            return self._state_manager
        if self._container is not None and self._container.exists("state_manager"):
            try:
                return self._container.resolve("state_manager")
            except Exception:
                return None
        return None

    def _bind_amplitude_telemetry(self) -> None:
        """Ensure recorder amplitude is forwarded to AssistantStateManager if available."""
        if not hasattr(self._recorder, "amplitude_callback"):
            return
        state_mgr = self._get_state_manager()
        if state_mgr is None or not hasattr(state_mgr, "update_mic_amplitude"):
            return

        existing_cb = getattr(self._recorder, "amplitude_callback", None)
        if existing_cb is not None and getattr(existing_cb, "_is_jarvis_state_bridge", False):
            return

        def _amplitude_forwarder(amp: float) -> None:
            try:
                state_mgr.update_mic_amplitude(amp)
            except Exception as exc:
                self._logger.debug("Error forwarding mic amplitude to state_manager: %s", exc)
            if existing_cb is not None and callable(existing_cb) and existing_cb != _amplitude_forwarder:
                try:
                    existing_cb(amp)
                except Exception as exc:
                    self._logger.debug("Error calling downstream amplitude callback: %s", exc)

        _amplitude_forwarder._is_jarvis_state_bridge = True  # type: ignore[attr-defined]
        try:
            self._recorder.amplitude_callback = _amplitude_forwarder
        except Exception as exc:
            self._logger.debug("Could not assign amplitude_callback on recorder: %s", exc)

    def _bind_player_amplitude_telemetry(self) -> None:
        """Ensure audio player amplitude is forwarded to AssistantStateManager if available."""
        if self._player is None or not hasattr(self._player, "amplitude_callback"):
            return
        state_mgr = self._get_state_manager()
        if state_mgr is None or not hasattr(state_mgr, "update_mic_amplitude"):
            return

        existing_cb = getattr(self._player, "amplitude_callback", None)
        if existing_cb is not None and getattr(existing_cb, "_is_jarvis_player_bridge", False):
            return

        def _player_amplitude_forwarder(amp: float) -> None:
            try:
                state_mgr.update_mic_amplitude(amp)
            except Exception as exc:
                self._logger.debug("Error forwarding player amplitude to state_manager: %s", exc)
            if existing_cb is not None and callable(existing_cb) and existing_cb != _player_amplitude_forwarder:
                try:
                    existing_cb(amp)
                except Exception as exc:
                    self._logger.debug("Error calling downstream player amplitude callback: %s", exc)

        _player_amplitude_forwarder._is_jarvis_player_bridge = True  # type: ignore[attr-defined]
        try:
            self._player.amplitude_callback = _player_amplitude_forwarder
        except Exception as exc:
            self._logger.debug("Could not assign amplitude_callback on player: %s", exc)

    def _reset_mic_amplitude(self) -> None:
        """Safely reset state manager mic amplitude to 0.0."""
        state_mgr = self._get_state_manager()
        if state_mgr is not None and hasattr(state_mgr, "update_mic_amplitude"):
            try:
                state_mgr.update_mic_amplitude(0.0)
            except Exception as exc:
                self._logger.debug("Error resetting mic amplitude on state_manager: %s", exc)

    # --------------------------------------------------------------------------
    # Interruption and Voice Control
    # --------------------------------------------------------------------------

    def interrupt(self) -> None:
        """Safely interrupt active audio playback, reset amplitude, and recover state."""
        with self._lock:
            if self._player is not None:
                try:
                    self._player.interrupt()
                except Exception as exc:
                    self._logger.debug("Error stopping audio player in interrupt(): %s", exc)

        self._reset_mic_amplitude()

        state_mgr = self._get_state_manager()
        if state_mgr is not None and hasattr(state_mgr, "transition_to"):
            try:
                from app.core.state import AssistantState
                # Use get_snapshot() — AssistantStateManager has no .snapshot property
                snap = state_mgr.get_snapshot() if hasattr(state_mgr, "get_snapshot") else None
                curr_state = snap.state if snap is not None else None
                if curr_state in (AssistantState.SPEAKING, AssistantState.LISTENING, AssistantState.THINKING):
                    state_mgr.transition_to(AssistantState.IDLE, status_message="Interrupted")
            except Exception as exc:
                self._logger.debug("Error transitioning state during interrupt: %s", exc)

    def stop_speaking(self) -> None:
        """Alias for interrupt() to stop spoken response immediately."""
        self.interrupt()

    # --------------------------------------------------------------------------
    # Lifecycle Control
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Activate the engine and its microphone recorder."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True

        self._bind_player_amplitude_telemetry()

        try:
            if hasattr(self._recorder, "start"):
                self._recorder.start()
        except Exception as exc:
            self._logger.warning("Could not start microphone recorder: %s", exc)

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_ENGINE_STARTED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_VOICE_ENGINE,
            )
        self._logger.info("VoiceConversationEngine started.")

    def stop(self) -> None:
        """Deactivate the engine, microphone recorder, and active playback."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._player is not None:
            try:
                self._player.stop()
            except Exception as exc:
                self._logger.warning("Could not stop audio player: %s", exc)

        try:
            if hasattr(self._recorder, "stop"):
                self._recorder.stop()
        except Exception as exc:
            self._logger.warning("Could not stop microphone recorder: %s", exc)

        self._reset_mic_amplitude()

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_ENGINE_STOPPED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_VOICE_ENGINE,
            )
        self._logger.info("VoiceConversationEngine stopped.")

    def close(self) -> None:
        """Release all allocated engine resources."""
        self.stop()
        if self._player is not None:
            try:
                self._player.shutdown()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    def __enter__(self) -> VoiceConversationEngine:
        """Context manager entry activates engine."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit deactivates engine."""
        self.stop()

    # --------------------------------------------------------------------------
    # Response Formatting Helper
    # --------------------------------------------------------------------------

    @staticmethod
    def _format_response(result: Any) -> str:
        """Extract a clean, speakable text response string from a skill result.

        Args:
            result: Raw execution result from the matched skill / router.

        Returns:
            Clean string representation suitable for TTS synthesis.
        """
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if hasattr(result, "content") and isinstance(result.content, str):
            return result.content.strip()

        if hasattr(result, "response") and isinstance(result.response, str):
            return result.response.strip()

        if hasattr(result, "message") and isinstance(result.message, str):
            return result.message.strip()

        if isinstance(result, dict):
            for key in ("response", "content", "message", "text", "output", "result"):
                val = result.get(key)
                if val is not None and isinstance(val, str) and val.strip():
                    return val.strip()
            return str(result).strip()

        for attr in ("response", "message", "text", "output"):
            if hasattr(result, attr):
                val = getattr(result, attr)
                if val:
                    return str(val).strip()

        return str(result).strip()

    # --------------------------------------------------------------------------
    # Core Interaction Flow: listen_once()
    # --------------------------------------------------------------------------

    def listen_once(
        self,
        *,
        duration: Optional[float] = None,
        audio_path: Optional[Union[str, Path]] = None,
        play_audio: bool = True,
        use_vad: Optional[bool] = None,
    ) -> VoiceConversationResult:
        """Perform exactly one voice interaction cycle.

        Flow:
            1. Record audio (via MicrophoneRecorder) or use supplied audio_path.
            2. Transcribe audio to text (via SpeechToText).
            3. Route transcribed text (via CommandRouter).
            4. Receive response and format speakable string.
            5. Synthesize TTS and play response through AudioPlayer with RMS telemetry.
            6. Return VoiceConversationResult.

        Args:
            duration: Optional duration in seconds to record. Defaults to self.record_duration.
            audio_path: Optional pre-recorded audio file path to bypass microphone recording.
            play_audio: If True, plays back synthesized response audio through AudioPlayer.
            use_vad: Optional flag to force dynamic VAD (True) or fixed-duration (False).

        Returns:
            VoiceConversationResult encapsulating all stage outcomes and metadata.
        """
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())
        rec_duration = duration if duration is not None else self._record_duration

        # 1. Record audio
        target_audio: Optional[Path] = None
        if audio_path is not None:
            self._reset_mic_amplitude()
            target_audio = Path(audio_path).resolve()
        else:
            self._bind_amplitude_telemetry()
            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_ENGINE_RECORDING,
                    payload={"session_id": session_id, "duration": rec_duration},
                    source=EVENT_SOURCE_VOICE_ENGINE,
                )
            try:
                if use_vad is not None:
                    target_audio = self._recorder.record(duration=rec_duration, use_vad=use_vad)
                else:
                    target_audio = self._recorder.record(duration=rec_duration)
            except Exception as exc:
                self._reset_mic_amplitude()
                err_msg = f"Audio recording failed: {exc}"
                self._logger.error(err_msg, exc_info=True)
                self._publish_failure(exc, session_id=session_id, phase="recording")
                return VoiceConversationResult(
                    session_id=session_id,
                    audio_path=None,
                    success=False,
                    error=err_msg,
                    duration=time.perf_counter() - start_time,
                )
            finally:
                self._reset_mic_amplitude()

        if not target_audio or not target_audio.exists():
            err_msg = f"Recorded audio file does not exist: {target_audio}"
            self._logger.error(err_msg)
            self._publish_failure(FileNotFoundError(err_msg), session_id=session_id, phase="recording")
            return VoiceConversationResult(
                session_id=session_id,
                audio_path=target_audio,
                success=False,
                error=err_msg,
                duration=time.perf_counter() - start_time,
            )

        # 2. Transcribe
        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_ENGINE_TRANSCRIBING,
                payload={"session_id": session_id, "audio_path": str(target_audio)},
                source=EVENT_SOURCE_VOICE_ENGINE,
            )

        transcription_res: Optional[TranscriptionResult] = None
        raw_text: str = ""
        try:
            transcription_res = self._stt.transcribe(target_audio)
            if transcription_res is not None:
                raw_text = (
                    transcription_res.text
                    if hasattr(transcription_res, "text")
                    else str(transcription_res)
                ).strip()
        except Exception as exc:
            err_msg = f"Speech-to-text transcription failed: {exc}"
            self._logger.error(err_msg, exc_info=True)
            self._publish_failure(exc, session_id=session_id, phase="transcription")
            return VoiceConversationResult(
                session_id=session_id,
                audio_path=target_audio,
                transcription=transcription_res,
                success=False,
                error=err_msg,
                duration=time.perf_counter() - start_time,
            )

        # Handle silence or empty transcription
        if not raw_text:
            self._logger.info("Speech-to-text returned empty transcription; interaction finished.")
            total_duration = time.perf_counter() - start_time
            return VoiceConversationResult(
                session_id=session_id,
                audio_path=target_audio,
                transcription=transcription_res,
                text="",
                command="",
                response=None,
                response_text="",
                speech_result=None,
                success=True,
                duration=total_duration,
            )

        # 3. Route through CommandRouter
        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_ENGINE_ROUTING,
                payload={"session_id": session_id, "command": raw_text},
                source=EVENT_SOURCE_VOICE_ENGINE,
            )

        command_result: Optional[Any] = None
        try:
            self._logger.info("Routing voice command: '%s'", raw_text)
            command_result = self._router.route(
                command=raw_text,
                source="voice",
                context={"session_id": session_id},
            )
        except Exception as exc:
            err_msg = f"Command routing failed for '{raw_text}': {exc}"
            self._logger.error(err_msg, exc_info=True)
            self._publish_failure(exc, session_id=session_id, phase="routing")
            return VoiceConversationResult(
                session_id=session_id,
                audio_path=target_audio,
                transcription=transcription_res,
                text=raw_text,
                command=raw_text,
                success=False,
                error=err_msg,
                duration=time.perf_counter() - start_time,
            )

        # 4. Receive and format response
        response_text = self._format_response(command_result)
        self._logger.debug("Received command response text: '%s'", response_text[:60])

        # 5. Speak response (Synthesize -> Speaking event -> Playback with RMS telemetry)
        speech_res: Optional[SpeechResult] = None
        if response_text:
            try:
                self._logger.info("Synthesizing speech response: '%s'", response_text[:40])
                speech_res = self._tts.synthesize(response_text)
            except Exception as exc:
                self._logger.warning("TTS speech synthesis failed: %s", exc, exc_info=True)
                # Synthesis failure is non-fatal to the conversation return value
                self._publish_failure(exc, session_id=session_id, phase="speech_synthesis")

            if self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_ENGINE_SPEAKING,
                    payload={"session_id": session_id, "response_text": response_text},
                    source=EVENT_SOURCE_VOICE_ENGINE,
                )

            if play_audio and self._player is not None and speech_res is not None and speech_res.audio_path:
                audio_file = Path(speech_res.audio_path)
                if audio_file.exists() and audio_file.stat().st_size > 0:
                    self._bind_player_amplitude_telemetry()
                    try:
                        self._player.play(audio_file, block=True)
                    except Exception as play_exc:
                        self._logger.warning("Audio playback failed: %s", play_exc)
                    finally:
                        self._reset_mic_amplitude()

        # 6. Return result
        total_duration = time.perf_counter() - start_time
        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_ENGINE_COMPLETED,
                payload={
                    "session_id": session_id,
                    "command": raw_text,
                    "response_text": response_text,
                    "duration": total_duration,
                },
                source=EVENT_SOURCE_VOICE_ENGINE,
            )

        return VoiceConversationResult(
            session_id=session_id,
            audio_path=target_audio,
            transcription=transcription_res,
            text=raw_text,
            command=raw_text,
            response=command_result,
            response_text=response_text,
            speech_result=speech_res,
            success=True,
            duration=total_duration,
        )

    async def listen_once_async(
        self,
        *,
        duration: Optional[float] = None,
        audio_path: Optional[Union[str, Path]] = None,
        play_audio: bool = True,
        use_vad: Optional[bool] = None,
    ) -> VoiceConversationResult:
        """Asynchronously perform exactly one voice interaction cycle.

        Executes listen_once() within a worker thread pool.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.listen_once(
                duration=duration,
                audio_path=audio_path,
                play_audio=play_audio,
                use_vad=use_vad,
            ),
        )

    # --------------------------------------------------------------------------
    # Continuous Loop Runner
    # --------------------------------------------------------------------------

    def run_loop(
        self,
        should_continue_fn: Optional[Callable[[], bool]] = None,
        on_turn_callback: Optional[Callable[[VoiceConversationResult], None]] = None,
    ) -> None:
        """Run continuous voice conversation turns until stopped.

        Args:
            should_continue_fn: Optional callable returning False to break loop.
            on_turn_callback: Optional callable receiving each VoiceConversationResult.
        """
        self.start()
        try:
            while self.is_running:
                if should_continue_fn is not None and not should_continue_fn():
                    break

                result = self.listen_once()
                if on_turn_callback is not None:
                    on_turn_callback(result)

                if result.command and result.command.lower() in ("exit", "quit"):
                    break
        finally:
            self.stop()

    # --------------------------------------------------------------------------
    # Internal Event Helpers
    # --------------------------------------------------------------------------

    def _publish_failure(
        self,
        error: Exception,
        *,
        session_id: str,
        phase: str,
    ) -> None:
        """Publish a voice_engine.failed event synchronously."""
        if self._event_bus is None:
            return
        self._event_bus.publish(
            event=EVENT_ENGINE_FAILED,
            payload={
                "session_id": session_id,
                "phase": phase,
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": time.time(),
            },
            source=EVENT_SOURCE_VOICE_ENGINE,
        )


# Global default convenience instance
voice_conversation_engine = VoiceConversationEngine(auto_register_in_container=False)

__all__ = [
    "EVENT_ENGINE_COMPLETED",
    "EVENT_ENGINE_FAILED",
    "EVENT_ENGINE_RECORDING",
    "EVENT_ENGINE_ROUTING",
    "EVENT_ENGINE_SPEAKING",
    "EVENT_ENGINE_STARTED",
    "EVENT_ENGINE_STOPPED",
    "EVENT_ENGINE_TRANSCRIBING",
    "EVENT_SOURCE_VOICE_ENGINE",
    "VoiceConversationEngine",
    "voice_conversation_engine",
]
