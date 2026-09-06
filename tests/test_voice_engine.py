"""Comprehensive unit and integration tests for VoiceConversationEngine and Voice Mode."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.application import (
    EVENT_APPLICATION_READY,
    EVENT_APPLICATION_STOPPED,
    JarvisApplication,
)
from app.core.config import Settings, load_settings
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.router.router import CommandRouter
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech
from app.voice.engine import (
    EVENT_ENGINE_COMPLETED,
    EVENT_ENGINE_FAILED,
    EVENT_ENGINE_RECORDING,
    EVENT_ENGINE_ROUTING,
    EVENT_ENGINE_SPEAKING,
    EVENT_ENGINE_STARTED,
    EVENT_ENGINE_STOPPED,
    EVENT_ENGINE_TRANSCRIBING,
    EVENT_SOURCE_VOICE_ENGINE,
    VoiceConversationEngine,
    voice_conversation_engine,
)
from app.voice.microphone import MicrophoneRecorder
from app.voice.models import RecordingError, VoiceConversationResult
from main import main, parse_args


class TestVoiceConversationResult(unittest.TestCase):
    """Unit tests for VoiceConversationResult data model."""

    def test_result_creation_and_properties(self) -> None:
        """Verify VoiceConversationResult fields, default values, and backward-compatible properties."""
        test_wav = Path(tempfile.gettempdir()) / "sample_audio.wav"
        test_mp3 = Path(tempfile.gettempdir()) / "sample_response.mp3"
        transcription = TranscriptionResult(text="what time is it", language="en")
        speech = SpeechResult(
            text="It is 3 PM",
            audio_path=test_mp3,
            voice="en-US-JennyNeural",
            file_size_bytes=1234,
        )

        result = VoiceConversationResult(
            session_id="session-42",
            audio_path=test_wav,
            transcription=transcription,
            text="what time is it",
            command="what time is it",
            response={"message": "It is 3 PM"},
            response_text="It is 3 PM",
            speech_result=speech,
            success=True,
            duration=1.25,
        )

        self.assertEqual(result.session_id, "session-42")
        self.assertEqual(result.audio_path, test_wav.resolve())
        self.assertEqual(result.audio_input, test_wav.resolve())
        self.assertEqual(result.text, "what time is it")
        self.assertEqual(result.command, "what time is it")
        self.assertEqual(result.command_text, "what time is it")
        self.assertEqual(result.response, {"message": "It is 3 PM"})
        self.assertEqual(result.command_result, {"message": "It is 3 PM"})
        self.assertEqual(result.response_text, "It is 3 PM")
        self.assertEqual(result.speech_result, speech)
        self.assertEqual(result.audio_output, test_mp3)
        self.assertTrue(result.success)
        self.assertTrue(bool(result))
        self.assertEqual(str(result), "It is 3 PM")

    def test_result_serialization_to_dict(self) -> None:
        """Verify to_dict produces complete, serialized dictionary representation."""
        result = VoiceConversationResult(
            session_id="sess-100",
            text="hello",
            command="hello",
            response="Hello there!",
            response_text="Hello there!",
            success=True,
            duration=0.5,
        )

        data = result.to_dict()
        self.assertEqual(data["session_id"], "sess-100")
        self.assertEqual(data["text"], "hello")
        self.assertEqual(data["command"], "hello")
        self.assertEqual(data["response"], "Hello there!")
        self.assertEqual(data["response_text"], "Hello there!")
        self.assertTrue(data["success"])
        self.assertEqual(data["duration"], 0.5)
        self.assertIn("timestamp", data)


class TestVoiceConversationEngine(unittest.TestCase):
    """Comprehensive test suite for VoiceConversationEngine."""

    def setUp(self) -> None:
        """Set up isolated container, mocks, and test environment."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.logger = logging.getLogger("TEST_VOICE_ENGINE")
        self.settings = load_settings()

        # Mock core collaborators
        self.mock_recorder = MagicMock(spec=MicrophoneRecorder)
        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_router = MagicMock(spec=CommandRouter)
        self.mock_tts = MagicMock(spec=TextToSpeech)

        # Temporary audio file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_audio_path = Path(self.temp_dir.name) / "test_recorded.wav"
        self.test_audio_path.write_bytes(b"RIFF dummy pcm wav data")
        self.mock_recorder.record.return_value = self.test_audio_path

        # Temporary synthesized response file
        self.test_speech_path = Path(self.temp_dir.name) / "test_response.mp3"
        self.test_speech_path.write_bytes(b"dummy mp3 data")
        self.mock_speech_result = SpeechResult(
            text="Command executed successfully.",
            audio_path=self.test_speech_path,
            voice="en-US-JennyNeural",
            file_size_bytes=100,
        )
        self.mock_tts.synthesize.return_value = self.mock_speech_result

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_initialization_with_injected_components(self) -> None:
        """Verify engine initialization with explicit components and container registration."""
        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            config=self.settings,
            logger=self.logger,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            record_duration=4.5,
            auto_register_in_container=True,
        )

        self.assertIs(engine.recorder, self.mock_recorder)
        self.assertIs(engine.stt, self.mock_stt)
        self.assertIs(engine.router, self.mock_router)
        self.assertIs(engine.tts, self.mock_tts)
        self.assertEqual(engine.record_duration, 4.5)
        self.assertFalse(engine.is_running)

        # Container registration check
        self.assertTrue(self.container.exists("voice_engine"))
        self.assertTrue(self.container.exists("voice_conversation_engine"))
        self.assertIs(self.container.resolve("voice_engine"), engine)

    def test_record_duration_validation(self) -> None:
        """Verify record_duration setter rejects non-positive values."""
        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            record_duration=3.0,
        )
        engine.record_duration = 5.0
        self.assertEqual(engine.record_duration, 5.0)

        with self.assertRaises(ValueError):
            engine.record_duration = 0.0

        with self.assertRaises(ValueError):
            engine.record_duration = -2.0

    def test_lifecycle_start_stop_and_context_manager(self) -> None:
        """Verify start(), stop(), context manager, and lifecycle events."""
        events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_STARTED, events.append)
        self.event_bus.subscribe(EVENT_ENGINE_STOPPED, events.append)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        # 1. Start
        engine.start()
        self.assertTrue(engine.is_running)
        self.mock_recorder.start.assert_called_once()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, EVENT_ENGINE_STARTED)

        # 2. Stop
        engine.stop()
        self.assertFalse(engine.is_running)
        self.mock_recorder.stop.assert_called_once()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].name, EVENT_ENGINE_STOPPED)

        # 3. Context manager
        with engine:
            self.assertTrue(engine.is_running)
        self.assertFalse(engine.is_running)

    def test_complete_interaction_flow(self) -> None:
        """Verify the exact execution flow:
        record audio -> transcribe -> route through CommandRouter -> receive response -> speak response -> return result
        """
        published_topics: list[str] = []
        for topic in (
            EVENT_ENGINE_RECORDING,
            EVENT_ENGINE_TRANSCRIBING,
            EVENT_ENGINE_ROUTING,
            EVENT_ENGINE_SPEAKING,
            EVENT_ENGINE_COMPLETED,
        ):
            self.event_bus.subscribe(topic, lambda ev: published_topics.append(ev.name))

        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="turn on the living room lights", language="en"
        )
        self.mock_router.route.return_value = "Living room lights turned on."

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            record_duration=2.5,
        )

        result = engine.listen_once()

        # 1. Check recording step
        self.mock_recorder.record.assert_called_once_with(duration=2.5)

        # 2. Check transcribe step
        self.mock_stt.transcribe.assert_called_once_with(self.test_audio_path)

        # 3. Check router step
        self.mock_router.route.assert_called_once()
        call_args = self.mock_router.route.call_args
        self.assertEqual(call_args.kwargs["command"], "turn on the living room lights")
        self.assertEqual(call_args.kwargs["source"], "voice")

        # 4. Check TTS synthesize step
        self.mock_tts.synthesize.assert_called_once_with("Living room lights turned on.")

        # 5. Check return result
        self.assertTrue(result.success)
        self.assertEqual(result.audio_path, self.test_audio_path)
        self.assertEqual(result.text, "turn on the living room lights")
        self.assertEqual(result.command, "turn on the living room lights")
        self.assertEqual(result.response, "Living room lights turned on.")
        self.assertEqual(result.response_text, "Living room lights turned on.")
        self.assertEqual(result.speech_result, self.mock_speech_result)
        self.assertGreaterEqual(result.duration, 0.0)

        # 6. Verify sequential lifecycle event topics
        self.assertEqual(
            published_topics,
            [
                EVENT_ENGINE_RECORDING,
                EVENT_ENGINE_TRANSCRIBING,
                EVENT_ENGINE_ROUTING,
                EVENT_ENGINE_SPEAKING,
                EVENT_ENGINE_COMPLETED,
            ],
        )

    def test_listen_once_with_supplied_audio_path(self) -> None:
        """Verify listen_once bypasses microphone recording when audio_path is provided."""
        custom_wav = Path(self.temp_dir.name) / "custom_input.wav"
        custom_wav.write_bytes(b"custom audio")
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="status report", language="en")
        self.mock_router.route.return_value = "All systems operational."

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once(audio_path=custom_wav)

        self.mock_recorder.record.assert_not_called()
        self.mock_stt.transcribe.assert_called_once_with(custom_wav)
        self.assertEqual(result.text, "status report")
        self.assertEqual(result.response_text, "All systems operational.")

    def test_empty_transcription_returns_clean_result_without_routing(self) -> None:
        """Verify empty transcription or silence terminates turn gracefully without routing or speaking."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="   ", language="en")

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once()

        self.assertTrue(result.success)
        self.assertEqual(result.text, "")
        self.assertEqual(result.command, "")
        self.assertIsNone(result.response)
        self.mock_router.route.assert_not_called()
        self.mock_tts.synthesize.assert_not_called()

    def test_response_formatting_varieties(self) -> None:
        """Verify extraction of speakable response string from varied skill output types."""
        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        # None -> empty
        self.assertEqual(engine._format_response(None), "")
        # String
        self.assertEqual(engine._format_response("Clean string"), "Clean string")
        # Dict with 'response'
        self.assertEqual(engine._format_response({"response": "Dictionary response"}), "Dictionary response")
        # Dict with 'message'
        self.assertEqual(engine._format_response({"message": "Status OK"}), "Status OK")
        # Object with .content
        mock_obj = MagicMock()
        mock_obj.content = "Object content message"
        self.assertEqual(engine._format_response(mock_obj), "Object content message")

    def test_recording_error_handling(self) -> None:
        """Verify recording failure returns failure result and publishes voice_engine.failed."""
        self.mock_recorder.record.side_effect = RecordingError("Microphone device busy")
        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_FAILED, failed_events.append)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once()

        self.assertFalse(result.success)
        self.assertIn("Microphone device busy", str(result.error))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["phase"], "recording")

    def test_transcription_error_handling(self) -> None:
        """Verify STT failure returns failure result and publishes failure event."""
        self.mock_stt.transcribe.side_effect = RuntimeError("STT model crashed")
        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_FAILED, failed_events.append)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once()

        self.assertFalse(result.success)
        self.assertIn("STT model crashed", str(result.error))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["phase"], "transcription")

    def test_routing_error_handling(self) -> None:
        """Verify router execution failure returns failure result and publishes failure event."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="execute test command", language="en"
        )
        self.mock_router.route.side_effect = ValueError("Unrecognized skill intent")
        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_FAILED, failed_events.append)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once()

        self.assertFalse(result.success)
        self.assertIn("Unrecognized skill intent", str(result.error))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["phase"], "routing")

    def test_tts_synthesis_error_is_non_fatal(self) -> None:
        """Verify TTS synthesis failure emits event but returns successful conversation turn."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="what is 2 + 2", language="en")
        self.mock_router.route.return_value = "2 + 2 is 4."
        self.mock_tts.synthesize.side_effect = RuntimeError("Network error during edge-tts")
        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_FAILED, failed_events.append)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        result = engine.listen_once()

        # Turn completed and command succeeded
        self.assertTrue(result.success)
        self.assertEqual(result.response_text, "2 + 2 is 4.")
        self.assertIsNone(result.speech_result)
        # Failure event emitted for TTS phase
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["phase"], "speech_synthesis")

    def test_listen_once_async(self) -> None:
        """Verify listen_once_async executes successfully via asyncio."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="async voice command", language="en"
        )
        self.mock_router.route.return_value = "Async response received."

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        async def run_test() -> VoiceConversationResult:
            return await engine.listen_once_async()

        result = asyncio.run(run_test())
        self.assertTrue(result.success)
        self.assertEqual(result.text, "async voice command")
        self.assertEqual(result.response_text, "Async response received.")

    def test_run_loop_terminates_on_exit_command(self) -> None:
        """Verify run_loop executes turns and stops when command is 'exit'."""
        turn_results: list[VoiceConversationResult] = []

        turn1 = VoiceConversationResult(command="turn one", text="turn one", success=True)
        turn2 = VoiceConversationResult(command="exit", text="exit", success=True)

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
        )

        engine.listen_once = MagicMock(side_effect=[turn1, turn2])  # type: ignore[assignment]
        engine.run_loop(on_turn_callback=turn_results.append)

        self.assertEqual(len(turn_results), 2)
        self.assertEqual(turn_results[0].command, "turn one")
        self.assertEqual(turn_results[1].command, "exit")
        self.assertFalse(engine.is_running)


class TestApplicationVoiceIntegration(unittest.TestCase):
    """Integration tests verifying JarvisApplication integration with VoiceConversationEngine."""

    def setUp(self) -> None:
        """Create fresh subsystems for testing."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.logger = logging.getLogger("TEST_APP_VOICE")
        self.settings = load_settings()

    def test_application_initializes_voice_engine(self) -> None:
        """Verify JarvisApplication initializes VoiceConversationEngine and registers in container."""
        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            print_ready=False,
        )

        self.assertIsInstance(app.voice_engine, VoiceConversationEngine)
        self.assertIs(app.voice_conversation_engine, app.voice_engine)
        self.assertFalse(app.voice_mode)

        # Service container registration check
        self.assertTrue(self.container.exists("voice_engine"))
        self.assertTrue(self.container.exists("voice_conversation_engine"))
        self.assertIs(self.container.resolve("voice_engine"), app.voice_engine)

    def test_application_voice_mode_property_and_setter(self) -> None:
        """Verify voice_mode property and setter on JarvisApplication."""
        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            voice_mode=False,
            print_ready=False,
        )
        self.assertFalse(app.voice_mode)
        app.voice_mode = True
        self.assertTrue(app.voice_mode)

    def test_application_listen_once_delegation(self) -> None:
        """Verify app.listen_once() delegates directly to voice_engine.listen_once()."""
        mock_engine = MagicMock(spec=VoiceConversationEngine)
        mock_result = VoiceConversationResult(
            text="hello",
            command="hello",
            response="Hi!",
            response_text="Hi!",
            success=True,
        )
        mock_engine.listen_once.return_value = mock_result

        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            voice_engine=mock_engine,
            print_ready=False,
        )

        result = app.listen_once(duration=4.0)
        mock_engine.listen_once.assert_called_once_with(duration=4.0, audio_path=None)
        self.assertIs(result, mock_result)

    def test_application_run_voice_mode_loop_execution_and_termination(self) -> None:
        """Verify app.run(voice_mode=True) executes voice loop and terminates cleanly on 'quit'."""
        mock_engine = MagicMock(spec=VoiceConversationEngine)

        turn1 = VoiceConversationResult(
            text="what time is it",
            command="time",
            response="It is 12:00",
            response_text="It is 12:00",
            success=True,
        )
        turn2 = VoiceConversationResult(
            text="quit",
            command="quit",
            response="Goodbye",
            response_text="Goodbye",
            success=True,
        )
        mock_engine.listen_once.side_effect = [turn1, turn2]

        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            voice_engine=mock_engine,
            voice_mode=True,
            print_ready=False,
        )

        outputs: list[str] = []
        exit_code = app.run(output_fn=outputs.append)

        self.assertEqual(exit_code, 0)
        self.assertFalse(app.is_running)
        self.assertEqual(mock_engine.listen_once.call_count, 2)
        # Check output captured both turns
        full_output = "".join(outputs)
        self.assertIn("Voice Mode Active", full_output)
        self.assertIn("what time is it", full_output)
        self.assertIn("It is 12:00", full_output)

    def test_application_run_voice_mode_override_flag(self) -> None:
        """Verify app.run(voice_mode=True) flag overrides voice_mode=False on instance."""
        mock_engine = MagicMock(spec=VoiceConversationEngine)
        mock_engine.listen_once.return_value = VoiceConversationResult(command="exit", success=True)

        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            voice_engine=mock_engine,
            voice_mode=False,  # default text
            print_ready=False,
        )

        exit_code = app.run(output_fn=lambda msg: None, voice_mode=True)
        self.assertEqual(exit_code, 0)
        mock_engine.listen_once.assert_called_once()


class TestMainCLI(unittest.TestCase):
    """Unit tests for main.py argument parsing and --voice CLI flag execution."""

    def test_parse_args_with_voice_flag(self) -> None:
        """Verify parse_args recognizes --voice."""
        args = parse_args(["--voice"])
        self.assertTrue(args.voice)

    def test_parse_args_without_voice_flag(self) -> None:
        """Verify parse_args defaults voice to False when --voice is absent."""
        args = parse_args([])
        self.assertFalse(args.voice)

    def test_main_with_voice_flag_activates_voice_mode(self) -> None:
        """Verify python main.py --voice passes voice_mode=True to app.run()."""
        mock_app = MagicMock(spec=JarvisApplication)
        mock_app.run.return_value = 0

        with patch("main.bootstrap", return_value=True):
            exit_code = main(args=["--voice"], app_instance=mock_app, interactive=True)

        self.assertEqual(exit_code, 0)
        mock_app.run.assert_called_once_with(voice_mode=True)

    def test_main_without_voice_flag_runs_standard(self) -> None:
        """Verify python main.py runs standard interactive loop."""
        mock_app = MagicMock(spec=JarvisApplication)
        mock_app.run.return_value = 0

        with patch("main.bootstrap", return_value=True):
            exit_code = main(args=[], app_instance=mock_app, interactive=True)

        self.assertEqual(exit_code, 0)
        mock_app.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
