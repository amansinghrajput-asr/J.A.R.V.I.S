"""Comprehensive unit tests for the J.A.R.V.I.S Voice Pipeline Subsystem."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import Settings, load_settings
from app.core.container import JarvisException, ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.memory.manager import MemoryManager
from app.router.router import CommandRouter
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech
from app.voice import (
    EVENT_SOURCE_VOICE,
    EVENT_VOICE_FAILED,
    EVENT_VOICE_LISTENING,
    EVENT_VOICE_PROCESSING,
    EVENT_VOICE_RESPONDING,
    EVENT_VOICE_STARTED,
    EVENT_VOICE_STOPPED,
    AudioProvider,
    AudioProviderError,
    InvalidStateError,
    PlaceholderAudioProvider,
    PipelineExecutionError,
    PipelineNotRunningError,
    VoicePipeline,
    VoicePipelineError,
    VoicePipelineResult,
    VoiceState,
    voice_pipeline,
)
from app.wakeword.detector import WakeWordDetector
from app.wakeword.models import WakeWordResult


class TestVoiceModels(unittest.TestCase):
    """Tests for Voice Pipeline models, enums, and exceptions."""

    def test_voice_state_enum(self) -> None:
        """Verify VoiceState enum values and string conversion."""
        self.assertEqual(VoiceState.IDLE.value, "idle")
        self.assertEqual(VoiceState.LISTENING.value, "listening")
        self.assertEqual(VoiceState.PROCESSING.value, "processing")
        self.assertEqual(VoiceState.SPEAKING.value, "speaking")
        self.assertEqual(str(VoiceState.IDLE), "idle")
        self.assertEqual(str(VoiceState.LISTENING), "listening")

    def test_exception_hierarchy(self) -> None:
        """Verify exception hierarchy inherits from VoicePipelineError and JarvisException."""
        self.assertTrue(issubclass(VoicePipelineError, JarvisException))
        self.assertTrue(issubclass(AudioProviderError, VoicePipelineError))
        self.assertTrue(issubclass(InvalidStateError, VoicePipelineError))
        self.assertTrue(issubclass(PipelineNotRunningError, VoicePipelineError))
        self.assertTrue(issubclass(PipelineExecutionError, VoicePipelineError))

    def test_voice_pipeline_result_dataclass(self) -> None:
        """Test VoicePipelineResult fields, normalization, and dictionary serialization."""
        test_path = Path("/tmp/test_audio.wav")
        transcription = TranscriptionResult(text="Hey Jarvis hello", language="en")
        wake_res = WakeWordResult(detected=True, matched_phrase="Hey Jarvis")
        speech_res = SpeechResult(
            text="Hello user",
            audio_path=Path("/tmp/speech.mp3"),
            voice="en-GB-RyanNeural",
            file_size_bytes=100,
        )

        res = VoicePipelineResult(
            session_id="test-session-123",
            state=VoiceState.LISTENING,
            audio_input=test_path,
            transcription=transcription,
            wake_word_result=wake_res,
            command_text="hello",
            command_result="Hello user",
            speech_result=speech_res,
            success=True,
            duration=0.45,
        )

        self.assertEqual(res.session_id, "test-session-123")
        self.assertEqual(res.state, VoiceState.LISTENING)
        self.assertEqual(res.audio_input, test_path.resolve())
        self.assertTrue(bool(res))

        data = res.to_dict()
        self.assertEqual(data["session_id"], "test-session-123")
        self.assertEqual(data["state"], "listening")
        self.assertEqual(data["command_text"], "hello")
        self.assertEqual(data["command_result"], "Hello user")
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["transcription"])
        self.assertIsNotNone(data["wake_word_result"])
        self.assertIsNotNone(data["speech_result"])


class TestPlaceholderAudioProvider(unittest.TestCase):
    """Tests for the PlaceholderAudioProvider interface implementation."""

    def test_initial_state_and_start_stop(self) -> None:
        """Verify start/stop lifecycle and active status."""
        provider = PlaceholderAudioProvider()
        self.assertFalse(provider.is_active())

        provider.start()
        self.assertTrue(provider.is_active())

        provider.stop()
        self.assertFalse(provider.is_active())

    def test_enqueue_and_capture(self) -> None:
        """Verify queueing, FIFO ordering, pending count, and capture behavior."""
        p1 = Path("/tmp/sample1.wav")
        p2 = Path("/tmp/sample2.wav")

        provider = PlaceholderAudioProvider([p1])
        self.assertEqual(provider.pending_count(), 1)

        # Before starting, capture returns None
        self.assertIsNone(provider.capture_audio())

        provider.start()
        provider.enqueue_audio(p2)
        self.assertEqual(provider.pending_count(), 2)

        # First capture returns p1
        captured1 = provider.capture_audio()
        self.assertEqual(captured1, p1.resolve())
        self.assertEqual(provider.pending_count(), 1)

        # Second capture returns p2
        captured2 = provider.capture_audio()
        self.assertEqual(captured2, p2.resolve())
        self.assertEqual(provider.pending_count(), 0)

        # Third capture returns None (queue empty)
        self.assertIsNone(provider.capture_audio())

    def test_clear_queue(self) -> None:
        """Verify clearing queued audio files."""
        provider = PlaceholderAudioProvider(["/tmp/a.wav", "/tmp/b.wav"])
        self.assertEqual(provider.pending_count(), 2)
        provider.clear()
        self.assertEqual(provider.pending_count(), 0)

    def test_async_capture(self) -> None:
        """Verify asynchronous capture delegation."""
        p = Path("/tmp/async.wav")
        provider = PlaceholderAudioProvider([p])
        provider.start()

        async def run() -> Optional[Path]:
            return await provider.capture_audio_async()

        captured = asyncio.run(run())
        self.assertEqual(captured, p.resolve())


class TestVoicePipelineState(unittest.TestCase):
    """Tests for state transitions and state machine validation."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.pipeline = VoicePipeline(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        self.pipeline.stop()
        self.container.clear()
        self.event_bus.clear()

    def test_initial_state_idle(self) -> None:
        """Verify initial pipeline state is IDLE."""
        self.assertEqual(self.pipeline.state, VoiceState.IDLE)
        self.assertFalse(self.pipeline.is_running)

    def test_valid_state_transitions(self) -> None:
        """Verify valid sequence: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> LISTENING -> IDLE."""
        events: list[Event] = []
        self.event_bus.subscribe("*", lambda e: events.append(e))

        # IDLE -> LISTENING
        self.pipeline._transition_to(VoiceState.LISTENING)
        self.assertEqual(self.pipeline.state, VoiceState.LISTENING)

        # LISTENING -> PROCESSING
        self.pipeline._transition_to(VoiceState.PROCESSING)
        self.assertEqual(self.pipeline.state, VoiceState.PROCESSING)

        # PROCESSING -> SPEAKING
        self.pipeline._transition_to(VoiceState.SPEAKING)
        self.assertEqual(self.pipeline.state, VoiceState.SPEAKING)

        # SPEAKING -> LISTENING
        self.pipeline._transition_to(VoiceState.LISTENING)
        self.assertEqual(self.pipeline.state, VoiceState.LISTENING)

        # LISTENING -> IDLE
        self.pipeline._transition_to(VoiceState.IDLE)
        self.assertEqual(self.pipeline.state, VoiceState.IDLE)

        # Check emitted events
        event_names = [e.name for e in events]
        self.assertIn(EVENT_VOICE_LISTENING, event_names)
        self.assertIn(EVENT_VOICE_PROCESSING, event_names)
        self.assertIn(EVENT_VOICE_RESPONDING, event_names)

    def test_invalid_state_transition_raises(self) -> None:
        """Verify invalid transition from IDLE directly to SPEAKING raises InvalidStateError."""
        with self.assertRaises(InvalidStateError):
            self.pipeline._transition_to(VoiceState.SPEAKING)

    def test_thread_safe_start_stop(self) -> None:
        """Verify thread safety during concurrent start and stop invocations."""
        pipeline = VoicePipeline(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        def worker(start_first: bool) -> None:
            for _ in range(50):
                if start_first:
                    pipeline.start()
                    pipeline.stop()
                else:
                    pipeline.stop()
                    pipeline.start()

        threads = [
            threading.Thread(target=worker, args=(True,)),
            threading.Thread(target=worker, args=(False,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pipeline.stop()
        self.assertEqual(pipeline.state, VoiceState.IDLE)


class TestVoicePipelineEndToEnd(unittest.TestCase):
    """End-to-end pipeline execution tests mocking downstream perception and cognition components."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.audio_provider = PlaceholderAudioProvider()

        # Mock STT
        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_stt.transcribe_async = AsyncMock()

        # Mock WakeWordDetector
        self.wake_detector = WakeWordDetector(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Mock Router
        self.mock_router = MagicMock(spec=CommandRouter)
        self.mock_router.route_async = AsyncMock()

        # Mock TTS
        self.mock_tts = MagicMock(spec=TextToSpeech)
        self.mock_tts.synthesize_async = AsyncMock()

        # Real MemoryManager
        self.memory = MemoryManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        self.pipeline = VoicePipeline(
            audio_provider=self.audio_provider,
            stt=self.mock_stt,
            wake_word_detector_instance=self.wake_detector,
            router=self.mock_router,
            tts=self.mock_tts,
            memory=self.memory,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            require_wake_word=True,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        self.pipeline.stop()
        self.container.clear()
        self.event_bus.clear()

    def test_full_voice_interaction_cycle(self) -> None:
        """Test complete voice cycle: audio -> STT -> wake word -> memory -> router -> TTS -> memory."""
        # 1. Setup mock returns
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Hey Jarvis what time is it?", language="en"
        )
        self.mock_router.route.return_value = "The current time is 12:00 PM."
        self.mock_tts.synthesize.return_value = SpeechResult(
            text="The current time is 12:00 PM.",
            audio_path=Path("/tmp/resp.mp3"),
            voice="en-GB-RyanNeural",
            file_size_bytes=500,
        )

        # 2. Track events
        events: list[Event] = []
        self.event_bus.subscribe("*", lambda e: events.append(e))

        # 3. Queue audio and start pipeline
        audio_file = Path("/tmp/time_query.wav")
        self.audio_provider.enqueue_audio(audio_file)
        self.pipeline.start()

        # 4. Process audio cycle
        result = self.pipeline.process_audio()

        # 5. Verify results
        self.assertTrue(result.success)
        self.assertEqual(result.command_text, "what time is it?")
        self.assertEqual(result.command_result, "The current time is 12:00 PM.")
        self.assertIsNotNone(result.speech_result)
        self.assertEqual(result.speech_result.text, "The current time is 12:00 PM.")

        # 6. Verify Memory Integration
        last_assistant_mem = self.memory.get_last()
        self.assertIsNotNone(last_assistant_mem)
        self.assertEqual(last_assistant_mem.role, "assistant")
        self.assertEqual(last_assistant_mem.content, "The current time is 12:00 PM.")

        # Check user memory is stored
        user_memories = self.memory.search("what time is it?")
        self.assertTrue(len(user_memories) > 0)
        self.assertEqual(user_memories[0].role, "user")
        self.assertEqual(user_memories[0].source, "voice")

        # 7. Verify Lifecycle Events
        event_names = [e.name for e in events]
        self.assertIn(EVENT_VOICE_STARTED, event_names)
        self.assertIn(EVENT_VOICE_LISTENING, event_names)
        self.assertIn(EVENT_VOICE_PROCESSING, event_names)
        self.assertIn(EVENT_VOICE_RESPONDING, event_names)

        # Pipeline is running, so state should return to LISTENING
        self.assertEqual(self.pipeline.state, VoiceState.LISTENING)

    def test_wake_word_not_detected_ignores_command(self) -> None:
        """Verify when wake word is absent and require_wake_word=True, command is not routed."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="what is the weather today", language="en"
        )

        audio_file = Path("/tmp/no_wake.wav")
        result = self.pipeline.process_audio(audio_file, require_wake_word=True)

        self.assertTrue(result.success)
        self.assertFalse(result.wake_word_result.detected)
        self.assertIsNone(result.command_text)
        self.assertIsNone(result.command_result)
        self.assertIsNone(result.speech_result)

        # Router and TTS should not be called
        self.mock_router.route.assert_not_called()
        self.mock_tts.synthesize.assert_not_called()

    def test_wake_word_detection_disabled(self) -> None:
        """Verify command routes directly if require_wake_word=False."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="turn on the living room lights", language="en"
        )
        self.mock_router.route.return_value = "Living room lights turned on."
        self.mock_tts.synthesize.return_value = SpeechResult(
            text="Living room lights turned on.",
            audio_path=Path("/tmp/resp2.mp3"),
            voice="en-GB-RyanNeural",
            file_size_bytes=400,
        )

        result = self.pipeline.process_audio(
            Path("/tmp/lights.wav"), require_wake_word=False
        )

        self.assertTrue(result.success)
        self.assertEqual(result.command_text, "turn on the living room lights")
        self.assertEqual(result.command_result, "Living room lights turned on.")
        self.mock_router.route.assert_called_once()
        self.mock_tts.synthesize.assert_called_once()

    def test_empty_transcription_returns_cleanly(self) -> None:
        """Verify empty or whitespace STT transcription exits cycle cleanly."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="", language="en"
        )

        result = self.pipeline.process_audio(Path("/tmp/silence.wav"))
        self.assertTrue(result.success)
        self.assertIsNone(result.command_text)
        self.mock_router.route.assert_not_called()

    def test_no_audio_captured_returns_cleanly(self) -> None:
        """Verify pipeline handles empty audio provider queue cleanly."""
        self.pipeline.start()
        result = self.pipeline.process_audio()
        self.assertTrue(result.success)
        self.assertIsNone(result.audio_input)

    def test_process_command_direct(self) -> None:
        """Verify process_command runs through Router -> Memory -> TTS without STT."""
        self.mock_router.route.return_value = "Command executed successfully."
        self.mock_tts.synthesize.return_value = SpeechResult(
            text="Command executed successfully.",
            audio_path=Path("/tmp/direct.mp3"),
            voice="en-GB-RyanNeural",
            file_size_bytes=300,
        )

        result = self.pipeline.process_command("open calculator")

        self.assertTrue(result.success)
        self.assertEqual(result.command_text, "open calculator")
        self.assertEqual(result.command_result, "Command executed successfully.")
        self.mock_router.route.assert_called_once_with(
            command="open calculator",
            source="voice",
            context={"session_id": result.session_id},
        )
        self.mock_tts.synthesize.assert_called_once_with("Command executed successfully.")

        # Verify Memory recording
        self.assertEqual(self.memory.get_last().content, "Command executed successfully.")

    def test_command_extraction_variations(self) -> None:
        """Verify command extraction removes wake word cleanly."""
        extracted1 = VoicePipeline._extract_command("Hey Jarvis, play jazz music", "Hey Jarvis")
        self.assertEqual(extracted1, "play jazz music")

        extracted2 = VoicePipeline._extract_command("activate jarvis: what is the weather", "activate jarvis")
        self.assertEqual(extracted2, "what is the weather")

        extracted3 = VoicePipeline._extract_command("hey jarvis", "hey jarvis")
        self.assertEqual(extracted3, "")


class TestVoicePipelineErrorHandling(unittest.TestCase):
    """Tests for fault isolation and failure lifecycle event publication."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_router = MagicMock(spec=CommandRouter)
        self.mock_tts = MagicMock(spec=TextToSpeech)

        self.pipeline = VoicePipeline(
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        self.pipeline.stop()
        self.container.clear()
        self.event_bus.clear()

    def test_stt_failure_emits_failed_event(self) -> None:
        """Verify STT transcription failure publishes voice.failed and returns failure result."""
        self.mock_stt.transcribe.side_effect = RuntimeError("Whisper model crashed.")

        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_VOICE_FAILED, lambda e: failed_events.append(e))

        result = self.pipeline.process_audio(Path("/tmp/corrupt.wav"))

        self.assertFalse(result.success)
        self.assertIn("Whisper model crashed", str(result.error))
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["error_type"], "RuntimeError")

    def test_router_failure_emits_failed_event(self) -> None:
        """Verify router execution failure publishes voice.failed and returns failure result."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Hey Jarvis fail now", language="en"
        )
        self.mock_router.route.side_effect = ValueError("Skill execution failed.")

        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_VOICE_FAILED, lambda e: failed_events.append(e))

        result = self.pipeline.process_audio(Path("/tmp/test.wav"))

        self.assertFalse(result.success)
        self.assertIn("Skill execution failed", str(result.error))
        self.assertEqual(len(failed_events), 1)

    def test_audio_provider_start_failure(self) -> None:
        """Verify failure in provider.start raises AudioProviderError."""
        failing_provider = MagicMock(spec=AudioProvider)
        failing_provider.start.side_effect = RuntimeError("Microphone hardware disconnected.")

        pipeline = VoicePipeline(
            audio_provider=failing_provider,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        with self.assertRaises(AudioProviderError):
            pipeline.start()
        self.assertFalse(pipeline.is_running)


class TestVoicePipelineAsync(unittest.IsolatedAsyncioTestCase):
    """Asynchronous pipeline orchestration tests."""

    async def asyncSetUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.provider = PlaceholderAudioProvider()

        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_stt.transcribe_async = AsyncMock()

        self.mock_router = MagicMock(spec=CommandRouter)
        self.mock_router.route_async = AsyncMock()

        self.mock_tts = MagicMock(spec=TextToSpeech)
        self.mock_tts.synthesize_async = AsyncMock()

        self.wake_detector = WakeWordDetector(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        self.pipeline = VoicePipeline(
            audio_provider=self.provider,
            stt=self.mock_stt,
            wake_word_detector_instance=self.wake_detector,
            router=self.mock_router,
            tts=self.mock_tts,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    async def asyncTearDown(self) -> None:
        await self.pipeline.stop_async()
        self.container.clear()
        self.event_bus.clear()

    async def test_async_start_and_stop(self) -> None:
        """Verify async start and stop transitions."""
        events: list[Event] = []
        self.event_bus.subscribe("*", lambda e: events.append(e))

        await self.pipeline.start_async()
        self.assertTrue(self.pipeline.is_running)
        self.assertEqual(self.pipeline.state, VoiceState.LISTENING)

        await self.pipeline.stop_async()
        self.assertFalse(self.pipeline.is_running)
        self.assertEqual(self.pipeline.state, VoiceState.IDLE)

        event_names = [e.name for e in events]
        self.assertIn(EVENT_VOICE_STARTED, event_names)
        self.assertIn(EVENT_VOICE_STOPPED, event_names)

    async def test_async_process_audio(self) -> None:
        """Verify asynchronous end-to-end processing."""
        self.mock_stt.transcribe_async.return_value = TranscriptionResult(
            text="Hey Jarvis tell me a joke", language="en"
        )
        self.mock_router.route_async.return_value = "Why did the computer cross the road? To get to the other byte."
        self.mock_tts.synthesize_async.return_value = SpeechResult(
            text="Why did the computer cross the road? To get to the other byte.",
            audio_path=Path("/tmp/joke.mp3"),
            voice="en-GB-RyanNeural",
            file_size_bytes=600,
        )

        audio_file = Path("/tmp/joke_request.wav")
        self.provider.enqueue_audio(audio_file)
        await self.pipeline.start_async()

        result = await self.pipeline.process_audio_async()

        self.assertTrue(result.success)
        self.assertEqual(result.command_text, "tell me a joke")
        self.assertIn("other byte", result.command_result)
        self.assertIsNotNone(result.speech_result)


class TestVoicePipelineContainerIntegration(unittest.TestCase):
    """Tests for Service Container dependency registration and discovery."""

    def test_auto_register_in_container(self) -> None:
        """Verify pipeline registers 'voice_pipeline' and 'voice' singletons."""
        custom_container = ServiceContainer()
        pipeline = VoicePipeline(
            container_instance=custom_container,
            auto_register_in_container=True,
        )

        self.assertTrue(custom_container.exists("voice_pipeline"))
        self.assertTrue(custom_container.exists("voice"))
        self.assertIs(custom_container.resolve("voice_pipeline"), pipeline)
        self.assertIs(custom_container.resolve("voice"), pipeline)

    def test_global_singleton_export(self) -> None:
        """Verify global voice_pipeline singleton export."""
        self.assertIsInstance(voice_pipeline, VoicePipeline)


if __name__ == "__main__":
    unittest.main()
