"""Comprehensive unit and integration tests for MicrophoneRecorder."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
import wave

from app.core.config import Settings, load_settings
from app.core.container import JarvisException, ServiceContainer
from app.core.event_bus import EventBus
from app.voice.microphone import (
    DEFAULT_CHANNELS,
    DEFAULT_RECORD_DURATION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    EVENT_MICROPHONE_RECORDED,
    EVENT_MICROPHONE_RECORDING,
    EVENT_MICROPHONE_STARTED,
    EVENT_MICROPHONE_STOPPED,
    EVENT_SOURCE_MICROPHONE,
    MicrophoneRecorder,
    microphone,
    microphone_recorder,
)
from app.voice.models import (
    AudioProvider,
    AudioProviderError,
    MicrophoneError,
    RecordingError,
    VoicePipelineError,
)
from app.voice.pipeline import VoicePipeline


class TestMicrophoneModelsAndExceptions(unittest.TestCase):
    """Verify exception hierarchy for microphone errors."""

    def test_exception_inheritance(self) -> None:
        """MicrophoneError and RecordingError must inherit from AudioProviderError and JarvisException."""
        self.assertTrue(issubclass(MicrophoneError, AudioProviderError))
        self.assertTrue(issubclass(RecordingError, MicrophoneError))
        self.assertTrue(issubclass(MicrophoneError, VoicePipelineError))
        self.assertTrue(issubclass(MicrophoneError, JarvisException))


class TestMicrophoneRecorderLifecycle(unittest.TestCase):
    """Unit tests for MicrophoneRecorder lifecycle and parameters."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.settings = load_settings()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_initialization(self) -> None:
        """Verify default parameters match Whisper/STT requirements."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.assertIsInstance(recorder, AudioProvider)
        self.assertEqual(recorder.sample_rate, 16000)
        self.assertEqual(recorder.channels, 1)
        self.assertEqual(recorder.sample_width, 2)
        self.assertEqual(recorder.record_duration, 3.0)
        self.assertEqual(recorder.output_dir, self.output_dir)
        self.assertFalse(recorder.is_active())

    def test_invalid_parameter_validation(self) -> None:
        """Non-positive parameters must raise ValueError."""
        with self.assertRaises(ValueError):
            MicrophoneRecorder(sample_rate=0)
        with self.assertRaises(ValueError):
            MicrophoneRecorder(channels=-1)
        with self.assertRaises(ValueError):
            MicrophoneRecorder(sample_width=0)
        with self.assertRaises(ValueError):
            MicrophoneRecorder(record_duration=-2.0)

        recorder = MicrophoneRecorder(auto_register_in_container=False)
        with self.assertRaises(ValueError):
            recorder.record_duration = 0.0

    def test_start_stop_lifecycle_and_events(self) -> None:
        """Verify start and stop toggle is_active and publish events."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        events_received: list[str] = []

        def on_started(event: Any) -> None:
            events_received.append(event.name)

        def on_stopped(event: Any) -> None:
            events_received.append(event.name)

        self.event_bus.subscribe(EVENT_MICROPHONE_STARTED, on_started)
        self.event_bus.subscribe(EVENT_MICROPHONE_STOPPED, on_stopped)

        self.assertFalse(recorder.is_active())
        recorder.start()
        self.assertTrue(recorder.is_active())

        # Starting again when already active should be idempotent
        recorder.start()
        self.assertTrue(recorder.is_active())

        recorder.stop()
        self.assertFalse(recorder.is_active())

        # Stopping again when already stopped should be idempotent
        recorder.stop()
        self.assertFalse(recorder.is_active())

        self.assertIn(EVENT_MICROPHONE_STARTED, events_received)
        self.assertIn(EVENT_MICROPHONE_STOPPED, events_received)

    def test_context_manager_support(self) -> None:
        """Verify context manager activates and deactivates recorder."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.assertFalse(recorder.is_active())
        with recorder as rec:
            self.assertIs(rec, recorder)
            self.assertTrue(recorder.is_active())
        self.assertFalse(recorder.is_active())


class TestMicrophoneRecording(unittest.TestCase):
    """Test audio recording and WAV formatting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.container = ServiceContainer()
        self.event_bus = EventBus()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_creates_valid_wav_file(self) -> None:
        """Verify record() produces a valid 16kHz mono 16-bit PCM WAV file."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        duration = 0.5
        out_path = recorder.record(duration=duration)

        self.assertTrue(out_path.exists())
        self.assertTrue(out_path.is_file())
        self.assertEqual(out_path.suffix.lower(), ".wav")

        # Validate WAV structure
        with wave.open(str(out_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            expected_frames = int(16000 * duration)
            self.assertEqual(wf.getnframes(), expected_frames)

    def test_record_explicit_destination(self) -> None:
        """Verify record() saves directly to specified path."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            auto_register_in_container=False,
        )
        dest = self.output_dir / "custom_sub" / "my_recording.wav"
        out_path = recorder.record(duration=0.2, output_path=dest)

        self.assertEqual(out_path, dest.resolve())
        self.assertTrue(dest.exists())

    def test_custom_audio_source_callback(self) -> None:
        """Verify custom audio generator callback is used during recording."""
        custom_frames = b"\x12\x34" * 1600  # 1600 frames of dummy data

        def dummy_audio_source(num_frames: int) -> bytes:
            return b"\x01\x02" * num_frames

        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            audio_source_callback=dummy_audio_source,
            auto_register_in_container=False,
        )
        out_path = recorder.record(duration=0.1)
        with wave.open(str(out_path), "rb") as wf:
            data = wf.readframes(wf.getnframes())
            self.assertTrue(data.startswith(b"\x01\x02"))

    def test_capture_audio_active_vs_inactive(self) -> None:
        """capture_audio returns None when inactive, and a Path when active."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            auto_register_in_container=False,
        )
        # 1. Inactive
        self.assertFalse(recorder.is_active())
        result = recorder.capture_audio(duration=0.1)
        self.assertIsNone(result)

        # 2. Active
        recorder.start()
        result = recorder.capture_audio(duration=0.1)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        recorder.stop()

    def test_capture_audio_async(self) -> None:
        """capture_audio_async works asynchronously."""
        async def run_async_test() -> None:
            recorder = MicrophoneRecorder(
                output_dir=self.output_dir,
                auto_register_in_container=False,
            )
            # Inactive
            self.assertIsNone(await recorder.capture_audio_async(duration=0.1))

            # Active
            recorder.start()
            path = await recorder.capture_audio_async(duration=0.1)
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            recorder.stop()

        asyncio.run(run_async_test())

    def test_record_invalid_duration_raises_microphone_error(self) -> None:
        """Passing duration <= 0 raises MicrophoneError."""
        recorder = MicrophoneRecorder(output_dir=self.output_dir, auto_register_in_container=False)
        with self.assertRaises(MicrophoneError):
            recorder.record(duration=0.0)
        with self.assertRaises(MicrophoneError):
            recorder.record(duration=-1.0)


class TestMicrophoneServiceContainerAndPipelineIntegration(unittest.TestCase):
    """Verify ServiceContainer registration and VoicePipeline integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.settings = load_settings()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_registers_only_microphone_service(self) -> None:
        """MicrophoneRecorder must register only 'microphone' in the ServiceContainer."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            auto_register_in_container=True,
        )
        self.assertTrue(self.container.exists("microphone"))
        self.assertIs(self.container.resolve("microphone"), recorder)

        # Verify no unnecessary aliases were introduced
        self.assertFalse(self.container.exists("microphone_recorder"))
        self.assertFalse(self.container.exists("audio_provider"))
        self.assertFalse(self.container.exists("voice_stt"))
        self.assertFalse(self.container.exists("voice_tts"))

    def test_voice_pipeline_resolves_microphone_from_container(self) -> None:
        """VoicePipeline must resolve 'microphone' from container when no audio_provider is passed."""
        recorder = MicrophoneRecorder(
            output_dir=self.output_dir,
            container_instance=self.container,
            auto_register_in_container=True,
        )
        pipeline = VoicePipeline(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            config=self.settings,
            auto_register_in_container=True,
        )

        # Pipeline resolved recorder as its audio provider
        self.assertIs(pipeline.audio_provider, recorder)

        # Pipeline registered itself as 'voice_pipeline'
        self.assertTrue(self.container.exists("voice_pipeline"))
        self.assertIs(self.container.resolve("voice_pipeline"), pipeline)

    def test_exported_singletons(self) -> None:
        """Verify module exports microphone and microphone_recorder singletons."""
        self.assertIsInstance(microphone_recorder, MicrophoneRecorder)
        self.assertIs(microphone, microphone_recorder)


if __name__ == "__main__":
    unittest.main()
