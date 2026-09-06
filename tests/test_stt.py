"""Comprehensive unit tests for the J.A.R.V.I.S Speech-to-Text (STT) Subsystem."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from app.core.config import Settings, load_settings
from app.core.container import JarvisException, ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.stt import (
    AudioFileNotFoundError,
    EVENT_SOURCE_STT,
    EVENT_STT_COMPLETED,
    EVENT_STT_FAILED,
    EVENT_STT_STARTED,
    InvalidAudioFormatError,
    ModelLoadError,
    STTError,
    SpeechToText,
    TranscriptionError,
    TranscriptionResult,
    speech_to_text,
)


class DummySegment:
    """Mock Segment structure returned by faster-whisper."""

    def __init__(
        self,
        text: str,
        start: float = 0.0,
        end: float = 1.0,
        avg_logprob: float = -0.2,
        compression_ratio: float = 1.2,
        no_speech_prob: float = 0.05,
    ) -> None:
        self.id = 0
        self.seek = 0
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob


class DummyTranscriptionInfo:
    """Mock TranscriptionInfo structure returned by faster-whisper."""

    def __init__(
        self,
        language: str = "en",
        language_probability: float = 0.98,
        duration: float = 2.5,
    ) -> None:
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


def create_test_wav_file(file_path: Path, duration_seconds: float = 0.1) -> Path:
    """Create a minimal valid PCM WAV audio file for testing."""
    sample_rate = 16000
    num_frames = int(sample_rate * duration_seconds)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return file_path


class TestSTTModelsAndExceptions(unittest.TestCase):
    """Test suite for STT exceptions and TranscriptionResult dataclass."""

    def test_exception_hierarchy(self) -> None:
        """Verify STT exception hierarchy roots at JarvisException."""
        self.assertTrue(issubclass(STTError, JarvisException))
        self.assertTrue(issubclass(AudioFileNotFoundError, STTError))
        self.assertTrue(issubclass(InvalidAudioFormatError, STTError))
        self.assertTrue(issubclass(ModelLoadError, STTError))
        self.assertTrue(issubclass(TranscriptionError, STTError))

    def test_transcription_result_attributes(self) -> None:
        """Verify TranscriptionResult stores all transcription metadata."""
        now = time.time()
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            language_probability=0.99,
            duration=3.2,
            segments=({"start": 0.0, "end": 1.5, "text": "Hello world"},),
            model_name="base",
            execution_time=0.45,
            timestamp=now,
        )
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.language_probability, 0.99)
        self.assertEqual(result.duration, 3.2)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.model_name, "base")
        self.assertEqual(result.execution_time, 0.45)
        self.assertEqual(result.timestamp, now)

    def test_transcription_result_boolean_evaluation(self) -> None:
        """Verify __bool__ returns True only when transcribed text is non-empty."""
        res_valid = TranscriptionResult(text="Testing", language="en")
        self.assertTrue(bool(res_valid))

        res_empty = TranscriptionResult(text="", language="en")
        self.assertFalse(bool(res_empty))

        res_whitespace = TranscriptionResult(text="   \n\t  ", language="en")
        self.assertFalse(bool(res_whitespace))

    def test_transcription_result_to_dict(self) -> None:
        """Verify serialization to dictionary."""
        result = TranscriptionResult(
            text="Jarvis online",
            language="en",
            duration=1.5,
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["text"], "Jarvis online")
        self.assertEqual(d["language"], "en")
        self.assertEqual(d["duration"], 1.5)
        self.assertEqual(d["model_name"], "base")
        self.assertIn("timestamp", d)


class TestSpeechToTextFileValidation(unittest.TestCase):
    """Test suite for file existence and format validation."""

    def setUp(self) -> None:
        """Set up isolated transcriber and temp directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.stt = SpeechToText(auto_register_in_container=False)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_nonexistent_file_raises_audio_file_not_found(self) -> None:
        """Verify AudioFileNotFoundError is raised when file does not exist."""
        nonexistent = self.temp_path / "missing.wav"
        with self.assertRaises(AudioFileNotFoundError):
            self.stt.validate_audio_file(nonexistent)

    def test_none_path_raises_audio_file_not_found(self) -> None:
        """Verify AudioFileNotFoundError is raised when None is passed."""
        with self.assertRaises(AudioFileNotFoundError):
            self.stt.validate_audio_file(None)  # type: ignore[arg-type]

    def test_directory_raises_audio_file_not_found(self) -> None:
        """Verify AudioFileNotFoundError is raised when path is a directory."""
        with self.assertRaises(AudioFileNotFoundError):
            self.stt.validate_audio_file(self.temp_path)

    def test_unsupported_extensions_raise_invalid_audio_format(self) -> None:
        """Verify InvalidAudioFormatError is raised for non-.wav formats."""
        for ext in (".mp3", ".ogg", ".flac", ".txt", ".m4a"):
            fake_file = self.temp_path / f"audio{ext}"
            fake_file.write_text("dummy")
            with self.subTest(extension=ext):
                with self.assertRaises(InvalidAudioFormatError):
                    self.stt.validate_audio_file(fake_file)

    def test_valid_wav_file_succeeds(self) -> None:
        """Verify a valid existing .wav file is accepted and resolved."""
        wav_file = self.temp_path / "sample.wav"
        create_test_wav_file(wav_file)
        resolved = self.stt.validate_audio_file(wav_file)
        self.assertEqual(resolved, wav_file.resolve())


class TestSpeechToTextModelManagement(unittest.TestCase):
    """Test suite for model loading, caching, configuration, and CPU mode."""

    def setUp(self) -> None:
        """Set up isolated transcriber."""
        self.test_container = ServiceContainer()
        self.test_bus = EventBus()
        self.stt = SpeechToText(
            model_name="base",
            container_instance=self.test_container,
            event_bus_instance=self.test_bus,
            auto_register_in_container=False,
        )

    def test_lazy_model_loading(self) -> None:
        """Verify model is NOT loaded upon SpeechToText initialization."""
        self.assertFalse(self.stt.is_model_loaded)
        self.assertEqual(self.stt.model_name, "base")

    def test_enforced_cpu_mode(self) -> None:
        """Verify device is strictly CPU and compute type is configured."""
        self.assertEqual(self.stt.device, "cpu")
        self.assertEqual(self.stt.compute_type, "int8")

    @patch("faster_whisper.WhisperModel")
    def test_model_loading_and_caching(self, mock_whisper_class: MagicMock) -> None:
        """Verify model is loaded once and cached for subsequent calls."""
        mock_instance = MagicMock()
        mock_whisper_class.return_value = mock_instance

        # First call loads model
        loaded_model_1 = self.stt.load_model()
        self.assertTrue(self.stt.is_model_loaded)
        self.assertEqual(mock_whisper_class.call_count, 1)
        self.assertIs(loaded_model_1, mock_instance)

        # Second call returns cached model without recreating
        loaded_model_2 = self.stt.load_model()
        self.assertEqual(mock_whisper_class.call_count, 1)
        self.assertIs(loaded_model_2, mock_instance)

    @patch("faster_whisper.WhisperModel")
    def test_set_model_and_switch(self, mock_whisper_class: MagicMock) -> None:
        """Verify changing model name switches active model."""
        mock_instance_base = MagicMock()
        mock_instance_tiny = MagicMock()
        mock_whisper_class.side_effect = [mock_instance_base, mock_instance_tiny]

        self.stt.load_model("base")
        self.stt.set_model("tiny")
        self.assertEqual(self.stt.model_name, "tiny")
        self.assertFalse(self.stt.is_model_loaded)

        loaded_tiny = self.stt.load_model()
        self.assertTrue(self.stt.is_model_loaded)
        self.assertIs(loaded_tiny, mock_instance_tiny)
        self.assertEqual(mock_whisper_class.call_count, 2)

    def test_set_model_invalid_name_raises_value_error(self) -> None:
        """Verify invalid model names raise ValueError."""
        with self.assertRaises(ValueError):
            self.stt.set_model("")
        with self.assertRaises(ValueError):
            self.stt.set_model("   ")
        with self.assertRaises(ValueError):
            self.stt.set_model(None)  # type: ignore[arg-type]

    @patch("faster_whisper.WhisperModel")
    def test_clear_cache(self, mock_whisper_class: MagicMock) -> None:
        """Verify clear_cache empties memory cache."""
        mock_whisper_class.return_value = MagicMock()
        self.stt.load_model()
        self.assertTrue(self.stt.is_model_loaded)

        self.stt.clear_cache()
        self.assertFalse(self.stt.is_model_loaded)

    @patch("faster_whisper.WhisperModel", side_effect=RuntimeError("Model weights corrupted"))
    def test_model_load_failure_raises_model_load_error(
        self, mock_whisper_class: MagicMock
    ) -> None:
        """Verify ModelLoadError is raised when faster-whisper fails to instantiate."""
        with self.assertRaises(ModelLoadError) as ctx:
            self.stt.load_model("invalid-model")
        self.assertIn("Failed to load faster-whisper model", str(ctx.exception))


class TestSpeechToTextTranscription(unittest.TestCase):
    """Test suite for synchronous and asynchronous transcription."""

    def setUp(self) -> None:
        """Set up isolated transcriber and temp audio file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.wav_file = self.temp_path / "test.wav"
        create_test_wav_file(self.wav_file, duration_seconds=0.2)

        self.test_bus = EventBus()
        self.test_container = ServiceContainer()
        self.stt = SpeechToText(
            model_name="base",
            event_bus_instance=self.test_bus,
            container_instance=self.test_container,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    @patch("faster_whisper.WhisperModel")
    def test_sync_transcription_success(self, mock_whisper_class: MagicMock) -> None:
        """Verify synchronous transcription returns expected TranscriptionResult."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        mock_segments = [
            DummySegment("Hello Jarvis,", start=0.0, end=1.0),
            DummySegment("what is the weather?", start=1.0, end=2.0),
        ]
        mock_info = DummyTranscriptionInfo(language="en", language_probability=0.99, duration=2.0)
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        result = self.stt.transcribe(self.wav_file)

        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello Jarvis, what is the weather?")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(result.language_probability, 0.99)
        self.assertEqual(len(result.segments), 2)
        self.assertGreater(result.execution_time, 0.0)

    @patch("faster_whisper.WhisperModel")
    def test_async_transcription_success(self, mock_whisper_class: MagicMock) -> None:
        """Verify asynchronous transcription returns expected TranscriptionResult."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        mock_segments = [DummySegment("Testing async STT", start=0.0, end=1.0)]
        mock_info = DummyTranscriptionInfo(language="en", duration=1.0)
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        async def _run_async() -> TranscriptionResult:
            return await self.stt.transcribe_async(self.wav_file)

        result = asyncio.run(_run_async())
        self.assertEqual(result.text, "Testing async STT")
        self.assertEqual(result.language, "en")

    @patch("faster_whisper.WhisperModel")
    def test_transcription_failure_raises_transcription_error(
        self, mock_whisper_class: MagicMock
    ) -> None:
        """Verify TranscriptionError is raised when transcribe fails."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        mock_model.transcribe.side_effect = RuntimeError("Audio decoding stream error")

        with self.assertRaises(TranscriptionError) as ctx:
            self.stt.transcribe(self.wav_file)
        self.assertIn("Transcription failed", str(ctx.exception))


class TestSTTEventBusIntegration(unittest.TestCase):
    """Test suite for STT lifecycle events published to EventBus."""

    def setUp(self) -> None:
        """Set up isolated transcriber and EventBus."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.wav_file = self.temp_path / "event_test.wav"
        create_test_wav_file(self.wav_file)

        self.test_bus = EventBus()
        self.stt = SpeechToText(
            model_name="base",
            event_bus_instance=self.test_bus,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    @patch("faster_whisper.WhisperModel")
    def test_events_published_on_success(self, mock_whisper_class: MagicMock) -> None:
        """Verify stt.started and stt.completed events are published."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        mock_segments = [DummySegment("System online")]
        mock_info = DummyTranscriptionInfo(duration=1.2)
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        started_events: list[Event] = []
        completed_events: list[Event] = []

        self.test_bus.subscribe(EVENT_STT_STARTED, lambda e: started_events.append(e))
        self.test_bus.subscribe(EVENT_STT_COMPLETED, lambda e: completed_events.append(e))

        result = self.stt.transcribe(self.wav_file)

        # 1. Started event checks
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].name, EVENT_STT_STARTED)
        self.assertEqual(started_events[0].source, EVENT_SOURCE_STT)
        self.assertEqual(started_events[0].payload["audio_path"], str(self.wav_file))
        self.assertEqual(started_events[0].payload["model_name"], "base")

        # 2. Completed event checks
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].name, EVENT_STT_COMPLETED)
        self.assertEqual(completed_events[0].source, EVENT_SOURCE_STT)
        self.assertEqual(completed_events[0].payload["text"], "System online")
        self.assertEqual(completed_events[0].payload["result"], result)

    @patch("faster_whisper.WhisperModel")
    def test_events_published_on_failure(self, mock_whisper_class: MagicMock) -> None:
        """Verify stt.started and stt.failed events are published on error."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        mock_model.transcribe.side_effect = RuntimeError("Inference fault")

        started_events: list[Event] = []
        failed_events: list[Event] = []

        self.test_bus.subscribe(EVENT_STT_STARTED, lambda e: started_events.append(e))
        self.test_bus.subscribe(EVENT_STT_FAILED, lambda e: failed_events.append(e))

        with self.assertRaises(TranscriptionError):
            self.stt.transcribe(self.wav_file)

        self.assertEqual(len(started_events), 1)
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].name, EVENT_STT_FAILED)
        self.assertEqual(failed_events[0].source, EVENT_SOURCE_STT)
        self.assertEqual(failed_events[0].payload["error_type"], "TranscriptionError")
        self.assertIn("Inference fault", failed_events[0].payload["error"])

    @patch("faster_whisper.WhisperModel")
    def test_async_events_published(self, mock_whisper_class: MagicMock) -> None:
        """Verify async transcribe publishes lifecycle events."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        mock_segments = [DummySegment("Async event test")]
        mock_info = DummyTranscriptionInfo()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        received_events: list[str] = []
        self.test_bus.subscribe(EVENT_STT_STARTED, lambda e: received_events.append(e.name))
        self.test_bus.subscribe(EVENT_STT_COMPLETED, lambda e: received_events.append(e.name))

        async def _run() -> None:
            await self.stt.transcribe_async(self.wav_file)

        asyncio.run(_run())
        self.assertEqual(received_events, [EVENT_STT_STARTED, EVENT_STT_COMPLETED])

    @patch("faster_whisper.WhisperModel")
    def test_publish_event_false_suppresses_events(
        self, mock_whisper_class: MagicMock
    ) -> None:
        """Verify setting publish_event=False emits no events."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter([DummySegment("Silent")]),
            DummyTranscriptionInfo(),
        )

        all_events: list[Event] = []
        self.test_bus.subscribe("*", lambda e: all_events.append(e))

        self.stt.transcribe(self.wav_file, publish_event=False)
        self.assertEqual(len(all_events), 0)


class TestSTTContainerAndThreadSafety(unittest.TestCase):
    """Test suite for service container integration and concurrent thread safety."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.wav_file = self.temp_path / "thread_test.wav"
        create_test_wav_file(self.wav_file)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_container_registration(self) -> None:
        """Verify SpeechToText auto-registers into ServiceContainer."""
        test_container = ServiceContainer()
        instance = SpeechToText(
            container_instance=test_container,
            auto_register_in_container=True,
        )
        self.assertTrue(test_container.exists("stt"))
        self.assertTrue(test_container.exists("speech_to_text"))
        self.assertIs(test_container.resolve("stt"), instance)
        self.assertIs(test_container.resolve("speech_to_text"), instance)

    @patch("faster_whisper.WhisperModel")
    def test_concurrent_transcription_thread_safety(
        self, mock_whisper_class: MagicMock
    ) -> None:
        """Verify concurrent multi-threaded transcribe calls execute safely."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        call_counter = 0
        counter_lock = threading.Lock()

        def _mock_transcribe(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            nonlocal call_counter
            time.sleep(0.01)  # small pause to simulate processing and trigger race conditions
            with counter_lock:
                call_counter += 1
                curr = call_counter
            return (
                iter([DummySegment(f"Segment {curr}")]),
                DummyTranscriptionInfo(duration=0.5),
            )

        mock_model.transcribe.side_effect = _mock_transcribe

        stt = SpeechToText(auto_register_in_container=False)

        num_threads = 10
        results: list[TranscriptionResult] = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(stt.transcribe, self.wav_file, publish_event=False)
                for _ in range(num_threads)
            ]
            for f in as_completed(futures):
                results.append(f.result())

        self.assertEqual(len(results), num_threads)
        for res in results:
            self.assertIn("Segment", res.text)
            self.assertEqual(res.duration, 0.5)

        # Model should only have been instantiated once
        self.assertEqual(mock_whisper_class.call_count, 1)

    def test_global_singleton_export(self) -> None:
        """Verify the module-level singleton instance is ready and valid."""
        self.assertIsInstance(speech_to_text, SpeechToText)
        self.assertEqual(speech_to_text.device, "cpu")


if __name__ == "__main__":
    unittest.main()
