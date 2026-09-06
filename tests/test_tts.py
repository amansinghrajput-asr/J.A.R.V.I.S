"""Comprehensive unit tests for the J.A.R.V.I.S Text-to-Speech (TTS) Subsystem."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import Settings, load_settings
from app.core.constants import (
    DEFAULT_TTS_PITCH,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOICE_EN,
    DEFAULT_TTS_VOLUME,
)
from app.core.container import JarvisException, ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.tts import (
    AudioFileError,
    ConfigurationError,
    EmptyTextError,
    EVENT_SOURCE_TTS,
    EVENT_TTS_COMPLETED,
    EVENT_TTS_FAILED,
    EVENT_TTS_STARTED,
    SpeechResult,
    SynthesisError,
    TTSError,
    TextToSpeech,
    VoiceNotFoundError,
    text_to_speech,
)
from app.tts.speaker import format_percentage, format_pitch


class DummyCommunicate:
    """Mock Communicate object emulating edge-tts Communicate behavior."""

    def __init__(
        self,
        text: str,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.kwargs = kwargs

    async def save(self, audio_fname: str, metadata_fname: Any = None) -> None:
        """Simulate saving audio bytes to disk."""
        target_file = Path(audio_fname)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        # Write dummy MP3 header bytes
        target_file.write_bytes(b"\xff\xfb\x90\x44" + (b"\x00" * 128))


class DummyFailingCommunicate:
    """Mock Communicate object that fails during save."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def save(self, audio_fname: str, metadata_fname: Any = None) -> None:
        raise ConnectionError("Simulated edge-tts network failure.")


class TestTTSModels(unittest.TestCase):
    """Unit tests for TTS models and exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """Verify that all TTS exceptions inherit from TTSError and JarvisException."""
        self.assertTrue(issubclass(TTSError, JarvisException))
        self.assertTrue(issubclass(EmptyTextError, TTSError))
        self.assertTrue(issubclass(VoiceNotFoundError, TTSError))
        self.assertTrue(issubclass(SynthesisError, TTSError))
        self.assertTrue(issubclass(AudioFileError, TTSError))
        self.assertTrue(issubclass(ConfigurationError, TTSError))

    def test_speech_result_dataclass(self) -> None:
        """Test SpeechResult fields, initialization, and properties."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"dummy audio content")
            temp_path = Path(f.name)

        try:
            result = SpeechResult(
                text="Hello world",
                audio_path=str(temp_path),  # tests __post_init__ path normalization
                voice="en-GB-RyanNeural",
                rate="+10%",
                volume="+5%",
                pitch="+2Hz",
                duration=1.5,
                file_size_bytes=19,
                execution_time=0.25,
            )

            self.assertIsInstance(result.audio_path, Path)
            self.assertEqual(result.audio_path, temp_path.resolve())
            self.assertEqual(result.text, "Hello world")
            self.assertEqual(result.voice, "en-GB-RyanNeural")
            self.assertEqual(result.rate, "+10%")
            self.assertEqual(result.volume, "+5%")
            self.assertEqual(result.pitch, "+2Hz")
            self.assertEqual(result.duration, 1.5)
            self.assertEqual(result.file_size_bytes, 19)
            self.assertEqual(result.execution_time, 0.25)
            self.assertTrue(bool(result))

            data = result.to_dict()
            self.assertIsInstance(data, dict)
            self.assertEqual(data["text"], "Hello world")
            self.assertEqual(data["audio_path"], str(temp_path.resolve()))
            self.assertEqual(data["file_size_bytes"], 19)

            # Test falsy SpeechResult
            empty_result = SpeechResult(
                text="",
                audio_path=temp_path,
                voice="en-GB-RyanNeural",
                file_size_bytes=0,
            )
            self.assertFalse(bool(empty_result))
        finally:
            temp_path.unlink(missing_ok=True)


class TestTTSFormattingHelpers(unittest.TestCase):
    """Unit tests for percentage and pitch formatting helpers."""

    def test_format_percentage(self) -> None:
        """Verify format_percentage handles strings, numbers, and defaults."""
        self.assertEqual(format_percentage(None), "+0%")
        self.assertEqual(format_percentage(None, default="+5%"), "+5%")
        self.assertEqual(format_percentage(""), "+0%")
        self.assertEqual(format_percentage(0), "+0%")
        self.assertEqual(format_percentage(10), "+10%")
        self.assertEqual(format_percentage(-15), "-15%")
        self.assertEqual(format_percentage("+20%"), "+20%")
        self.assertEqual(format_percentage("-5%"), "-5%")
        self.assertEqual(format_percentage("25%"), "+25%")
        self.assertEqual(format_percentage("30"), "+30%")

        with self.assertRaises(ConfigurationError):
            format_percentage("invalid_percent")

        with self.assertRaises(ConfigurationError):
            format_percentage("++5%")

    def test_format_pitch(self) -> None:
        """Verify format_pitch handles strings, numbers, and defaults."""
        self.assertEqual(format_pitch(None), "+0Hz")
        self.assertEqual(format_pitch(None, default="+10Hz"), "+10Hz")
        self.assertEqual(format_pitch(""), "+0Hz")
        self.assertEqual(format_pitch(0), "+0Hz")
        self.assertEqual(format_pitch(5), "+5Hz")
        self.assertEqual(format_pitch(-10), "-10Hz")
        self.assertEqual(format_pitch("+8Hz"), "+8Hz")
        self.assertEqual(format_pitch("-3hz"), "-3Hz")
        self.assertEqual(format_pitch("15Hz"), "+15Hz")
        self.assertEqual(format_pitch("20"), "+20Hz")

        with self.assertRaises(ConfigurationError):
            format_pitch("invalid_pitch")


class TestTextToSpeechLifecycle(unittest.TestCase):
    """Unit tests for TextToSpeech initialization, configuration, and dependencies."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.custom_container = ServiceContainer()
        self.custom_event_bus = EventBus()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up test environment."""
        self.temp_dir.cleanup()
        self.custom_container.clear()
        self.custom_event_bus.clear()

    def test_default_initialization(self) -> None:
        """Verify default initialization parameters and properties."""
        tts = TextToSpeech(
            container_instance=self.custom_container,
            event_bus_instance=self.custom_event_bus,
            temp_dir=self.temp_path,
            auto_register_in_container=True,
        )

        self.assertEqual(tts.voice, DEFAULT_TTS_VOICE_EN)
        self.assertEqual(tts.rate, DEFAULT_TTS_RATE)
        self.assertEqual(tts.volume, DEFAULT_TTS_VOLUME)
        self.assertEqual(tts.pitch, DEFAULT_TTS_PITCH)
        self.assertEqual(tts.temp_dir, self.temp_path.resolve())
        self.assertFalse(tts.is_initialized)

        # Service container registration check
        self.assertTrue(self.custom_container.exists("tts"))
        self.assertTrue(self.custom_container.exists("text_to_speech"))
        self.assertIs(self.custom_container.resolve("tts"), tts)

    def test_custom_initialization_parameters(self) -> None:
        """Verify explicit voice, rate, volume, and pitch parameters."""
        tts = TextToSpeech(
            voice="hi-IN-MadhurNeural",
            rate=15,
            volume=-10,
            pitch=5,
            container_instance=self.custom_container,
            event_bus_instance=self.custom_event_bus,
            temp_dir=self.temp_path,
            auto_register_in_container=False,
        )

        self.assertEqual(tts.voice, "hi-IN-MadhurNeural")
        self.assertEqual(tts.rate, "+15%")
        self.assertEqual(tts.volume, "-10%")
        self.assertEqual(tts.pitch, "+5Hz")
        self.assertFalse(self.custom_container.exists("tts"))

    def test_setters_and_validation(self) -> None:
        """Verify set_voice, set_rate, set_volume, set_pitch and error checking."""
        tts = TextToSpeech(
            container_instance=self.custom_container,
            event_bus_instance=self.custom_event_bus,
            temp_dir=self.temp_path,
            auto_register_in_container=False,
        )

        tts.set_voice("en-US-GuyNeural")
        self.assertEqual(tts.voice, "en-US-GuyNeural")

        tts.set_rate("+25%")
        self.assertEqual(tts.rate, "+25%")

        tts.set_volume("-5%")
        self.assertEqual(tts.volume, "-5%")

        tts.set_pitch("+10Hz")
        self.assertEqual(tts.pitch, "+10Hz")

        # Validation errors
        with self.assertRaises(ConfigurationError):
            tts.set_voice("")

        with self.assertRaises(ConfigurationError):
            tts.set_voice("   ")

        with self.assertRaises(ConfigurationError):
            tts.set_rate("bad_rate")

        with self.assertRaises(ConfigurationError):
            tts.set_volume("bad_volume")

        with self.assertRaises(ConfigurationError):
            tts.set_pitch("bad_pitch")

    def test_lazy_initialization(self) -> None:
        """Verify lazy initialization behavior."""
        sub_dir = self.temp_path / "lazy_created_dir"
        self.assertFalse(sub_dir.exists())

        tts = TextToSpeech(
            container_instance=self.custom_container,
            event_bus_instance=self.custom_event_bus,
            temp_dir=sub_dir,
            auto_register_in_container=False,
        )

        self.assertFalse(tts.is_initialized)
        self.assertFalse(sub_dir.exists())

        # Calling initialize directly
        tts.initialize()
        self.assertTrue(tts.is_initialized)
        self.assertTrue(sub_dir.exists())

        # Calling initialize again is idempotent
        tts.initialize()
        self.assertTrue(tts.is_initialized)

    def test_global_singleton_export(self) -> None:
        """Verify that default text_to_speech singleton is exported."""
        self.assertIsInstance(text_to_speech, TextToSpeech)


class TestTextToSpeechSynthesis(unittest.TestCase):
    """Unit tests for synchronous and asynchronous synthesis workflows."""

    def setUp(self) -> None:
        """Set up test environment with mock edge-tts Communicate."""
        self.custom_container = ServiceContainer()
        self.custom_event_bus = EventBus()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.tts = TextToSpeech(
            container_instance=self.custom_container,
            event_bus_instance=self.custom_event_bus,
            temp_dir=self.temp_path,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        """Clean up test environment."""
        self.temp_dir.cleanup()
        self.custom_container.clear()
        self.custom_event_bus.clear()

    @patch("edge_tts.Communicate", side_effect=DummyCommunicate)
    def test_synthesize_to_temporary_file(self, mock_comm: MagicMock) -> None:
        """Test synchronous synthesis with default temporary file output."""
        received_events: list[Event] = []
        self.custom_event_bus.subscribe("*", lambda e: received_events.append(e))

        result = self.tts.synthesize("Hello from JARVIS")

        self.assertIsInstance(result, SpeechResult)
        self.assertEqual(result.text, "Hello from JARVIS")
        self.assertEqual(result.voice, DEFAULT_TTS_VOICE_EN)
        self.assertEqual(result.rate, DEFAULT_TTS_RATE)
        self.assertEqual(result.volume, DEFAULT_TTS_VOLUME)
        self.assertEqual(result.pitch, DEFAULT_TTS_PITCH)
        self.assertTrue(result.audio_path.exists())
        self.assertGreater(result.file_size_bytes, 0)
        self.assertGreater(result.duration, 0.0)
        self.assertTrue(bool(result))

        # Verify lifecycle events
        event_names = [e.name for e in received_events]
        self.assertIn(EVENT_TTS_STARTED, event_names)
        self.assertIn(EVENT_TTS_COMPLETED, event_names)
        self.assertNotIn(EVENT_TTS_FAILED, event_names)

        started_evt = next(e for e in received_events if e.name == EVENT_TTS_STARTED)
        self.assertEqual(started_evt.source, EVENT_SOURCE_TTS)
        self.assertEqual(started_evt.payload["text"], "Hello from JARVIS")

        completed_evt = next(e for e in received_events if e.name == EVENT_TTS_COMPLETED)
        self.assertEqual(completed_evt.source, EVENT_SOURCE_TTS)
        self.assertEqual(completed_evt.payload["text"], "Hello from JARVIS")
        self.assertEqual(completed_evt.payload["audio_path"], str(result.audio_path))

    @patch("edge_tts.Communicate", side_effect=DummyCommunicate)
    def test_synthesize_to_explicit_output_path(self, mock_comm: MagicMock) -> None:
        """Test synthesis saving directly to a specified destination path."""
        target_path = self.temp_path / "custom" / "greeting.mp3"
        self.assertFalse(target_path.exists())

        result = self.tts.synthesize("System is operational", output_path=target_path)

        self.assertEqual(result.audio_path, target_path.resolve())
        self.assertTrue(target_path.exists())
        self.assertGreater(target_path.stat().st_size, 0)

    @patch("edge_tts.Communicate", side_effect=DummyCommunicate)
    def test_synthesize_with_per_call_overrides(self, mock_comm: MagicMock) -> None:
        """Test overriding voice, rate, volume, and pitch in a single call."""
        result = self.tts.synthesize(
            text="Override test",
            voice="hi-IN-MadhurNeural",
            rate="+20%",
            volume="-10%",
            pitch="+5Hz",
        )

        self.assertEqual(result.voice, "hi-IN-MadhurNeural")
        self.assertEqual(result.rate, "+20%")
        self.assertEqual(result.volume, "-10%")
        self.assertEqual(result.pitch, "+5Hz")

        # Verify instance defaults were not mutated
        self.assertEqual(self.tts.voice, DEFAULT_TTS_VOICE_EN)
        self.assertEqual(self.tts.rate, DEFAULT_TTS_RATE)

    def test_synthesize_empty_text_error(self) -> None:
        """Test that empty or whitespace-only text raises EmptyTextError."""
        with self.assertRaises(EmptyTextError):
            self.tts.synthesize("")

        with self.assertRaises(EmptyTextError):
            self.tts.synthesize("    ")

        with self.assertRaises(EmptyTextError):
            self.tts.synthesize(None)  # type: ignore[arg-type]

    @patch("edge_tts.Communicate", side_effect=DummyFailingCommunicate)
    def test_synthesize_failure_events_and_exception(self, mock_comm: MagicMock) -> None:
        """Test that synthesis failure publishes tts.failed and raises SynthesisError."""
        received_events: list[Event] = []
        self.custom_event_bus.subscribe("*", lambda e: received_events.append(e))

        with self.assertRaises(SynthesisError):
            self.tts.synthesize("This will fail")

        event_names = [e.name for e in received_events]
        self.assertIn(EVENT_TTS_STARTED, event_names)
        self.assertIn(EVENT_TTS_FAILED, event_names)
        self.assertNotIn(EVENT_TTS_COMPLETED, event_names)

        failed_evt = next(e for e in received_events if e.name == EVENT_TTS_FAILED)
        self.assertEqual(failed_evt.source, EVENT_SOURCE_TTS)
        self.assertEqual(failed_evt.payload["text"], "This will fail")
        self.assertIn("Simulated edge-tts network failure", failed_evt.payload["error"])

    @patch("edge_tts.Communicate", side_effect=DummyCommunicate)
    def test_synthesize_publish_event_false(self, mock_comm: MagicMock) -> None:
        """Test synthesis without publishing events when publish_event=False."""
        received_events: list[Event] = []
        self.custom_event_bus.subscribe("*", lambda e: received_events.append(e))

        result = self.tts.synthesize("Silent run", publish_event=False)
        self.assertIsInstance(result, SpeechResult)
        self.assertEqual(len(received_events), 0)

    def test_synthesize_async(self) -> None:
        """Test asynchronous synthesize_async API."""
        async def _test() -> None:
            with patch("edge_tts.Communicate", side_effect=DummyCommunicate):
                result = await self.tts.synthesize_async("Async test message")
                self.assertIsInstance(result, SpeechResult)
                self.assertEqual(result.text, "Async test message")
                self.assertTrue(result.audio_path.exists())

        asyncio.run(_test())

    def test_cleanup_temp_files(self) -> None:
        """Test temporary audio file cleanup utility."""
        with patch("edge_tts.Communicate", side_effect=DummyCommunicate):
            r1 = self.tts.synthesize("Speech one")
            r2 = self.tts.synthesize("Speech two")

            self.assertTrue(r1.audio_path.exists())
            self.assertTrue(r2.audio_path.exists())

            deleted_count = self.tts.cleanup_temp_files()
            self.assertGreaterEqual(deleted_count, 2)
            self.assertFalse(r1.audio_path.exists())
            self.assertFalse(r2.audio_path.exists())

    @patch("edge_tts.list_voices", new_callable=AsyncMock)
    def test_list_and_get_available_voices(self, mock_list_voices: AsyncMock) -> None:
        """Test querying available voices from edge-tts."""
        mock_list_voices.return_value = [
            {"Name": "Microsoft Server Speech (en-GB, RyanNeural)", "ShortName": "en-GB-RyanNeural"},
            {"Name": "Microsoft Server Speech (hi-IN, MadhurNeural)", "ShortName": "hi-IN-MadhurNeural"},
        ]

        voices = self.tts.get_available_voices()
        self.assertEqual(len(voices), 2)
        self.assertEqual(voices[0]["ShortName"], "en-GB-RyanNeural")

        # Test error handling when list_voices fails
        mock_list_voices.side_effect = RuntimeError("Network error")
        error_voices = self.tts.get_available_voices()
        self.assertEqual(error_voices, [])


class TestTTSThreadSafety(unittest.TestCase):
    """Unit tests verifying thread safety under concurrent execution."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.tts = TextToSpeech(
            temp_dir=self.temp_path,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch("edge_tts.Communicate", side_effect=DummyCommunicate)
    def test_concurrent_synthesis(self, mock_comm: MagicMock) -> None:
        """Verify that multiple threads can synthesize speech concurrently."""
        num_threads = 10
        phrases = [f"Concurrent speech task {i}" for i in range(num_threads)]
        results: list[SpeechResult] = []

        def _worker(phrase: str) -> SpeechResult:
            return self.tts.synthesize(phrase)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_worker, p) for p in phrases]
            for future in as_completed(futures):
                results.append(future.result())

        self.assertEqual(len(results), num_threads)
        for res in results:
            self.assertTrue(res.audio_path.exists())
            self.assertGreater(res.file_size_bytes, 0)

    def test_concurrent_property_updates(self) -> None:
        """Verify thread-safe reading and updating of parameters."""
        num_threads = 20

        def _writer(i: int) -> None:
            if i % 4 == 0:
                self.tts.set_voice(f"voice-{i}")
            elif i % 4 == 1:
                self.tts.set_rate(i)
            elif i % 4 == 2:
                self.tts.set_volume(i)
            else:
                self.tts.set_pitch(i)

        def _reader() -> None:
            _ = self.tts.voice
            _ = self.tts.rate
            _ = self.tts.volume
            _ = self.tts.pitch
            _ = self.tts.is_initialized

        threads: list[threading.Thread] = []
        for i in range(num_threads):
            t_write = threading.Thread(target=_writer, args=(i,))
            t_read = threading.Thread(target=_reader)
            threads.extend([t_write, t_read])

        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    unittest.main()
