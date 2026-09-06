"""Comprehensive unit tests for the production Wake Word Subsystem (openWakeWord)."""

from __future__ import annotations

import logging
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.voice.models import VoiceConversationResult
from app.wakeword.engine import (
    CHUNK_SIZE_SAMPLES,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_DETECTION_THRESHOLD,
    SAMPLE_RATE,
    WakeWordEngine,
    _resolve_model_path,
)
from app.wakeword.models import (
    AudioStreamError,
    ModelLoadError,
    WakeWordError,
    WakeWordResult,
)


class MockOpenWakeWordModel:
    """Mock openwakeword Model for deterministic unit testing."""

    def __init__(self, return_score: float = 0.0, model_key: str = "hey_jarvis_v0.1.onnx") -> None:
        self.return_score = return_score
        self.model_key = model_key
        self.models = {model_key: None}
        self.predict_calls: list[np.ndarray] = []

    def predict(self, audio_data: np.ndarray) -> dict[str, float]:
        self.predict_calls.append(audio_data)
        return {self.model_key: self.return_score}


class TestWakeWordEngineInitialization(unittest.TestCase):
    """Test suite for WakeWordEngine initialization and configuration."""

    def setUp(self) -> None:
        self.bus = EventBus()
        self.container = ServiceContainer()
        self.mock_model = MockOpenWakeWordModel(return_score=0.0)

    def test_default_wake_word_and_threshold(self) -> None:
        """Verify default wake word is 'Jarvis' and default threshold is 0.5."""
        engine = WakeWordEngine(
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )
        self.assertEqual(engine.wake_word, "jarvis")
        self.assertEqual(engine.threshold, DEFAULT_DETECTION_THRESHOLD)
        self.assertFalse(engine.is_running)
        self.assertFalse(engine.is_paused)

    def test_custom_wake_word_and_threshold(self) -> None:
        """Verify custom wake word and threshold are assigned correctly."""
        engine = WakeWordEngine(
            wake_word="Hey Jarvis",
            threshold=0.75,
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )
        self.assertEqual(engine.wake_word, "Hey Jarvis")
        self.assertEqual(engine.threshold, 0.75)

    def test_threshold_property_validation(self) -> None:
        """Verify threshold validation rejects values outside [0.0, 1.0]."""
        engine = WakeWordEngine(
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )
        engine.threshold = 0.8
        self.assertEqual(engine.threshold, 0.8)

        with self.assertRaises(ValueError):
            engine.threshold = -0.1

        with self.assertRaises(ValueError):
            engine.threshold = 1.1

    def test_container_registration(self) -> None:
        """Verify WakeWordEngine self-registers in the ServiceContainer."""
        engine = WakeWordEngine(
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=True,
        )
        self.assertTrue(self.container.exists("wakeword_engine"))
        self.assertTrue(self.container.exists("wake_word_engine"))
        self.assertTrue(self.container.exists("wakeword_listener"))
        self.assertTrue(self.container.exists("wake_word"))
        self.assertIs(self.container.resolve("wake_word_engine"), engine)

    def test_resolve_model_path_helper(self) -> None:
        """Verify _resolve_model_path resolves jarvis model name."""
        path = _resolve_model_path("Jarvis")
        self.assertTrue("hey_jarvis" in path or path.endswith(".onnx"))


class TestWakeWordPredictionAndAudioFeeding(unittest.TestCase):
    """Test suite for direct audio feeding and prediction logic."""

    def setUp(self) -> None:
        self.bus = EventBus()
        self.container = ServiceContainer()
        self.mock_model = MockOpenWakeWordModel(return_score=0.0)
        self.engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
            cooldown_seconds=0.1,
        )

    def test_feed_audio_silence_no_detection(self) -> None:
        """Verify feeding silent audio does not trigger detection."""
        silent_chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = self.engine.feed_audio(silent_chunk)

        self.assertFalse(result.detected)
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.matched_phrase)
        self.assertFalse(bool(result))

    def test_feed_audio_as_bytes(self) -> None:
        """Verify feeding audio as raw bytes works identical to numpy array."""
        raw_bytes = bytes(CHUNK_SIZE_SAMPLES * 2)  # 16-bit PCM silence
        result = self.engine.feed_audio(raw_bytes)

        self.assertFalse(result.detected)
        self.assertEqual(len(self.mock_model.predict_calls), 1)

    def test_feed_audio_invalid_type_raises_error(self) -> None:
        """Verify feeding invalid data type raises AudioStreamError."""
        with self.assertRaises(AudioStreamError):
            self.engine.feed_audio("not_audio_data")  # type: ignore

    def test_feed_audio_positive_detection(self) -> None:
        """Verify detection occurs when model prediction exceeds threshold."""
        self.mock_model.return_score = 0.85
        captured_events: list[Event] = []
        self.bus.subscribe("wakeword.detected", lambda ev: captured_events.append(ev))

        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = self.engine.feed_audio(chunk)

        self.assertTrue(result.detected)
        self.assertEqual(result.matched_phrase, "Jarvis")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.model_name, "hey_jarvis_v0.1.onnx")

        # Verify wakeword.detected event published
        self.assertEqual(len(captured_events), 1)
        ev = captured_events[0]
        self.assertEqual(ev.name, "wakeword.detected")
        self.assertEqual(ev.payload["wake_word"], "Jarvis")
        self.assertEqual(ev.payload["score"], 0.85)
        self.assertEqual(ev.source, "wakeword")

    def test_cooldown_suppresses_rapid_retrigger(self) -> None:
        """Verify cooldown suppresses rapid repeated detections."""
        self.mock_model.return_score = 0.9
        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)

        # First trigger succeeds
        res1 = self.engine.feed_audio(chunk)
        self.assertTrue(res1.detected)

        # Immediate second trigger is suppressed by cooldown
        res2 = self.engine.feed_audio(chunk)
        self.assertFalse(res2.detected)

        # Wait past cooldown (0.1s)
        time.sleep(0.15)
        res3 = self.engine.feed_audio(chunk)
        self.assertTrue(res3.detected)

    def test_paused_engine_ignores_feed(self) -> None:
        """Verify paused engine ignores incoming audio."""
        self.mock_model.return_score = 0.95
        self.engine.pause()
        self.assertTrue(self.engine.is_paused)

        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = self.engine.feed_audio(chunk)
        self.assertFalse(result.detected)

        self.engine.resume()
        self.assertFalse(self.engine.is_paused)


class TestWakeWordLifecycle(unittest.TestCase):
    """Test suite for start/stop lifecycle and background thread management."""

    def setUp(self) -> None:
        self.bus = EventBus()
        self.container = ServiceContainer()
        self.mock_model = MockOpenWakeWordModel(return_score=0.0)
        self.engine = WakeWordEngine(
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )

    def test_start_and_stop_lifecycle_events(self) -> None:
        """Verify start and stop publish events on EventBus and update state."""
        events: list[Event] = []
        self.bus.subscribe("wakeword.started", lambda ev: events.append(ev))
        self.bus.subscribe("wakeword.stopped", lambda ev: events.append(ev))

        self.engine.start()
        self.assertTrue(self.engine.is_running)

        # Idempotent start
        self.engine.start()
        self.assertTrue(self.engine.is_running)

        self.engine.stop()
        self.assertFalse(self.engine.is_running)

        # Idempotent stop
        self.engine.stop()
        self.assertFalse(self.engine.is_running)

        # Check events
        event_names = [e.name for e in events]
        self.assertIn("wakeword.started", event_names)
        self.assertIn("wakeword.stopped", event_names)

    def test_context_manager_lifecycle(self) -> None:
        """Verify context manager activates and deactivates listener."""
        with self.engine as eng:
            self.assertTrue(eng.is_running)
        self.assertFalse(self.engine.is_running)

    def test_queue_audio_method(self) -> None:
        """Verify queue_audio enqueues frames for the background thread."""
        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        self.engine.queue_audio(chunk)
        self.assertEqual(self.engine._audio_queue.qsize(), 1)


class TestVoiceConversationEngineIntegration(unittest.TestCase):
    """Test suite for automatic VoiceConversationEngine.listen_once() execution."""

    def setUp(self) -> None:
        self.bus = EventBus()
        self.container = ServiceContainer()
        self.mock_model = MockOpenWakeWordModel(return_score=0.9)
        self.mock_voice_engine = MagicMock()
        self.mock_voice_engine.listen_once.return_value = VoiceConversationResult(
            text="hello jarvis",
            command="hello jarvis",
            response_text="Hello sir, how may I assist you?",
        )

    def test_automatic_listen_once_trigger_on_wake_word(self) -> None:
        """Verify listen_once() is automatically executed upon wake word detection."""
        engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            model_instance=self.mock_model,
            voice_engine=self.mock_voice_engine,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_listen=True,
            auto_register_in_container=False,
            cooldown_seconds=0.1,
        )

        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = engine.feed_audio(chunk)

        self.assertTrue(result.detected)
        self.mock_voice_engine.listen_once.assert_called_once()

    def test_auto_listen_disabled(self) -> None:
        """Verify listen_once() is NOT called when auto_listen is False."""
        engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            model_instance=self.mock_model,
            voice_engine=self.mock_voice_engine,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_listen=False,
            auto_register_in_container=False,
            cooldown_seconds=0.1,
        )

        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = engine.feed_audio(chunk)

        self.assertTrue(result.detected)
        self.mock_voice_engine.listen_once.assert_not_called()

    def test_listen_once_failure_emits_wakeword_failed(self) -> None:
        """Verify errors in listen_once() emit wakeword.failed without crashing."""
        self.mock_voice_engine.listen_once.side_effect = RuntimeError("Microphone hardware error")

        failed_events: list[Event] = []
        self.bus.subscribe("wakeword.failed", lambda ev: failed_events.append(ev))

        engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            model_instance=self.mock_model,
            voice_engine=self.mock_voice_engine,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_listen=True,
            auto_register_in_container=False,
            cooldown_seconds=0.1,
        )

        chunk = np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = engine.feed_audio(chunk)

        self.assertTrue(result.detected)
        self.assertEqual(len(failed_events), 1)
        ev = failed_events[0]
        self.assertEqual(ev.name, "wakeword.failed")
        self.assertIn("Microphone hardware error", ev.payload["error"])
        self.assertEqual(ev.payload["stage"], "conversation_trigger")
        self.assertFalse(engine.is_paused)  # resumed after error

    def test_lazy_voice_engine_container_resolution(self) -> None:
        """Verify voice_engine is resolved from container if registered later."""
        engine = WakeWordEngine(
            wake_word="Jarvis",
            model_instance=self.mock_model,
            container_instance=self.container,
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )
        self.assertIsNone(engine.voice_engine)

        self.container.register_singleton("voice_engine", self.mock_voice_engine)
        self.assertIs(engine.voice_engine, self.mock_voice_engine)


class TestBackgroundListeningLoop(unittest.TestCase):
    """Test suite for continuous listening loop via audio callbacks and queues."""

    def setUp(self) -> None:
        self.bus = EventBus()
        self.container = ServiceContainer()
        self.mock_model = MockOpenWakeWordModel(return_score=0.9)
        self.mock_voice_engine = MagicMock()

    def test_audio_source_callback_triggers_detection(self) -> None:
        """Verify audio_source_callback provides audio chunks to background loop."""
        detected_events: list[Event] = []
        self.bus.subscribe("wakeword.detected", lambda ev: detected_events.append(ev))

        call_count = 0

        def sample_callback() -> Optional[np.ndarray]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return np.zeros(CHUNK_SIZE_SAMPLES, dtype=np.int16)
            return None

        engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            model_instance=self.mock_model,
            voice_engine=self.mock_voice_engine,
            container_instance=self.container,
            event_bus_instance=self.bus,
            audio_source_callback=sample_callback,
            auto_register_in_container=False,
            cooldown_seconds=0.01,
        )

        engine.start()
        # Give thread time to read chunks
        time.sleep(0.1)
        engine.stop()

        self.assertGreaterEqual(len(detected_events), 1)
        self.mock_voice_engine.listen_once.assert_called()


class TestRealOpenWakeWordInference(unittest.TestCase):
    """Integration test verifying real openwakeword ONNX model inference."""

    def test_real_openwakeword_model_loading_and_inference(self) -> None:
        """Verify real openWakeWord model loads and predicts on synthetic noise."""
        engine = WakeWordEngine(
            wake_word="Jarvis",
            threshold=0.5,
            auto_register_in_container=False,
            auto_listen=False,
        )
        self.assertIsNotNone(engine.model)

        # Feed 16-bit 16kHz synthetic noise
        noise = np.random.randint(-100, 100, size=CHUNK_SIZE_SAMPLES, dtype=np.int16)
        result = engine.feed_audio(noise)

        # Noise should not detect 'Jarvis'
        self.assertFalse(result.detected)
        self.assertGreaterEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
