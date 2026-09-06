"""Comprehensive unit tests for the J.A.R.V.I.S Wake Word Subsystem."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, patch

from app.core.config import Settings, load_settings
from app.core.constants import DEFAULT_WAKE_PHRASES
from app.core.container import ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.wakeword import (
    DEFAULT_WAKE_PHRASES as EXPORTED_DEFAULT_WAKE_PHRASES,
    EVENT_SOURCE_WAKEWORD,
    EVENT_WAKEWORD_DETECTED,
    EVENT_WAKEWORD_IGNORED,
    InvalidInputError,
    WakeWordDetector,
    WakeWordError,
    WakeWordResult,
    normalize_text,
    wake_word_detector,
)


class TestTextNormalization(unittest.TestCase):
    """Test suite for normalize_text function."""

    def test_case_folding(self) -> None:
        """Verify uppercase and mixed case text is normalized to lowercase."""
        self.assertEqual(normalize_text("HEY JARVIS"), "hey jarvis")
        self.assertEqual(normalize_text("HeY jArViS"), "hey jarvis")
        self.assertEqual(normalize_text("Jarvis Activate"), "jarvis activate")

    def test_punctuation_removal(self) -> None:
        """Verify commas, periods, exclamation points, etc. are stripped and ignored."""
        self.assertEqual(normalize_text("Hey, Jarvis!"), "hey jarvis")
        self.assertEqual(normalize_text("Activate Jarvis..."), "activate jarvis")
        self.assertEqual(normalize_text("Jarvish, Activate???"), "jarvish activate")
        self.assertEqual(normalize_text("“Hello, Jarvis!”"), "hello jarvis")
        self.assertEqual(normalize_text("jarvis-activate"), "jarvis activate")
        self.assertEqual(normalize_text("hey; jarvis: activate!"), "hey jarvis activate")

    def test_extra_whitespace_handling(self) -> None:
        """Verify multiple spaces, tabs, newlines, and surrounding padding are collapsed."""
        self.assertEqual(normalize_text("   Hey    Jarvis   "), "hey jarvis")
        self.assertEqual(normalize_text("\t\nHey\t\n\tJarvis\n"), "hey jarvis")

    def test_empty_and_none_input(self) -> None:
        """Verify empty strings and None are handled gracefully without errors."""
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("   "), "")
        self.assertEqual(normalize_text(None), "")


class TestWakeWordDetection(unittest.TestCase):
    """Test suite for default wake word detection logic."""

    def setUp(self) -> None:
        """Create fresh isolated detector and event bus for testing."""
        self.test_bus = EventBus()
        self.test_container = ServiceContainer()
        self.detector = WakeWordDetector(
            event_bus_instance=self.test_bus,
            container_instance=self.test_container,
            auto_register_in_container=False,
        )

    def test_default_wake_phrases(self) -> None:
        """Verify all 5 required default wake phrases trigger detection."""
        required_phrases = [
            ("Hey Jarvis", "Hey Jarvis"),
            ("Activate Jarvis", "Activate Jarvis"),
            ("Jarvis Activate", "Jarvis Activate"),
            ("Jarvish Activate", "Jarvish Activate"),
            ("Hello Jarvis", "Hello Jarvis"),
        ]
        for phrase_input, expected_match in required_phrases:
            with self.subTest(phrase=phrase_input):
                result = self.detector.detect(phrase_input)
                self.assertTrue(result.detected)
                self.assertEqual(result.matched_phrase, expected_match)
                self.assertTrue(bool(result))

    def test_case_and_punctuation_insensitivity(self) -> None:
        """Verify matching succeeds regardless of casing, punctuation, and extra spaces."""
        test_cases = [
            ("  hey,   jarvis...  ", "Hey Jarvis"),
            ("HEY, JARVIS!", "Hey Jarvis"),
            ("...activate jarvis???", "Activate Jarvis"),
            ("JARVIS, ACTIVATE!", "Jarvis Activate"),
            ("jarvish... activate.", "Jarvish Activate"),
            ("  HELLO,   JARVIS!  ", "Hello Jarvis"),
        ]
        for raw_input, expected_phrase in test_cases:
            with self.subTest(input_text=raw_input):
                result = self.detector.detect(raw_input)
                self.assertTrue(result.detected)
                self.assertEqual(result.matched_phrase, expected_phrase)

    def test_wake_phrase_with_command_context(self) -> None:
        """Verify detection succeeds when wake phrase precedes a user command."""
        test_cases = [
            ("Hey Jarvis, what is the weather today?", "Hey Jarvis"),
            ("Activate Jarvis! Set an alarm for 7 AM.", "Activate Jarvis"),
            ("Hello Jarvis, could you play music?", "Hello Jarvis"),
            ("Jarvis activate and open chrome", "Jarvis Activate"),
        ]
        for command, expected_phrase in test_cases:
            with self.subTest(command=command):
                result = self.detector.detect(command)
                self.assertTrue(result.detected)
                self.assertEqual(result.matched_phrase, expected_phrase)

    def test_non_wake_words_ignored(self) -> None:
        """Verify random text without wake words is properly ignored."""
        negatives = [
            "What is the capital of France?",
            "Open the pod bay doors.",
            "Turn on the living room lights.",
            "Hello world",
            "Hey assistant",
            "",
            "   ",
            None,
        ]
        for text in negatives:
            with self.subTest(text=text):
                result = self.detector.detect(text)
                self.assertFalse(result.detected)
                self.assertIsNone(result.matched_phrase)
                self.assertFalse(bool(result))

    def test_partial_word_boundaries_rejected(self) -> None:
        """Verify words containing wake phrases as substrings are not falsely triggered."""
        partials = [
            "hey jarvisite",
            "activate jarvising",
            "unjarvis activate",
            "hello jarvises",
        ]
        for text in partials:
            with self.subTest(text=text):
                result = self.detector.detect(text)
                self.assertFalse(result.detected)
                self.assertIsNone(result.matched_phrase)

    def test_convenience_helper(self) -> None:
        """Verify is_wake_phrase returns clean boolean value."""
        self.assertTrue(self.detector.is_wake_phrase("Hey Jarvis"))
        self.assertFalse(self.detector.is_wake_phrase("Good morning"))


class TestWakeWordResultModel(unittest.TestCase):
    """Test suite for WakeWordResult model."""

    def test_model_fields_and_immutability(self) -> None:
        """Verify WakeWordResult fields, types, and frozen immutability."""
        t0 = time.time()
        result = WakeWordResult(
            detected=True,
            matched_phrase="Hey Jarvis",
            normalized_text="hey jarvis",
            timestamp=t0,
        )
        self.assertTrue(result.detected)
        self.assertEqual(result.matched_phrase, "Hey Jarvis")
        self.assertEqual(result.normalized_text, "hey jarvis")
        self.assertEqual(result.timestamp, t0)
        self.assertTrue(bool(result))

        # Test immutability
        with self.assertRaises(AttributeError):
            result.detected = False  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict returns appropriate dictionary structure."""
        result = WakeWordResult(
            detected=False,
            matched_phrase=None,
            normalized_text="random query",
        )
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertFalse(data["detected"])
        self.assertIsNone(data["matched_phrase"])
        self.assertEqual(data["normalized_text"], "random query")
        self.assertIn("timestamp", data)


class TestEventBusIntegration(unittest.TestCase):
    """Test suite for Event Bus publishing upon detection and rejection."""

    def setUp(self) -> None:
        """Set up isolated event bus and detector."""
        self.bus = EventBus()
        self.detector = WakeWordDetector(
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )

    def test_publish_detected_event(self) -> None:
        """Verify wakeword.detected event is published when wake word matches."""
        events: list[Event] = []
        self.bus.subscribe(EVENT_WAKEWORD_DETECTED, lambda evt: events.append(evt))

        result = self.detector.detect("Hey, Jarvis!")
        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertEqual(evt.name, EVENT_WAKEWORD_DETECTED)
        self.assertEqual(evt.source, EVENT_SOURCE_WAKEWORD)
        self.assertTrue(evt.payload["detected"])
        self.assertEqual(evt.payload["matched_phrase"], "Hey Jarvis")
        self.assertEqual(evt.payload["normalized_text"], "hey jarvis")
        self.assertEqual(evt.payload["result"], result)

    def test_publish_ignored_event(self) -> None:
        """Verify wakeword.ignored event is published when input has no wake word."""
        events: list[Event] = []
        self.bus.subscribe(EVENT_WAKEWORD_IGNORED, lambda evt: events.append(evt))

        result = self.detector.detect("Turn off the bedroom lamp.")
        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertEqual(evt.name, EVENT_WAKEWORD_IGNORED)
        self.assertEqual(evt.source, EVENT_SOURCE_WAKEWORD)
        self.assertFalse(evt.payload["detected"])
        self.assertIsNone(evt.payload["matched_phrase"])
        self.assertEqual(evt.payload["normalized_text"], "turn off the bedroom lamp")
        self.assertEqual(evt.payload["result"], result)

    def test_suppress_event_publishing(self) -> None:
        """Verify publish_event=False suppresses Event Bus notifications."""
        events: list[Event] = []
        self.bus.subscribe("*", lambda evt: events.append(evt))

        result = self.detector.detect("Hey Jarvis", publish_event=False)
        self.assertTrue(result.detected)
        self.assertEqual(len(events), 0)

        result_neg = self.detector.detect("hello world", publish_event=False)
        self.assertFalse(result_neg.detected)
        self.assertEqual(len(events), 0)


class TestAsyncDetection(unittest.IsolatedAsyncioTestCase):
    """Test suite for asynchronous wake word detection."""

    async def asyncSetUp(self) -> None:
        """Set up async test environment."""
        self.bus = EventBus()
        self.detector = WakeWordDetector(
            event_bus_instance=self.bus,
            auto_register_in_container=False,
        )

    async def test_detect_async_detected(self) -> None:
        """Verify detect_async correctly identifies wake word and publishes async event."""
        events: list[Event] = []

        async def async_handler(evt: Event) -> None:
            await asyncio.sleep(0.01)
            events.append(evt)

        self.bus.subscribe(EVENT_WAKEWORD_DETECTED, async_handler)

        result = await self.detector.detect_async("Activate Jarvis, please.")
        self.assertTrue(result.detected)
        self.assertEqual(result.matched_phrase, "Activate Jarvis")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, EVENT_WAKEWORD_DETECTED)

    async def test_detect_async_ignored(self) -> None:
        """Verify detect_async correctly ignores non-matching phrases asynchronously."""
        events: list[Event] = []

        async def async_handler(evt: Event) -> None:
            await asyncio.sleep(0.01)
            events.append(evt)

        self.bus.subscribe(EVENT_WAKEWORD_IGNORED, async_handler)

        result = await self.detector.detect_async("play some music")
        self.assertFalse(result.detected)
        self.assertIsNone(result.matched_phrase)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, EVENT_WAKEWORD_IGNORED)


class TestConfigurableWakePhrases(unittest.TestCase):
    """Test suite for configurable wake phrases via Settings, env, and runtime methods."""

    def test_custom_wake_phrases_via_init(self) -> None:
        """Verify WakeWordDetector accepts custom wake phrases in constructor."""
        custom_detector = WakeWordDetector(
            wake_phrases=["Computer", "Jarvis Online"],
            auto_register_in_container=False,
        )
        self.assertEqual(custom_detector.wake_phrases, ("Computer", "Jarvis Online"))
        self.assertTrue(custom_detector.detect("Computer, report status!").detected)
        self.assertTrue(custom_detector.detect("Jarvis Online now").detected)
        # Default phrases should no longer trigger
        self.assertFalse(custom_detector.detect("Hey Jarvis").detected)

    def test_add_remove_reset_wake_phrase(self) -> None:
        """Verify dynamic addition, removal, and reset of wake phrases."""
        detector = WakeWordDetector(auto_register_in_container=False)
        self.assertFalse(detector.detect("System wake").detected)

        # Add
        detector.add_wake_phrase("System Wake")
        self.assertIn("System Wake", detector.wake_phrases)
        self.assertTrue(detector.detect("system wake now!").detected)

        # Remove
        removed = detector.remove_wake_phrase("System Wake")
        self.assertTrue(removed)
        self.assertNotIn("System Wake", detector.wake_phrases)
        self.assertFalse(detector.detect("system wake now!").detected)

        # Non-existent remove returns False
        self.assertFalse(detector.remove_wake_phrase("NonExistent"))

        # Reset
        detector.set_wake_phrases(["Single Phrase"])
        self.assertEqual(len(detector.wake_phrases), 1)
        detector.reset_wake_phrases()
        self.assertEqual(detector.wake_phrases, DEFAULT_WAKE_PHRASES)

    def test_invalid_wake_phrase_inputs(self) -> None:
        """Verify invalid input raises InvalidInputError."""
        detector = WakeWordDetector(auto_register_in_container=False)
        with self.assertRaises(InvalidInputError):
            detector.add_wake_phrase("")
        with self.assertRaises(InvalidInputError):
            detector.add_wake_phrase("   ")
        with self.assertRaises(InvalidInputError):
            detector.set_wake_phrases(None)  # type: ignore[arg-type]

    def test_load_from_config(self) -> None:
        """Verify detector loads wake phrases from Settings loaded from environment."""
        with patch.dict(os.environ, {"WAKE_PHRASES": "Wake Up Jarvis, Jarvis Wake"}):
            settings_instance = load_settings(env_file=None)
            detector = WakeWordDetector(
                config=settings_instance, auto_register_in_container=False
            )
            self.assertEqual(
                detector.wake_phrases, ("Wake Up Jarvis", "Jarvis Wake")
            )
            self.assertTrue(detector.detect("wake up jarvis").detected)


class TestServiceContainerIntegration(unittest.TestCase):
    """Test suite for Service Container registration and resolution."""

    def test_container_auto_registration(self) -> None:
        """Verify WakeWordDetector registers into the provided container."""
        test_container = ServiceContainer()
        detector = WakeWordDetector(
            container_instance=test_container,
            auto_register_in_container=True,
        )
        self.assertTrue(test_container.exists("wakeword_detector"))
        self.assertTrue(test_container.exists("wake_word_detector"))
        self.assertTrue(test_container.exists("wakeword"))
        self.assertIs(test_container.resolve("wakeword_detector"), detector)
        self.assertIs(test_container.resolve("wake_word_detector"), detector)
        self.assertIs(test_container.resolve("wakeword"), detector)

    def test_exported_singleton(self) -> None:
        """Verify wake_word_detector module singleton exists and is initialized."""
        self.assertIsInstance(wake_word_detector, WakeWordDetector)
        self.assertEqual(wake_word_detector.wake_phrases, DEFAULT_WAKE_PHRASES)


class TestThreadSafety(unittest.TestCase):
    """Test suite for thread safety during concurrent detection and phrase updates."""

    def test_concurrent_detection(self) -> None:
        """Verify concurrent calls to detect from multiple threads return consistent results."""
        detector = WakeWordDetector(auto_register_in_container=False)
        test_inputs = [
            ("Hey Jarvis, check mail", True, "Hey Jarvis"),
            ("Random command text", False, None),
            ("Hello Jarvis!", True, "Hello Jarvis"),
            ("Activate Jarvis now", True, "Activate Jarvis"),
            ("Jarvis Activate", True, "Jarvis Activate"),
            ("Jarvish, activate please", True, "Jarvish Activate"),
            ("Weather report today", False, None),
        ] * 20

        results: list[tuple[bool, bool]] = []

        def worker(item: tuple[str, bool, Any]) -> tuple[bool, bool]:
            text, expected_detected, _ = item
            res = detector.detect(text, publish_event=False)
            return res.detected, expected_detected

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, item) for item in test_inputs]
            for future in as_completed(futures):
                actual, expected = future.result()
                results.append((actual, expected))

        for actual, expected in results:
            self.assertEqual(actual, expected)

    def test_concurrent_phrase_modification_and_detection(self) -> None:
        """Verify thread safety when phrases are added/removed while detection runs."""
        detector = WakeWordDetector(auto_register_in_container=False)
        stop_event = threading.Event()
        errors: list[Exception] = []

        def detection_worker() -> None:
            while not stop_event.is_set():
                try:
                    detector.detect("Hey Jarvis, run task", publish_event=False)
                    detector.detect("Unknown command", publish_event=False)
                except Exception as exc:
                    errors.append(exc)

        def modification_worker() -> None:
            for i in range(50):
                try:
                    detector.add_wake_phrase(f"Phrase {i}")
                    time.sleep(0.001)
                    detector.remove_wake_phrase(f"Phrase {i}")
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=detection_worker) for _ in range(3)
        ] + [threading.Thread(target=modification_worker) for _ in range(2)]

        for t in threads:
            t.start()

        # Let it run briefly
        time.sleep(0.2)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
