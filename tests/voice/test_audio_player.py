"""Tests for AudioPlayer subsystem in Phase 26.

Validates non-blocking playback, interruption, amplitude telemetry, single-stream
concurrency, error resilience, and event bus lifecycle without hardware dependencies.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import wave

import numpy as np
import pytest

from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.voice.player import (
    AudioPlayer,
    EVENT_PLAYBACK_COMPLETED,
    EVENT_PLAYBACK_FAILED,
    EVENT_PLAYBACK_STARTED,
    EVENT_PLAYBACK_STOPPED,
)


def _create_sine_wav(file_path: Path, duration_sec: float = 0.5, freq: float = 440.0, sample_rate: int = 16000) -> Path:
    """Create a standard 16-bit mono PCM WAV file containing a sine wave."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Amplitude 0.5 of max int16
    samples = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return file_path


def _create_silent_wav(file_path: Path, duration_sec: float = 0.3, sample_rate: int = 16000) -> Path:
    """Create a silent 16-bit mono PCM WAV file."""
    num_samples = int(duration_sec * sample_rate)
    samples = np.zeros(num_samples, dtype=np.int16)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return file_path


class TestAudioPlayer:
    """Test suite for AudioPlayer."""

    def setup_method(self) -> None:
        """Set up test environment with isolated container, event bus, and temp dir."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def teardown_method(self) -> None:
        """Clean up temporary resources."""
        self.temp_dir.cleanup()

    def test_audio_player_initialization_and_properties(self) -> None:
        """Verify initialization defaults, container registration, and property getters."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )
        assert not player.is_playing
        assert player.amplitude_callback is None
        assert self.container.exists("audio_player")
        assert self.container.resolve("audio_player") is player

    def test_playback_missing_file_fails_gracefully(self) -> None:
        """Verify playing a non-existent file returns False and triggers error callback."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        errors: list[Exception] = []
        result = player.play(
            self.tmp_path / "non_existent_audio.wav",
            block=True,
            on_error=lambda exc: errors.append(exc),
        )
        assert result is False
        assert len(errors) == 1
        assert isinstance(errors[0], FileNotFoundError)
        assert not player.is_playing

    def test_playback_empty_corrupt_file_fails_gracefully(self) -> None:
        """Verify playing an empty file returns False without unhandled exceptions."""
        empty_file = self.tmp_path / "empty.wav"
        empty_file.touch()

        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        result = player.play(empty_file, block=True)
        assert result is False
        assert not player.is_playing

    def test_playback_synchronous_success_and_lifecycle_events(self) -> None:
        """Verify successful synchronous playback triggers lifecycle events and callbacks."""
        audio_file = _create_sine_wav(self.tmp_path / "tone.wav", duration_sec=0.2)
        events_published: list[str] = []

        self.event_bus.subscribe(EVENT_PLAYBACK_STARTED, lambda ev: events_published.append(ev.name))
        self.event_bus.subscribe(EVENT_PLAYBACK_COMPLETED, lambda ev: events_published.append(ev.name))

        completed_called = threading.Event()
        amplitudes: list[float] = []

        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            amplitude_callback=lambda amp: amplitudes.append(amp),
            auto_register_in_container=False,
        )

        success = player.play(
            audio_file,
            block=True,
            on_completed=lambda: completed_called.set(),
        )

        assert success is True
        assert completed_called.is_set()
        assert EVENT_PLAYBACK_STARTED in events_published
        assert EVENT_PLAYBACK_COMPLETED in events_published
        assert len(amplitudes) > 0
        # Positive sine amplitude reported
        assert max(amplitudes) > 0.05
        # Amplitude normalized into [0.0, 1.0]
        assert all(0.0 <= a <= 1.0 for a in amplitudes)
        # Final amplitude reset to 0.0
        assert amplitudes[-1] == 0.0
        assert not player.is_playing

    def test_playback_asynchronous_non_blocking(self) -> None:
        """Verify non-blocking playback runs in background thread."""
        audio_file = _create_sine_wav(self.tmp_path / "tone_async.wav", duration_sec=0.3)
        completed_flag = threading.Event()

        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        ret = player.play(audio_file, block=False, on_completed=lambda: completed_flag.set())
        assert ret is True
        # Completes asynchronously within 1 second
        assert completed_flag.wait(timeout=2.0) is True
        assert not player.is_playing

    def test_playback_interruption_stop_resets_amplitude(self) -> None:
        """Verify stop() / interrupt() halts active playback immediately and emits stopped event."""
        # Long audio 3.0s
        audio_file = _create_sine_wav(self.tmp_path / "long_tone.wav", duration_sec=3.0)
        stopped_events: list[str] = []
        self.event_bus.subscribe(EVENT_PLAYBACK_STOPPED, lambda ev: stopped_events.append(ev.name))

        amps: list[float] = []
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            amplitude_callback=lambda a: amps.append(a),
            auto_register_in_container=False,
        )

        t0 = time.perf_counter()
        player.play(audio_file, block=False)
        time.sleep(0.1)
        assert player.is_playing

        player.interrupt()
        elapsed = time.perf_counter() - t0

        # Stopped well before 3 seconds
        assert elapsed < 1.5
        assert not player.is_playing
        assert len(stopped_events) == 1
        assert amps[-1] == 0.0

    def test_repeated_stop_and_interrupt_are_safe_and_idempotent(self) -> None:
        """Verify repeated calls to stop() or interrupt() do not raise errors."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        # Calling stop before any playback
        player.stop()
        player.interrupt()
        player.shutdown()
        assert not player.is_playing

    def test_overlapping_playback_prevention(self) -> None:
        """Verify that starting playback while already playing stops previous playback."""
        audio1 = _create_sine_wav(self.tmp_path / "tone1.wav", duration_sec=2.0, freq=300.0)
        audio2 = _create_sine_wav(self.tmp_path / "tone2.wav", duration_sec=0.2, freq=600.0)

        completed2 = threading.Event()
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Start long first audio
        player.play(audio1, block=False)
        time.sleep(0.05)
        assert player.is_playing

        # Starting second audio must interrupt first and play second
        player.play(audio2, block=False, on_completed=lambda: completed2.set())
        assert completed2.wait(timeout=2.0) is True
        assert not player.is_playing
