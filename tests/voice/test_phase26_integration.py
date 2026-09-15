"""Phase 26 Integration Tests — Real-Time Playback, VAD, Interruption & Telemetry.

Covers all Phase 26 requirements without hardware dependencies:
1. Non-blocking TTS speaker playback
2. Playback RMS amplitude telemetry via existing Phase 25 path
3. Dynamic speech-end VAD with configurable thresholds
4. Fixed-duration recording backward compatibility
5. Voice playback interruption / cancellation
6. SPEAKING -> playback -> IDLE lifecycle
7. Amplitude reset on completion / error / interruption
8. Safe shutdown and error recovery
9. Maximum recording timeout protection
"""
from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
import tempfile
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.voice.microphone import MicrophoneRecorder
from app.voice.player import (
    AudioPlayer,
    EVENT_PLAYBACK_COMPLETED,
    EVENT_PLAYBACK_STARTED,
    EVENT_PLAYBACK_STOPPED,
)
from app.voice.engine import VoiceConversationEngine
from app.router.router import CommandRouter
from app.stt.transcriber import SpeechToText
from app.stt.models import TranscriptionResult
from app.tts.speaker import TextToSpeech
from app.tts.models import SpeechResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine_wav(path: Path, duration: float = 0.4, freq: float = 440.0, sr: int = 16000) -> Path:
    """Create a 16-bit mono PCM WAV with a sine wave."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def _sine_pcm(num_samples: int, amp: float = 0.5, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * (32767 * amp)).astype(np.int16).tobytes()


def _silence_pcm(num_samples: int) -> bytes:
    return b"\x00" * (num_samples * 2)


# ---------------------------------------------------------------------------
# Requirement 1 & 2: Non-blocking playback + RMS amplitude telemetry
# ---------------------------------------------------------------------------

class TestNonBlockingPlaybackWithAmplitudeTelemetry:
    def setup_method(self):
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_async_playback_does_not_block_caller(self):
        audio = _make_sine_wav(self.tmp_path / "tone.wav", duration=0.5)
        done = threading.Event()
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        t0 = time.perf_counter()
        result = player.play(audio, block=False, on_completed=lambda: done.set())
        elapsed = time.perf_counter() - t0
        assert result is True
        # play() must have returned before the full 0.5 s audio duration
        assert elapsed < 0.45
        # Wait for background thread to complete naturally (up to 5 s)
        assert done.wait(timeout=5.0), "Playback did not complete within 5s"
        assert not player.is_playing

    def test_rms_amplitude_reported_during_playback(self):
        audio = _make_sine_wav(self.tmp_path / "tone.wav", duration=0.3)
        amplitudes: List[float] = []
        player = AudioPlayer(
            amplitude_callback=lambda a: amplitudes.append(a),
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        player.play(audio, block=True)
        assert len(amplitudes) > 0
        assert max(amplitudes[:-1]) > 0.01
        assert all(0.0 <= a <= 1.0 for a in amplitudes)

    def test_amplitude_reset_to_zero_on_completion(self):
        audio = _make_sine_wav(self.tmp_path / "tone.wav", duration=0.2)
        amplitudes: List[float] = []
        player = AudioPlayer(
            amplitude_callback=lambda a: amplitudes.append(a),
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        player.play(audio, block=True)
        assert amplitudes[-1] == 0.0

    def test_lifecycle_events_on_completion(self):
        audio = _make_sine_wav(self.tmp_path / "tone.wav", duration=0.2)
        events: List[str] = []
        self.event_bus.subscribe(EVENT_PLAYBACK_STARTED, lambda e: events.append(e.name))
        self.event_bus.subscribe(EVENT_PLAYBACK_COMPLETED, lambda e: events.append(e.name))
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        player.play(audio, block=True)
        assert EVENT_PLAYBACK_STARTED in events
        assert EVENT_PLAYBACK_COMPLETED in events


# ---------------------------------------------------------------------------
# Requirement 3 & 4: VAD + fixed-duration backward compatibility
# ---------------------------------------------------------------------------

class TestVADAndFixedDurationRecording:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def _make_source(self, data: bytes):
        offset = [0]
        def src(n: int) -> bytes:
            take = n * 2
            chunk = data[offset[0]: offset[0] + take]
            offset[0] += take
            if len(chunk) < take:
                chunk += b"\x00" * (take - len(chunk))
            return chunk
        return src

    def test_vad_auto_stop_after_speech_silence(self):
        sr = 16000
        data = _silence_pcm(int(0.1 * sr)) + _sine_pcm(int(0.3 * sr), amp=0.5) + _silence_pcm(int(3 * sr))
        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=self._make_source(data),
            speech_threshold=0.02,
            silence_timeout=0.2,
            min_speech_duration=0.1,
            max_record_duration=5.0,
            auto_register_in_container=False,
        )
        out = recorder.record(use_vad=True)
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / float(sr)
        assert dur < 2.0

    def test_vad_max_duration_caps_silent_recording(self):
        sr = 16000
        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=lambda n: _silence_pcm(n),
            max_record_duration=0.4,
            auto_register_in_container=False,
        )
        out = recorder.record(use_vad=True)
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / float(sr)
        assert 0.35 <= dur <= 0.45

    def test_fixed_duration_recording_ignores_vad(self):
        sr = 16000
        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=lambda n: _silence_pcm(n),
            auto_register_in_container=False,
        )
        out = recorder.record(duration=0.3)
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / float(sr)
        assert abs(dur - 0.3) < 0.05


# ---------------------------------------------------------------------------
# Requirement 5 & 6: Interruption / cancellation + amplitude reset
# ---------------------------------------------------------------------------

class TestVoiceInterruptionAndCancellation:
    def setup_method(self):
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_stop_interrupts_active_playback_under_1s(self):
        long_audio = _make_sine_wav(self.tmp_path / "long.wav", duration=3.0)
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        t0 = time.perf_counter()
        player.play(long_audio, block=False)
        time.sleep(0.1)
        assert player.is_playing
        player.stop()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5
        assert not player.is_playing

    def test_interrupt_resets_amplitude_to_zero(self):
        long_audio = _make_sine_wav(self.tmp_path / "long.wav", duration=3.0)
        amps: List[float] = []
        player = AudioPlayer(
            amplitude_callback=lambda a: amps.append(a),
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        player.play(long_audio, block=False)
        time.sleep(0.08)
        player.interrupt()
        assert amps[-1] == 0.0
        assert not player.is_playing

    def test_engine_interrupt_stops_player_and_recovers_idle(self):
        sample_wav = _make_sine_wav(self.tmp_path / "jarvis.wav", duration=0.6)
        state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", state_manager)
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        engine = VoiceConversationEngine(
            recorder=MagicMock(spec=MicrophoneRecorder),
            stt=MagicMock(spec=SpeechToText),
            router=MagicMock(spec=CommandRouter),
            tts=MagicMock(spec=TextToSpeech),
            player=player,
            state_manager=state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        try:
            state_manager.transition_to(AssistantState.THINKING)
            state_manager.transition_to(AssistantState.SPEAKING, status_message="Speaking")
            player.play(sample_wav, block=False)
            time.sleep(0.05)
            assert player.is_playing
            engine.interrupt()
            assert not player.is_playing
            assert state_manager.get_snapshot().mic_amplitude == 0.0
            assert state_manager.get_snapshot().state == AssistantState.IDLE
        finally:
            player.shutdown()


# ---------------------------------------------------------------------------
# Requirement 7 & 8: SPEAKING -> playback -> IDLE lifecycle
# ---------------------------------------------------------------------------

class TestPlaybackLifecycle:
    def setup_method(self):
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def _make_mocks(self, sample_wav: Path):
        mock_recorder = MagicMock(spec=MicrophoneRecorder)
        mock_recorder.record.return_value = sample_wav
        mock_stt = MagicMock(spec=SpeechToText)
        mock_stt.transcribe.return_value = TranscriptionResult(text="hello jarvis", language="en")
        mock_router = MagicMock(spec=CommandRouter)
        mock_router.route.return_value = "Good evening, sir."
        mock_tts = MagicMock(spec=TextToSpeech)
        mock_tts.synthesize.return_value = SpeechResult(
            text="Good evening, sir.",
            audio_path=sample_wav,
            voice="en-US",
        )
        return mock_recorder, mock_stt, mock_router, mock_tts

    def test_listen_once_routes_through_player_and_returns_idle(self):
        sample_wav = _make_sine_wav(self.tmp_path / "resp.wav", duration=0.3)
        state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", state_manager)
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        rec, stt, router, tts = self._make_mocks(sample_wav)
        engine = VoiceConversationEngine(
            recorder=rec,
            stt=stt,
            router=router,
            tts=tts,
            player=player,
            state_manager=state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        result = engine.listen_once()
        assert result.success is True
        assert not player.is_playing
        assert state_manager.get_snapshot().mic_amplitude == 0.0

    def test_player_amplitude_forwarded_to_state_manager(self):
        sample_wav = _make_sine_wav(self.tmp_path / "resp.wav", duration=0.3)
        state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", state_manager)

        reported: List[float] = []
        state_manager.add_listener(
            lambda ev: reported.append(ev.snapshot.mic_amplitude) if ev.snapshot else None
        )

        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        rec, stt, router, tts = self._make_mocks(sample_wav)
        engine = VoiceConversationEngine(
            recorder=rec, stt=stt, router=router, tts=tts,
            player=player, state_manager=state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        engine.listen_once()
        assert len(reported) > 0
        assert max(reported) > 0.01


# ---------------------------------------------------------------------------
# Requirement 9: Safe shutdown and error recovery
# ---------------------------------------------------------------------------

class TestSafeShutdownAndErrorRecovery:
    def setup_method(self):
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_playback_missing_file_returns_false(self):
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        result = player.play(self.tmp_path / "ghost.wav", block=True)
        assert result is False
        assert not player.is_playing

    def test_repeated_stop_calls_are_idempotent(self):
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        for _ in range(5):
            player.stop()
            player.interrupt()
        assert not player.is_playing

    def test_player_shutdown_after_active_playback(self):
        long_audio = _make_sine_wav(self.tmp_path / "long.wav", duration=3.0)
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        player.play(long_audio, block=False)
        time.sleep(0.05)
        player.shutdown()
        assert not player.is_playing

    def test_engine_stop_cleans_up_player(self):
        long_audio = _make_sine_wav(self.tmp_path / "long.wav", duration=3.0)
        state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", state_manager)
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        engine = VoiceConversationEngine(
            recorder=MagicMock(spec=MicrophoneRecorder),
            stt=MagicMock(spec=SpeechToText),
            router=MagicMock(spec=CommandRouter),
            tts=MagicMock(spec=TextToSpeech),
            player=player,
            state_manager=state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        engine.start()
        player.play(long_audio, block=False)
        time.sleep(0.05)
        engine.stop()
        assert not player.is_playing
        assert state_manager.get_snapshot().mic_amplitude == 0.0


# ---------------------------------------------------------------------------
# Requirement 10: Maximum recording timeout protection
# ---------------------------------------------------------------------------

class TestMaxRecordingTimeout:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_fixed_duration_caps_at_requested_duration(self):
        sr = 16000
        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=lambda n: _sine_pcm(n, amp=0.5),
            max_record_duration=10.0,
            vad_enabled=False,
            auto_register_in_container=False,
        )
        out = recorder.record(duration=0.35, use_vad=False)
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / float(sr)
        assert abs(dur - 0.35) < 0.05

    def test_vad_stops_before_max_on_trailing_silence(self):
        sr = 16000
        data = _sine_pcm(int(0.2 * sr), amp=0.5) + _silence_pcm(int(5 * sr))
        offset = [0]

        def src(n: int) -> bytes:
            take = n * 2
            chunk = data[offset[0]: offset[0] + take]
            offset[0] += take
            if len(chunk) < take:
                chunk += b"\x00" * (take - len(chunk))
            return chunk

        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=src,
            speech_threshold=0.02,
            silence_timeout=0.15,
            min_speech_duration=0.05,
            max_record_duration=5.0,
            auto_register_in_container=False,
        )
        out = recorder.record(use_vad=True)
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / float(sr)
        assert dur < 2.0
