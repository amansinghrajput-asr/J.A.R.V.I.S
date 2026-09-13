"""Tests for Dynamic RMS Voice Activity Detection (VAD) and Voice Interruption in Phase 26.

Validates:
- Speech onset detection
- Trailing silence auto-stop timeout
- Tolerance of short conversational pauses
- Timeout limits on continuous silence/noise
- Backward compatibility for explicit fixed-duration recordings
- Voice response playback lifecycle: SYNTHESIZING -> SPEAKING -> IDLE
- AudioPlayer amplitude telemetry forwarding to AssistantStateManager
- Safe voice interruption and state recovery
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch
import wave

import numpy as np
import pytest

from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.presentation import PresentationAdapter
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.router.router import CommandRouter
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech
from app.ui.bridge import UIBridge
from app.voice.engine import VoiceConversationEngine
from app.voice.microphone import (
    DEFAULT_MAX_RECORD_DURATION,
    DEFAULT_MIN_SPEECH_DURATION,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_SILENCE_TIMEOUT,
    DEFAULT_SPEECH_THRESHOLD,
    MicrophoneRecorder,
)
from app.voice.player import AudioPlayer


def _create_pcm_sine(num_samples: int, sample_rate: int = 16000, freq: float = 440.0, amp: float = 0.5) -> bytes:
    """Generate raw 16-bit mono PCM bytes for a sine wave."""
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * (32767 * amp)).astype(np.int16)
    return samples.tobytes()


def _create_pcm_silence(num_samples: int) -> bytes:
    """Generate raw 16-bit mono PCM bytes for silence (all zeros)."""
    return (np.zeros(num_samples, dtype=np.int16)).tobytes()


class TestVadDetection:
    """Test suite for RMS-based dynamic speech-end detection."""

    def setup_method(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def teardown_method(self) -> None:
        self.temp_dir.cleanup()

    def test_vad_property_initialization_and_setters(self) -> None:
        """Verify VAD thresholds, timeouts, and properties."""
        recorder = MicrophoneRecorder(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            vad_enabled=True,
            speech_threshold=0.03,
            silence_threshold=0.01,
            silence_timeout=0.6,
            min_speech_duration=0.15,
            max_record_duration=8.0,
            auto_register_in_container=False,
        )

        assert recorder.vad_enabled is True
        assert recorder.speech_threshold == 0.03
        assert recorder.silence_threshold == 0.01
        assert recorder.silence_timeout == 0.6
        assert recorder.min_speech_duration == 0.15
        assert recorder.max_record_duration == 8.0

        # Test setters
        recorder.speech_threshold = 0.05
        assert recorder.speech_threshold == 0.05
        recorder.silence_timeout = 1.0
        assert recorder.silence_timeout == 1.0

    def test_calculate_amplitude_silence_vs_signal(self) -> None:
        """Verify amplitude calculation correctly normalizes between 0.0 and 1.0."""
        recorder = MicrophoneRecorder(auto_register_in_container=False)

        silence = _create_pcm_silence(1024)
        amp_silence = recorder.calculate_amplitude(silence)
        assert amp_silence == 0.0

        sine_half = _create_pcm_sine(1024, amp=0.5)
        amp_sine = recorder.calculate_amplitude(sine_half)
        # Half scale RMS is ~ 0.5 / sqrt(2) ~ 0.35
        assert 0.30 <= amp_sine <= 0.40

        # Empty data
        assert recorder.calculate_amplitude(b"") == 0.0

    def test_vad_auto_stop_after_speech_and_silence(self) -> None:
        """Verify VAD terminates recording early after speech ends and silence threshold passes."""
        # Simulated audio source: 0.1s silence -> 0.3s speech -> trailing silence
        sr = 16000
        speech_pcm = _create_pcm_sine(int(0.3 * sr), sample_rate=sr, amp=0.6)
        silence_pcm = _create_pcm_silence(int(0.1 * sr))

        stream_data = silence_pcm + speech_pcm + (_create_pcm_silence(int(3.0 * sr)))
        offset = [0]

        def _mock_source(take_frames: int) -> bytes:
            take_bytes = take_frames * 2  # 16-bit mono
            start = offset[0]
            end = min(len(stream_data), start + take_bytes)
            offset[0] = end
            chunk = stream_data[start:end]
            if len(chunk) < take_bytes:
                chunk += b"\x00" * (take_bytes - len(chunk))
            return chunk

        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=_mock_source,
            speech_threshold=0.02,
            silence_timeout=0.2,  # Quick 200ms silence timeout for test
            min_speech_duration=0.1,
            max_record_duration=5.0,
            auto_register_in_container=False,
        )

        out_path = recorder.record(use_vad=True)
        assert out_path.exists()

        # Check recorded length: 0.1s silence + 0.3s speech + ~0.2s silence ~ 0.6s total (well under 5.0s max!)
        with wave.open(str(out_path), "rb") as wf:
            frames = wf.getnframes()
            dur = frames / float(sr)

        assert 0.4 <= dur <= 0.8

    def test_vad_runs_to_max_duration_when_all_silent(self) -> None:
        """Verify VAD gracefully caps at max duration when no speech is detected."""
        sr = 16000

        def _mock_silent(take_frames: int) -> bytes:
            return _create_pcm_silence(take_frames)

        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=_mock_silent,
            max_record_duration=0.4,
            silence_timeout=0.2,
            auto_register_in_container=False,
        )

        out_path = recorder.record(use_vad=True)
        assert out_path.exists()

        with wave.open(str(out_path), "rb") as wf:
            frames = wf.getnframes()
            dur = frames / float(sr)

        assert 0.35 <= dur <= 0.45

    def test_vad_tolerates_short_conversational_pause(self) -> None:
        """Verify short pause (< silence_timeout) does not stop recording."""
        sr = 16000
        # Word 1 (0.2s) -> Pause (0.08s < 0.25s silence_timeout) -> Word 2 (0.2s) -> Long silence (0.4s)
        word1 = _create_pcm_sine(int(0.2 * sr), sample_rate=sr, amp=0.6)
        pause = _create_pcm_silence(int(0.08 * sr))
        word2 = _create_pcm_sine(int(0.2 * sr), sample_rate=sr, amp=0.6)
        final_silence = _create_pcm_silence(int(2.0 * sr))

        stream_data = word1 + pause + word2 + final_silence
        offset = [0]

        def _mock_speech(take_frames: int) -> bytes:
            take_bytes = take_frames * 2
            start = offset[0]
            end = min(len(stream_data), start + take_bytes)
            offset[0] = end
            chunk = stream_data[start:end]
            if len(chunk) < take_bytes:
                chunk += b"\x00" * (take_bytes - len(chunk))
            return chunk

        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=_mock_speech,
            speech_threshold=0.02,
            silence_timeout=0.25,
            min_speech_duration=0.1,
            max_record_duration=4.0,
            auto_register_in_container=False,
        )

        out_path = recorder.record(use_vad=True)
        assert out_path.exists()

        with wave.open(str(out_path), "rb") as wf:
            dur = wf.getnframes() / float(sr)

        # Should contain Word 1 + Pause + Word 2 + Silence timeout ~ 0.73s
        assert dur >= 0.6
        assert dur < 1.5

    def test_explicit_fixed_duration_ignores_vad(self) -> None:
        """Verify backward compatibility: when duration is passed without use_vad=True, fixed duration is preserved."""
        sr = 16000
        recorder = MicrophoneRecorder(
            sample_rate=sr,
            output_dir=self.tmp_path,
            audio_source_callback=lambda n: _create_pcm_silence(n),
            auto_register_in_container=False,
        )

        out_path = recorder.record(duration=0.3)
        assert out_path.exists()

        with wave.open(str(out_path), "rb") as wf:
            dur = wf.getnframes() / float(sr)

        assert abs(dur - 0.3) < 0.05


class TestVoiceResponsePlaybackAndInterruption:
    """Test suite for voice engine response playback lifecycle and interruption."""

    def setup_method(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Synthesize helper wav file
        self.sample_wav = self.tmp_path / "jarvis_response.wav"
        samples = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.4, int(0.4 * 16000))) * 16384).astype(np.int16)
        with wave.open(str(self.sample_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())

        # State manager
        self.state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", self.state_manager)

        # Mocks
        self.mock_recorder = MagicMock(spec=MicrophoneRecorder)
        self.mock_recorder.record.return_value = self.sample_wav
        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="hello jarvis", language="en")
        self.mock_router = MagicMock(spec=CommandRouter)
        self.mock_router.route.return_value = "Good day, sir."
        self.mock_tts = MagicMock(spec=TextToSpeech)
        self.mock_tts.synthesize.return_value = SpeechResult(
            text="Good day, sir.",
            audio_path=self.sample_wav,
            voice="en-US",
        )

    def teardown_method(self) -> None:
        self.state_manager.close()
        self.temp_dir.cleanup()

    def test_voice_engine_playback_lifecycle_and_amplitude_telemetry(self) -> None:
        """Verify listen_once() executes playback through AudioPlayer and forwards RMS telemetry."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            player=player,
            state_manager=self.state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

        reported_amplitudes: list[float] = []
        self.state_manager.add_listener(
            lambda ev: reported_amplitudes.append(ev.snapshot.mic_amplitude) if ev.snapshot else None
        )

        result = engine.listen_once()

        assert result.success is True
        assert result.response_text == "Good day, sir."
        assert not player.is_playing
        # Player forwarded real amplitude during playback
        assert len(reported_amplitudes) > 0
        assert max(reported_amplitudes) > 0.05
        # Final amplitude was reset to 0.0
        assert self.state_manager.get_snapshot().mic_amplitude == 0.0
        # Engine transitioned cleanly back to IDLE
        assert self.state_manager.get_snapshot().state == AssistantState.IDLE

    def test_interruption_during_playback_recovers_state_to_idle(self) -> None:
        """Verify calling engine.interrupt() halts active playback and recovers to IDLE."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.mock_router,
            tts=self.mock_tts,
            player=player,
            state_manager=self.state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Transition through valid states: IDLE -> THINKING -> SPEAKING
        self.state_manager.transition_to(AssistantState.THINKING)
        self.state_manager.transition_to(AssistantState.SPEAKING, status_message="Speaking response...")
        player.play(self.sample_wav, block=False)
        time.sleep(0.05)

        assert self.state_manager.get_snapshot().state == AssistantState.SPEAKING

        # Trigger interrupt
        engine.interrupt()

        assert not player.is_playing
        assert self.state_manager.get_snapshot().mic_amplitude == 0.0
        assert self.state_manager.get_snapshot().state == AssistantState.IDLE

    def test_presentation_adapter_and_ui_bridge_interruption(self) -> None:
        """Verify PresentationAdapter and UIBridge interruption pathways."""
        player = AudioPlayer(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

        adapter = PresentationAdapter(
            state_manager=self.state_manager,
            container_instance=self.container,
        )
        self.container.register_singleton("presentation_adapter", adapter)

        bridge = UIBridge(
            presentation_adapter=adapter,
            parent=None,
        )

        # Set assistant state: IDLE -> THINKING -> SPEAKING
        self.state_manager.transition_to(AssistantState.THINKING)
        self.state_manager.transition_to(AssistantState.SPEAKING, status_message="Speaking...")
        player.play(self.sample_wav, block=False)
        time.sleep(0.05)

        # Drain bridge event queue to sync state
        bridge._poll_backend_events()
        assert bridge._last_state == "SPEAKING"

        # Calling start_voice_interaction while SPEAKING must act as an interrupt
        interrupted = bridge.start_voice_interaction()
        assert interrupted is True
        assert not player.is_playing
        assert self.state_manager.get_snapshot().state == AssistantState.IDLE

        bridge.close()
        adapter.close()
