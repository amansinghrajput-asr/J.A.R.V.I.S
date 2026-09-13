"""Phase 25 Tests — Microphone Streaming & Real-Time Amplitude Telemetry.

Verifies:
1. Normalized RMS calculation: silence -> 0, full scale -> ~1.0, clamped in [0, 1].
2. Continuous chunk-level callbacks during recording.
3. Callback failure isolation (exceptions do not abort recording).
4. Amplitude callback property setter/getter.
5. VoiceConversationEngine binding to AssistantStateManager.
6. Amplitude lifecycle reset to 0.0 upon completion, stop, or failure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.core.container import ServiceContainer
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.voice.engine import VoiceConversationEngine
from app.voice.microphone import MicrophoneRecorder


class TestMicrophoneStreaming(unittest.TestCase):
    """Test suite for MicrophoneRecorder streaming telemetry and RMS calculations."""

    def setUp(self) -> None:
        self.sample_rate = 16000
        self.chunk_size = 1024

    def test_rms_calculation_silence_and_signals(self) -> None:
        """Test RMS calculation on silence, tones, and clamping."""
        collected: list[float] = []

        recorder = MicrophoneRecorder(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            auto_register_in_container=False,
            amplitude_callback=lambda amp: collected.append(amp),
        )

        # 1. Silence (zeros) -> 0.0
        silence = np.zeros(self.chunk_size, dtype=np.int16)
        recorder._report_amplitude(silence)
        self.assertEqual(len(collected), 1)
        self.assertAlmostEqual(collected[0], 0.0, places=4)

        # 2. Medium amplitude signal
        # Sine wave with amplitude 16384 (half-scale)
        t = np.linspace(0, 1, self.chunk_size, endpoint=False)
        half_scale = (16384 * np.sin(2 * math.pi * 440 * t)).astype(np.int16)
        recorder._report_amplitude(half_scale)
        self.assertEqual(len(collected), 2)
        # Expected RMS of sine wave with peak A is A / sqrt(2)
        # 16384 / sqrt(2) ≈ 11585. 11585 / 32768 ≈ 0.3535
        self.assertGreater(collected[1], 0.3)
        self.assertLess(collected[1], 0.4)

        # 3. Full scale square wave (32767) -> ~1.0
        full_scale = np.full(self.chunk_size, 32767, dtype=np.int16)
        recorder._report_amplitude(full_scale)
        self.assertEqual(len(collected), 3)
        self.assertAlmostEqual(collected[2], 32767.0 / 32768.0, places=3)
        self.assertLessEqual(collected[2], 1.0)
        self.assertGreaterEqual(collected[2], 0.0)

        # 4. Value reporting clamping
        recorder._report_amplitude_val(1.5)
        self.assertEqual(collected[-1], 1.0)
        recorder._report_amplitude_val(-0.5)
        self.assertEqual(collected[-1], 0.0)

    def test_continuous_callbacks_chunked_streaming(self) -> None:
        """Verify multiple chunks produce multiple callback invocations within [0, 1]."""
        collected: list[float] = []

        # Synthetic source providing 4 chunks (4096 samples)
        current_chunk = 0
        def _mock_source(num_frames: int) -> bytes:
            nonlocal current_chunk
            current_chunk += 1
            # alternating loudness
            amp = 8000 * current_chunk
            arr = np.full(num_frames, min(32000, amp), dtype=np.int16)
            return arr.tobytes()

        recorder = MicrophoneRecorder(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            auto_register_in_container=False,
            audio_source_callback=_mock_source,
            amplitude_callback=lambda amp: collected.append(amp),
        )

        out_path = recorder.record(duration=0.25)  # 0.25 * 16000 = 4000 samples (~4 chunks)
        self.assertTrue(out_path.exists())
        # Should have received at least 4 chunk callbacks + 1 final reset to 0.0
        self.assertGreaterEqual(len(collected), 4)
        for amp in collected:
            self.assertIsInstance(amp, float)
            self.assertGreaterEqual(amp, 0.0)
            self.assertLessEqual(amp, 1.0)
        # Final value must be reset to 0.0
        self.assertEqual(collected[-1], 0.0)

    def test_callback_failure_isolation(self) -> None:
        """Verify callback exceptions never crash audio recording."""
        def _exploding_callback(amp: float) -> None:
            raise RuntimeError("UI or subscriber exploded unexpectedly")

        def _mock_source(num_frames: int) -> bytes:
            return np.zeros(num_frames, dtype=np.int16).tobytes()

        recorder = MicrophoneRecorder(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            auto_register_in_container=False,
            audio_source_callback=_mock_source,
            amplitude_callback=_exploding_callback,
        )

        # Recording must succeed despite callback raising
        out_path = recorder.record(duration=0.1)
        self.assertTrue(out_path.exists())

    def test_amplitude_callback_property(self) -> None:
        """Test getter and setter for amplitude_callback property."""
        recorder = MicrophoneRecorder(
            auto_register_in_container=False,
            amplitude_callback=None,
        )
        self.assertIsNone(recorder.amplitude_callback)

        dummy_cb = lambda a: None
        recorder.amplitude_callback = dummy_cb
        self.assertEqual(recorder.amplitude_callback, dummy_cb)

        recorder.amplitude_callback = None
        self.assertIsNone(recorder.amplitude_callback)

    def test_lifecycle_amplitude_reset_on_stop(self) -> None:
        """Verify amplitude resets to 0.0 when recorder is stopped."""
        collected: list[float] = []
        recorder = MicrophoneRecorder(
            auto_register_in_container=False,
            amplitude_callback=lambda amp: collected.append(amp),
        )

        recorder.start()
        recorder._report_amplitude_val(0.75)
        self.assertEqual(collected[-1], 0.75)

        recorder.stop()
        self.assertEqual(collected[-1], 0.0)

    def test_voice_engine_state_manager_telemetry_integration(self) -> None:
        """Verify VoiceConversationEngine binds recorder amplitude to AssistantStateManager."""
        test_container = ServiceContainer()
        state_mgr = AssistantStateManager()
        test_container.register_singleton("state_manager", state_mgr)

        recorded_amplitudes: list[float] = []
        def _mock_audio_source(num_frames: int) -> bytes:
            arr = np.full(num_frames, 15000, dtype=np.int16)
            return arr.tobytes()

        recorder = MicrophoneRecorder(
            sample_rate=16000,
            chunk_size=1024,
            container_instance=test_container,
            audio_source_callback=_mock_audio_source,
            auto_register_in_container=False,
        )

        # Mock STT to prevent real speech model loading
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = MagicMock(text="hello test")

        mock_router = MagicMock()
        mock_router.route.return_value = "Response test"

        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = MagicMock()

        engine = VoiceConversationEngine(
            recorder=recorder,
            stt=mock_stt,
            router=mock_router,
            tts=mock_tts,
            container_instance=test_container,
            auto_register_in_container=False,
        )

        # Track state manager mic amplitude changes
        def _on_state_event(evt: Any) -> None:
            if evt.event_type == "telemetry.mic_amplitude":
                payload = dict(evt.payload)
                recorded_amplitudes.append(payload.get("amplitude", 0.0))

        state_mgr.add_listener(_on_state_event)

        # Execute one conversation turn
        res = engine.listen_once(duration=0.1)
        self.assertTrue(res.success)

        # Assert that telemetry events reached the state manager
        self.assertGreater(len(recorded_amplitudes), 0)
        for amp in recorded_amplitudes:
            self.assertGreaterEqual(amp, 0.0)
            self.assertLessEqual(amp, 1.0)

        # At the end, state manager amplitude must be reset to 0.0
        snap = state_mgr.get_snapshot()
        self.assertEqual(snap.mic_amplitude, 0.0)


if __name__ == "__main__":
    unittest.main()
