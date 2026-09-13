"""Microphone Audio Capture Subsystem for J.A.R.V.I.S.

Provides the production audio acquisition layer implementing AudioProvider:
coordinates hardware/simulated microphone capture, writes standardized 16kHz
mono 16-bit PCM WAV audio files (Whisper-compatible), and manages thread-safe
lifecycle events and Service Container registration.
"""

from __future__ import annotations

import sounddevice as sd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Final, Optional, Union
import uuid
import wave

import sounddevice as sd
import soundfile as sf

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.voice.models import (
    AudioProvider,
    AudioProviderError,
    MicrophoneError,
    RecordingError,
)

# Lifecycle Event Constants
EVENT_MICROPHONE_STARTED: Final[str] = "microphone.started"
EVENT_MICROPHONE_STOPPED: Final[str] = "microphone.stopped"
EVENT_MICROPHONE_RECORDING: Final[str] = "microphone.recording"
EVENT_MICROPHONE_RECORDED: Final[str] = "microphone.recorded"
EVENT_SOURCE_MICROPHONE: Final[str] = "microphone"

# Audio Standard Constants (Whisper and STT standard)
DEFAULT_SAMPLE_RATE: Final[int] = 16000  # 16 kHz
DEFAULT_CHANNELS: Final[int] = 1         # Mono
DEFAULT_SAMPLE_WIDTH: Final[int] = 2     # 16-bit PCM (2 bytes)
DEFAULT_CHUNK_SIZE: Final[int] = 1024
DEFAULT_RECORD_DURATION: Final[float] = 6.0
DEFAULT_SPEECH_THRESHOLD: Final[float] = 0.02
DEFAULT_SILENCE_THRESHOLD: Final[float] = 0.015
DEFAULT_SILENCE_TIMEOUT: Final[float] = 0.8
DEFAULT_MIN_SPEECH_DURATION: Final[float] = 0.2
DEFAULT_MAX_RECORD_DURATION: Final[float] = 10.0


class MicrophoneRecorder(AudioProvider):
    """Production audio recorder capturing microphone input to standard WAV files.

    Inherits from AudioProvider to seamlessly integrate with VoicePipeline.
    Features:
    - Standardized 16kHz, mono, 16-bit PCM WAV format.
    - Thread-safe start, stop, and active state management.
    - Hardware-ready with graceful simulated audio frame generation for headless/CI.
    - Dynamic RMS voice activity detection (VAD) with configurable thresholds.
    - Context manager interface support.
    - Synchronous and asynchronous capture methods.
    - EventBus lifecycle notifications.
    - Registers as 'microphone' in ServiceContainer.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        record_duration: float = DEFAULT_RECORD_DURATION,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
        audio_source_callback: Optional[Callable[[int], bytes]] = None,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        vad_enabled: bool = True,
        speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION,
        max_record_duration: float = DEFAULT_MAX_RECORD_DURATION,
    ) -> None:
        """Initialize the MicrophoneRecorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000).
            channels: Number of audio channels (default: 1 for mono).
            sample_width: Bytes per sample (default: 2 for 16-bit PCM).
            chunk_size: Buffer chunk size in frames.
            record_duration: Default recording duration in seconds.
            output_dir: Directory where temporary recordings are stored.
            config: Optional Settings instance. Defaults to container or global settings.
            logger: Optional Logger instance. Defaults to 'MICROPHONE' logger.
            container_instance: Optional ServiceContainer. Defaults to global container.
            event_bus_instance: Optional EventBus. Defaults to container or global event bus.
            auto_register_in_container: If True, registers 'microphone' in ServiceContainer.
            audio_source_callback: Optional callback(num_frames) -> bytes returning raw PCM data.
            amplitude_callback: Optional callback(float) -> None receiving normalized RMS amplitude (0.0-1.0).
            vad_enabled: Whether automatic end-of-speech VAD is enabled by default.
            speech_threshold: Normalized RMS threshold [0.0, 1.0] indicating speech onset.
            silence_threshold: Normalized RMS threshold [0.0, 1.0] below which is silence.
            silence_timeout: Seconds of continuous silence after speech before auto-stopping.
            min_speech_duration: Minimum seconds of speech required before silence triggers stop.
            max_record_duration: Maximum duration timeout in seconds for any recording.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if sample_width <= 0:
            raise ValueError(f"sample_width must be positive, got {sample_width}")
        if record_duration <= 0:
            raise ValueError(f"record_duration must be positive, got {record_duration}")

        self._lock = threading.RLock()
        self._active: bool = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MicrophoneRecorder")

        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._chunk_size = chunk_size
        self._record_duration = float(record_duration)
        self._audio_source_callback = audio_source_callback
        self._amplitude_callback = amplitude_callback

        # VAD parameters
        self._vad_enabled = bool(vad_enabled)
        self._speech_threshold = max(0.001, float(speech_threshold))
        self._silence_threshold = max(0.0005, float(silence_threshold))
        self._silence_timeout = max(0.05, float(silence_timeout))
        self._min_speech_duration = max(0.05, float(min_speech_duration))
        self._max_record_duration = max(0.1, float(max_record_duration))

        # 1. Dependency Resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency Resolution: Logger
        self._logger = logger if logger is not None else get_logger("MICROPHONE")

        # 3. Dependency Resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Dependency Resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = event_bus

        # 5. Output Directory Configuration
        if output_dir is not None:
            self._output_dir = Path(output_dir).resolve()
        elif hasattr(self._config, "data_dir") and self._config.data_dir:
            self._output_dir = Path(self._config.data_dir) / "audio"
        else:
            self._output_dir = Path(tempfile.gettempdir()) / "jarvis_audio"

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._logger.warning(f"Could not create audio output directory '{self._output_dir}': {exc}")

        # 6. Service Container Registration
        if auto_register_in_container:
            try:
                self._container.register_singleton("microphone", self, allow_override=True)
                self._logger.debug("Registered 'microphone' singleton in Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register MicrophoneRecorder in container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Sampling rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def sample_width(self) -> int:
        """Sample width in bytes."""
        return self._sample_width

    @property
    def record_duration(self) -> float:
        """Default recording duration in seconds."""
        with self._lock:
            return self._record_duration

    @record_duration.setter
    def record_duration(self, val: float) -> None:
        """Update the default recording duration in seconds."""
        if val <= 0:
            raise ValueError(f"record_duration must be positive, got {val}")
        with self._lock:
            self._record_duration = float(val)

    @property
    def output_dir(self) -> Path:
        """Directory for audio recordings."""
        return self._output_dir

    @property
    def amplitude_callback(self) -> Optional[Callable[[float], None]]:
        """Active amplitude callback receiving normalized RMS (0.0 - 1.0)."""
        with self._lock:
            return self._amplitude_callback

    @amplitude_callback.setter
    def amplitude_callback(self, cb: Optional[Callable[[float], None]]) -> None:
        """Set or update the amplitude callback."""
        with self._lock:
            self._amplitude_callback = cb

    @property
    def vad_enabled(self) -> bool:
        """Return whether dynamic VAD auto-stop is enabled by default."""
        with self._lock:
            return self._vad_enabled

    @vad_enabled.setter
    def vad_enabled(self, val: bool) -> None:
        """Configure whether dynamic VAD auto-stop is enabled by default."""
        with self._lock:
            self._vad_enabled = bool(val)

    @property
    def speech_threshold(self) -> float:
        """Normalized RMS threshold [0.0, 1.0] for speech onset."""
        with self._lock:
            return self._speech_threshold

    @speech_threshold.setter
    def speech_threshold(self, val: float) -> None:
        """Set normalized RMS threshold for speech onset."""
        with self._lock:
            self._speech_threshold = max(0.001, float(val))

    @property
    def silence_threshold(self) -> float:
        """Normalized RMS threshold [0.0, 1.0] below which is treated as silence."""
        with self._lock:
            return self._silence_threshold

    @silence_threshold.setter
    def silence_threshold(self, val: float) -> None:
        """Set normalized RMS threshold for silence."""
        with self._lock:
            self._silence_threshold = max(0.0005, float(val))

    @property
    def silence_timeout(self) -> float:
        """Duration in seconds of trailing silence before auto-stopping."""
        with self._lock:
            return self._silence_timeout

    @silence_timeout.setter
    def silence_timeout(self, val: float) -> None:
        """Set duration in seconds of trailing silence before auto-stopping."""
        with self._lock:
            self._silence_timeout = max(0.1, float(val))

    @property
    def min_speech_duration(self) -> float:
        """Minimum duration in seconds of speech required before silence triggers stop."""
        with self._lock:
            return self._min_speech_duration

    @min_speech_duration.setter
    def min_speech_duration(self, val: float) -> None:
        """Set minimum duration in seconds of speech required."""
        with self._lock:
            self._min_speech_duration = max(0.05, float(val))

    @property
    def max_record_duration(self) -> float:
        """Maximum recording timeout duration in seconds."""
        with self._lock:
            return self._max_record_duration

    @max_record_duration.setter
    def max_record_duration(self, val: float) -> None:
        """Set maximum recording timeout duration in seconds."""
        with self._lock:
            self._max_record_duration = max(0.1, float(val))

    # --------------------------------------------------------------------------
    # Lifecycle Control
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Activate the microphone recorder.

        Emits 'microphone.started' event on the EventBus.
        """
        with self._lock:
            if self._active:
                return
            self._active = True

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_STARTED,
                payload={
                    "sample_rate": self._sample_rate,
                    "channels": self._channels,
                    "timestamp": time.time(),
                },
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.info("MicrophoneRecorder started and listening.")

    def stop(self) -> None:
        """Deactivate the microphone recorder.

        Emits 'microphone.stopped' event on the EventBus.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False

        self._report_amplitude_val(0.0)

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_STOPPED,
                payload={"timestamp": time.time()},
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.info("MicrophoneRecorder stopped.")

    def is_active(self) -> bool:
        """Check whether the microphone recorder is active."""
        with self._lock:
            return self._active

    def __enter__(self) -> MicrophoneRecorder:
        """Context manager entry activates the recorder."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit stops the recorder."""
        self.stop()

    # --------------------------------------------------------------------------
    # Audio Frame Generation & Capture
    # --------------------------------------------------------------------------

    def calculate_amplitude(self, data: Union[np.ndarray, bytes, bytearray]) -> float:
        """Calculate normalized RMS amplitude (0.0 - 1.0) from int16 PCM data."""
        try:
            if isinstance(data, (bytes, bytearray)):
                samples = np.frombuffer(data, dtype=np.int16)
            elif isinstance(data, np.ndarray):
                samples = data
            else:
                return 0.0

            if samples.size == 0:
                return 0.0

            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            return float(min(1.0, max(0.0, rms / 32768.0)))
        except Exception:
            return 0.0

    def _report_amplitude(self, data: Union[np.ndarray, bytes, bytearray]) -> None:
        """Calculate normalized RMS amplitude (0.0 - 1.0) and dispatch to amplitude_callback safely."""
        amp = self.calculate_amplitude(data)
        self._report_amplitude_val(amp)

    def _report_amplitude_val(self, amp: float) -> None:
        """Dispatch explicit normalized amplitude value (0.0 - 1.0) to amplitude_callback safely."""
        if self._amplitude_callback is None:
            return
        try:
            clamped = float(min(1.0, max(0.0, amp)))
            self._amplitude_callback(clamped)
        except Exception as exc:
            self._logger.debug("amplitude_callback error ignored: %s", exc)

    def _generate_pcm_frames(
        self,
        num_frames: int,
        *,
        vad_active: bool = False,
        max_duration: Optional[float] = None,
    ) -> bytes:
        """Record audio from the system microphone or custom audio callback."""
        if self._audio_source_callback is not None:
            try:
                chunks: list[bytes] = []
                frames_left = num_frames
                effective_max_sec = max_duration if max_duration is not None else (num_frames / float(self._sample_rate))
                speech_detected = False
                speech_frames_accum = 0
                min_speech_frames = int(self._min_speech_duration * self._sample_rate)
                last_speech_sec = 0.0
                elapsed_sim_sec = 0.0

                while frames_left > 0:
                    take = min(self._chunk_size, frames_left)
                    data = self._audio_source_callback(take)
                    if isinstance(data, (bytes, bytearray)):
                        raw_bytes = bytes(data)
                    else:
                        raw_bytes = b"\x00" * (take * self._sample_width * self._channels)

                    amp = self.calculate_amplitude(raw_bytes)
                    self._report_amplitude_val(amp)
                    chunks.append(raw_bytes)
                    frames_left -= take

                    chunk_sec = take / float(self._sample_rate)
                    elapsed_sim_sec += chunk_sec

                    if vad_active:
                        if amp >= self._speech_threshold:
                            speech_frames_accum += take
                            last_speech_sec = elapsed_sim_sec
                            if speech_frames_accum >= min_speech_frames:
                                speech_detected = True
                        elif speech_detected:
                            if (elapsed_sim_sec - last_speech_sec) >= self._silence_timeout:
                                break
                        if elapsed_sim_sec >= effective_max_sec:
                            break

                return b"".join(chunks)
            except Exception as exc:
                self._logger.warning(f"Error in audio_source_callback: {exc}")

        # Streaming capture fallback
        total_samples = num_frames
        chunks_recorded: list[np.ndarray] = []

        def _stream_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            chunk = indata.copy()
            chunks_recorded.append(chunk)
            self._report_amplitude(chunk)

        try:
            target_sec = total_samples / float(self._sample_rate)
            t_start = time.perf_counter()
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=self._chunk_size,
                callback=_stream_callback,
            ):
                while (time.perf_counter() - t_start) < target_sec:
                    sd.sleep(10)
        except Exception as stream_exc:
            self._logger.debug(f"Streaming input failed; falling back to rec: {stream_exc}")
            recording = sd.rec(
                num_frames,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
            )
            sd.wait()
            self._report_amplitude(recording)
            return recording.tobytes()

        if chunks_recorded:
            all_frames = np.concatenate(chunks_recorded, axis=0)
        else:
            all_frames = np.zeros((0, self._channels), dtype=np.int16)

        if len(all_frames) < total_samples:
            pad_len = total_samples - len(all_frames)
            if self._channels == 1:
                all_frames = np.pad(all_frames, ((0, pad_len), (0, 0)) if all_frames.ndim == 2 else (0, pad_len), "constant")
            else:
                all_frames = np.pad(all_frames, ((0, pad_len), (0, 0)), "constant")
        elif len(all_frames) > total_samples:
            all_frames = all_frames[:total_samples]

        return all_frames.tobytes()

    def record(
        self,
        duration: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None,
        *,
        use_vad: Optional[bool] = None,
    ) -> Path:
        """Record audio for the specified duration or automatically until end of speech.

        Args:
            duration: Explicit duration in seconds. If None, uses max_record_duration with VAD.
            output_path: Optional target file path. If None, generates in output_dir.
            use_vad: If True, forces dynamic VAD auto-stop. If False, records fixed duration.
                     If None, activates VAD when duration is None, and fixed-duration when duration is given.

        Returns:
            Resolved Path to the recorded .wav file.

        Raises:
            MicrophoneError: If recording fails or duration is invalid.
        """
        if use_vad is True:
            vad_active = True
        elif use_vad is False:
            vad_active = False
        else:
            vad_active = self._vad_enabled if duration is None else False

        max_duration = float(duration) if duration is not None else self._max_record_duration
        if max_duration <= 0:
            raise MicrophoneError(f"Recording duration must be positive, got {max_duration}")

        if output_path is not None:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            unique_name = f"mic_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            out_file = self._output_dir / unique_name

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_RECORDING,
                payload={"duration": max_duration, "output_path": str(out_file), "vad_active": vad_active},
                source=EVENT_SOURCE_MICROPHONE,
            )

        recorded_duration = max_duration
        try:
            if self._audio_source_callback is not None:
                total_req_frames = int(max_duration * self._sample_rate)
                pcm_bytes = self._generate_pcm_frames(
                    total_req_frames,
                    vad_active=vad_active,
                    max_duration=max_duration,
                )
                with wave.open(str(out_file), "wb") as wf:
                    wf.setnchannels(self._channels)
                    wf.setsampwidth(self._sample_width)
                    wf.setframerate(self._sample_rate)
                    wf.writeframes(pcm_bytes)
                bytes_per_sec = self._sample_rate * self._channels * self._sample_width
                recorded_duration = len(pcm_bytes) / float(bytes_per_sec) if bytes_per_sec > 0 else max_duration
            else:
                total_samples = int(max_duration * self._sample_rate)
                chunks_recorded: list[np.ndarray] = []
                stop_event = threading.Event()
                speech_detected = False
                speech_frames_accum = 0
                last_speech_time = [0.0]
                min_speech_frames = int(self._min_speech_duration * self._sample_rate)

                def _stream_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
                    chunk = indata.copy()
                    chunks_recorded.append(chunk)
                    amp = self.calculate_amplitude(chunk)
                    self._report_amplitude_val(amp)

                    if vad_active:
                        nonlocal speech_detected, speech_frames_accum
                        now = time.perf_counter()
                        if amp >= self._speech_threshold:
                            speech_frames_accum += frames
                            last_speech_time[0] = now
                            if speech_frames_accum >= min_speech_frames:
                                speech_detected = True
                        elif speech_detected:
                            if (now - last_speech_time[0]) >= self._silence_timeout:
                                stop_event.set()

                try:
                    t_start = time.perf_counter()
                    with sd.InputStream(
                        samplerate=self._sample_rate,
                        channels=self._channels,
                        dtype="int16",
                        blocksize=self._chunk_size,
                        callback=_stream_callback,
                    ):
                        while not stop_event.is_set() and (time.perf_counter() - t_start) < max_duration:
                            sd.sleep(10)
                except Exception as stream_exc:
                    self._logger.debug(f"Streaming InputStream failed; falling back to rec: {stream_exc}")
                    recording = sd.rec(
                        total_samples,
                        samplerate=self._sample_rate,
                        channels=self._channels,
                        dtype="int16",
                    )
                    sd.wait()

                    cutoff_sample = total_samples
                    if vad_active:
                        rec_speech_frames = 0
                        rec_speech_detected = False
                        rec_last_speech_sample = 0
                        silence_timeout_samples = int(self._silence_timeout * self._sample_rate)

                        for offset in range(0, len(recording), self._chunk_size):
                            sub_chunk = recording[offset : offset + self._chunk_size]
                            sub_amp = self.calculate_amplitude(sub_chunk)
                            self._report_amplitude_val(sub_amp)
                            if sub_amp >= self._speech_threshold:
                                rec_speech_frames += len(sub_chunk)
                                rec_last_speech_sample = offset + len(sub_chunk)
                                if rec_speech_frames >= min_speech_frames:
                                    rec_speech_detected = True
                            elif rec_speech_detected:
                                if (offset - rec_last_speech_sample) >= silence_timeout_samples:
                                    cutoff_sample = offset + len(sub_chunk)
                                    break
                        chunks_recorded = [recording[:cutoff_sample]]
                    else:
                        for offset in range(0, len(recording), self._chunk_size):
                            sub_chunk = recording[offset : offset + self._chunk_size]
                            self._report_amplitude_val(self.calculate_amplitude(sub_chunk))
                        chunks_recorded = [recording]

                if chunks_recorded:
                    all_frames = np.concatenate(chunks_recorded, axis=0)
                else:
                    all_frames = np.zeros((0, self._channels), dtype=np.int16)

                if not vad_active:
                    if len(all_frames) < total_samples:
                        pad_len = total_samples - len(all_frames)
                        if self._channels == 1:
                            all_frames = np.pad(all_frames, ((0, pad_len), (0, 0)) if all_frames.ndim == 2 else (0, pad_len), "constant")
                        else:
                            all_frames = np.pad(all_frames, ((0, pad_len), (0, 0)), "constant")
                    elif len(all_frames) > total_samples:
                        all_frames = all_frames[:total_samples]

                sf.write(
                    str(out_file),
                    all_frames,
                    self._sample_rate,
                    subtype="PCM_16",
                )
                recorded_duration = len(all_frames) / float(self._sample_rate) if self._sample_rate > 0 else max_duration

        except Exception as exc:
            err_msg = f"Failed to record audio to '{out_file}': {exc}"
            self._logger.error(err_msg, exc_info=True)
            raise RecordingError(err_msg) from exc
        finally:
            self._report_amplitude_val(0.0)

        if self._event_bus is not None:
            self._event_bus.publish(
                event=EVENT_MICROPHONE_RECORDED,
                payload={
                    "output_path": str(out_file),
                    "duration": recorded_duration,
                    "file_size": out_file.stat().st_size if out_file.exists() else 0,
                },
                source=EVENT_SOURCE_MICROPHONE,
            )

        self._logger.debug(f"Audio recorded successfully to '{out_file}' ({recorded_duration:.2f}s).")
        return out_file

    def capture_audio(
        self,
        duration: Optional[float] = None,
        *,
        use_vad: Optional[bool] = None,
    ) -> Optional[Path]:
        """Capture an audio sample if recorder is active.

        Implements AudioProvider.capture_audio contract.

        Returns:
            Path to recorded audio (.wav) file, or None if recorder is not active.
        """
        with self._lock:
            if not self._active:
                self._logger.debug("MicrophoneRecorder.capture_audio called while inactive; returning None.")
                return None

        try:
            return self.record(duration=duration, use_vad=use_vad)
        except Exception as exc:
            self._logger.error(f"Error capturing audio from microphone: {exc}")
            raise AudioProviderError(f"Microphone audio capture failed: {exc}") from exc

    async def capture_audio_async(
        self,
        duration: Optional[float] = None,
        *,
        use_vad: Optional[bool] = None,
    ) -> Optional[Path]:
        """Asynchronously capture an audio sample if recorder is active.

        Runs capture_audio() in a background thread pool.

        Returns:
            Path to recorded audio (.wav) file, or None if recorder is not active.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.capture_audio(duration=duration, use_vad=use_vad),
        )

    def close(self) -> None:
        """Release recorder resources and shutdown executor."""
        self.stop()
        self._executor.shutdown(wait=False)


# ------------------------------------------------------------------------------
# Default Singleton Instance
# ------------------------------------------------------------------------------
microphone_recorder = MicrophoneRecorder(auto_register_in_container=True)
microphone = microphone_recorder

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_RECORD_DURATION",
    "DEFAULT_MIN_SPEECH_DURATION",
    "DEFAULT_RECORD_DURATION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_WIDTH",
    "DEFAULT_SILENCE_THRESHOLD",
    "DEFAULT_SILENCE_TIMEOUT",
    "DEFAULT_SPEECH_THRESHOLD",
    "EVENT_MICROPHONE_RECORDED",
    "EVENT_MICROPHONE_RECORDING",
    "EVENT_MICROPHONE_STARTED",
    "EVENT_MICROPHONE_STOPPED",
    "EVENT_SOURCE_MICROPHONE",
    "MicrophoneRecorder",
    "microphone",
    "microphone_recorder",
]
