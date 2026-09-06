"""Voice Pipeline Subsystem for J.A.R.V.I.S.

Provides the end-to-end voice orchestration layer integrating audio input,
Speech-to-Text (STT), wake word detection, command routing, conversational memory,
and Text-to-Speech (TTS) response synthesis within an async-ready state machine.
"""

from __future__ import annotations

from app.voice.models import (
    AudioProvider,
    AudioProviderError,
    InvalidStateError,
    PlaceholderAudioProvider,
    PipelineExecutionError,
    PipelineNotRunningError,
    VoicePipelineError,
    VoicePipelineResult,
    VoiceState,
)
from app.voice.pipeline import (
    EVENT_SOURCE_VOICE,
    EVENT_VOICE_FAILED,
    EVENT_VOICE_LISTENING,
    EVENT_VOICE_PROCESSING,
    EVENT_VOICE_RESPONDING,
    EVENT_VOICE_STARTED,
    EVENT_VOICE_STOPPED,
    VoicePipeline,
    voice_pipeline,
)

__all__ = [
    "AudioProvider",
    "AudioProviderError",
    "EVENT_SOURCE_VOICE",
    "EVENT_VOICE_FAILED",
    "EVENT_VOICE_LISTENING",
    "EVENT_VOICE_PROCESSING",
    "EVENT_VOICE_RESPONDING",
    "EVENT_VOICE_STARTED",
    "EVENT_VOICE_STOPPED",
    "InvalidStateError",
    "PlaceholderAudioProvider",
    "PipelineExecutionError",
    "PipelineNotRunningError",
    "VoicePipeline",
    "VoicePipelineError",
    "VoicePipelineResult",
    "VoiceState",
    "voice_pipeline",
]
