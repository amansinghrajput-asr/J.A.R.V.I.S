"""Comprehensive hardware-free integration test suite for Phase 27.5: Voice + Vision End-to-End.

Verifies the complete spoken interaction pipeline:
MicrophoneRecorder (mock) -> SpeechToText (mock) -> CommandRouter -> VisionSkills
-> SecureVisionManager -> OCRProvider / GeminiProvider -> SystemSkillResult
-> VoiceConversationEngine response formatting -> TextToSpeech (mock) -> AudioPlayer (mock).

SAFETY INVARIANTS:
1. 100% offline, deterministic, and hardware-free.
2. NO physical audio recording or microphone hardware access.
3. NO physical desktop capture, NO Win32 GDI calls.
4. NO real external network or Gemini API requests.
5. NO real audio device playback.
6. ZERO screenshot files written to disk.
7. ZERO raw pixels or base64 payloads in TTS, EventBus, or conversation memory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from app.ai.intent_router import IntentRouter, IntentType
from app.ai.models import AIResponse, ImagePart
from app.ai.planner.events import PlannerEventBus
from app.core.config import load_settings
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.router.intent import Intent
from app.router.router import CommandRouter
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy
from app.skills.system.vision_skills import VisionSkills
from app.stt.models import TranscriptionResult
from app.stt.transcriber import SpeechToText
from app.tts.models import SpeechResult
from app.tts.speaker import TextToSpeech
from app.vision.capture import DesktopCaptureEngine
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    OCRResult,
    OCRTextBlock,
    ScreenCapture,
    ScreenObservation,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.preprocessing import ImagePreprocessor
from app.vision.security import EphemeralBufferManager, SecureVisionManager, VisionSecurityPolicy
from app.voice.engine import (
    EVENT_ENGINE_COMPLETED,
    EVENT_ENGINE_FAILED,
    EVENT_ENGINE_RECORDING,
    EVENT_ENGINE_ROUTING,
    EVENT_ENGINE_SPEAKING,
    EVENT_ENGINE_TRANSCRIBING,
    VoiceConversationEngine,
)
from app.voice.microphone import MicrophoneRecorder
from app.voice.models import VoiceConversationResult
from app.voice.player import AudioPlayer


class TestVoiceVisionIntegration(unittest.TestCase):
    """Integration test suite for Phase 27.5 Voice + Vision end-to-end pipeline."""

    def setUp(self) -> None:
        """Set up isolated mock dependencies for hardware-free testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()
        self.config = load_settings()

        # State Manager
        self.state_manager = AssistantStateManager(event_bus=self.event_bus)
        self.container.register_singleton("state_manager", self.state_manager)

        # Vision Subsystem Mocks
        self.mock_engine = MagicMock(spec=DesktopCaptureEngine)
        self.fake_capture = ScreenCapture(
            raw_data=b"\x00\xFF\x00\xFF" * (64 * 64),
            width=64,
            height=64,
            source="window",
            bounds=WindowBounds(left=0, top=0, right=64, bottom=64),
            metadata={"window_title": "Visual Studio Code - main.py", "process_name": "Code.exe"},
        )
        self.mock_engine.get_active_window_handle.return_value = 12345
        self.mock_engine.capture_screen.return_value = self.fake_capture
        self.mock_engine.capture_active_window.return_value = self.fake_capture

        self.sec_policy = VisionSecurityPolicy()
        self.buffer_mgr = EphemeralBufferManager(event_bus_instance=self.event_bus)
        self.secure_vision = SecureVisionManager(
            engine=self.mock_engine,
            policy=self.sec_policy,
            buffer_manager=self.buffer_mgr,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Visual Studio Code - main.py", "Code.exe")
        )

        self.ocr_provider = MockOCRProvider()
        self.mock_gemini = MagicMock()
        self.mock_gemini.supports_multimodal = True
        self.mock_gemini.model = "gemini-2.5-flash"

        self.sys_sec_policy = SystemSecurityPolicy(event_bus=self.planner_bus)
        self.conf_manager = SystemConfirmationManager(event_bus=self.planner_bus)

        self.vision_skills = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ocr_provider=self.ocr_provider,
            ai_provider=self.mock_gemini,
            security_policy=self.sys_sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )

        # Skill Manager & Command Router
        self.skill_manager = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        self.skill_manager.register(self.vision_skills)

        self.router = CommandRouter(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            skill_manager_instance=self.skill_manager,
            auto_register_in_container=False,
        )

        # Voice Pipeline Mocks
        self.mock_recorder = MagicMock(spec=MicrophoneRecorder)
        self.mock_audio_file = Path(self.temp_dir.name) / "test_mic.wav"
        self.mock_audio_file.write_bytes(b"RIFFmockaudio")
        self.mock_recorder.record.return_value = self.mock_audio_file
        self.mock_recorder.amplitude_callback = None

        self.mock_stt = MagicMock(spec=SpeechToText)
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="what is on my screen", language="en"
        )

        self.mock_tts = MagicMock(spec=TextToSpeech)
        self.mock_tts_output = Path(self.temp_dir.name) / "tts_out.wav"
        self.mock_tts_output.write_bytes(b"RIFFmocktts")
        self.mock_speech_result = SpeechResult(
            text="Active window analyzed.",
            audio_path=self.mock_tts_output,
            voice="en-US-JennyNeural",
            duration=1.2,
        )
        self.mock_tts.synthesize.return_value = self.mock_speech_result

        self.mock_player = MagicMock(spec=AudioPlayer)
        self.mock_player.play.return_value = True
        self.mock_player.amplitude_callback = None

        # Voice Engine
        self.engine = VoiceConversationEngine(
            recorder=self.mock_recorder,
            stt=self.mock_stt,
            router=self.router,
            tts=self.mock_tts,
            player=self.mock_player,
            state_manager=self.state_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        """Clean up resources."""
        self.engine.close()
        self.temp_dir.cleanup()

    # =======================================================================
    # Group A: Conversational Response Formatting
    # =======================================================================

    def test_format_capture_screen_result(self) -> None:
        """1. Verify capture_screen produces concise human text, not dataclass repr."""
        res = SystemSkillResult(
            operation="capture_screen",
            success=True,
            data={"observation_id": "obs-1", "width": 1920, "height": 1080},
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "Screen captured successfully.")
        self.assertNotIn("SystemSkillResult", formatted)

    def test_format_read_screen_text_result(self) -> None:
        """2. Verify read_screen_text prefers the OCR text field."""
        res = SystemSkillResult(
            operation="read_screen_text",
            success=True,
            data={"text": "File Edit Selection View Go Run", "line_count": 1},
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "File Edit Selection View Go Run")

    def test_format_read_screen_text_empty_result(self) -> None:
        """3. Verify read_screen_text with empty text returns safe notification."""
        res = SystemSkillResult(
            operation="read_screen_text",
            success=True,
            data={"text": "", "line_count": 0},
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "No text was detected on your screen.")

    def test_format_explain_active_window_result(self) -> None:
        """4. Verify explain_active_window prefers visual summary field."""
        res = SystemSkillResult(
            operation="explain_active_window",
            success=True,
            data={"summary": "A Python source code file is open in Visual Studio Code."},
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "A Python source code file is open in Visual Studio Code.")

    def test_format_diagnose_screen_error_found(self) -> None:
        """5. Verify diagnose_screen_error with error_found formats error summary and fix."""
        res = SystemSkillResult(
            operation="diagnose_screen_error",
            success=True,
            data={
                "error_found": True,
                "error_title": "ModuleNotFoundError: No module named 'numpy'",
                "root_cause": "Missing dependency in python environment",
                "recommended_fix": "Run pip install numpy",
            },
        )
        formatted = self.engine._format_response(res)
        self.assertIn("ModuleNotFoundError", formatted)
        self.assertIn("Root cause: Missing dependency in python environment", formatted)
        self.assertIn("Recommended fix: Run pip install numpy", formatted)

    def test_format_diagnose_screen_error_none_detected(self) -> None:
        """6. Verify diagnose_screen_error with no error returns clean reassurance."""
        res = SystemSkillResult(
            operation="diagnose_screen_error",
            success=True,
            data={"error_found": False, "error_title": None},
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "No errors were detected on your screen.")

    def test_format_failure_result_sanitized(self) -> None:
        """7. Verify failed SystemSkillResult formats safe error without stack traces."""
        res = SystemSkillResult(
            operation="capture_screen",
            success=False,
            error="CaptureBlockedError: Window title matches protected category 'PASSWORDS'",
        )
        formatted = self.engine._format_response(res)
        self.assertEqual(formatted, "Screen capture was blocked by vision privacy policy.")
        self.assertNotIn("CaptureBlockedError", formatted)

    def test_format_compatibility_raw_string(self) -> None:
        """8. Verify _format_response preserves plain string responses."""
        self.assertEqual(self.engine._format_response("  All systems operational.  "), "All systems operational.")

    def test_format_compatibility_dict(self) -> None:
        """9. Verify _format_response preserves dictionary extraction."""
        self.assertEqual(self.engine._format_response({"response": "Done."}), "Done.")
        self.assertEqual(self.engine._format_response({"content": "Content."}), "Content.")

    def test_format_compatibility_ai_response(self) -> None:
        """10. Verify _format_response preserves AIResponse objects."""
        ai_resp = AIResponse(content="AI assistant response text", model="gemini-2.5-flash")
        self.assertEqual(self.engine._format_response(ai_resp), "AI assistant response text")

    # =======================================================================
    # Group B: Voice -> Vision Command Routing
    # =======================================================================

    def test_spoken_capture_screen_routes_to_vision_skills(self) -> None:
        """11. Verify spoken 'take a screenshot' routes directly to VisionSkills.capture_screen."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="take a screenshot", language="en")

        result = self.engine.listen_once()

        self.assertTrue(result.success)
        self.assertEqual(result.command, "take a screenshot")
        self.assertEqual(result.response_text, "Screen captured successfully.")
        self.assertTrue(self.mock_engine.capture_screen.called or self.mock_engine.capture_active_window.called)

    def test_spoken_read_text_routes_to_vision_skills(self) -> None:
        """12. Verify spoken 'read what's on my screen' routes to VisionSkills.read_screen_text."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="read what's on my screen", language="en")
        self.ocr_provider.extract_text = MagicMock(
            return_value=OCRResult(
                text="Active window terminal logs: Build complete.",
                blocks=[OCRTextBlock(text="Active window terminal logs: Build complete.", bounds=WindowBounds(0, 0, 10, 10), confidence=0.99)],
                duration=0.05,
            )
        )

        result = self.engine.listen_once()

        self.assertTrue(result.success)
        self.assertIn("Build complete", result.response_text)
        self.mock_tts.synthesize.assert_called_with("Active window terminal logs: Build complete.")

    def test_spoken_explain_window_routes_to_vision_skills(self) -> None:
        """13. Verify spoken 'explain this window' routes to VisionSkills.explain_active_window."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="explain this window", language="en")
        self.mock_gemini.generate.return_value = AIResponse(
            content="This window displays the user's IDE with main.py open.",
            model="gemini-2.5-flash",
        )

        result = self.engine.listen_once()

        self.assertTrue(result.success)
        self.assertEqual(result.response_text, "This window displays the user's IDE with main.py open.")
        self.mock_gemini.generate.assert_called_once()

    def test_spoken_diagnose_error_routes_to_vision_skills(self) -> None:
        """14. Verify spoken 'diagnose this error' routes to VisionSkills.diagnose_screen_error."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="diagnose this error", language="en")
        self.mock_gemini.generate.return_value = AIResponse(
            content=(
                "Error Found: True\n"
                "Error Title: SyntaxError: invalid syntax\n"
                "Root Cause: Missing colon at end of line 42\n"
                "Recommended Fix: Add a colon after the def statement"
            ),
            model="gemini-2.5-flash",
        )

        result = self.engine.listen_once()

        self.assertTrue(result.success)
        self.assertIn("SyntaxError: invalid syntax", result.response_text)
        self.assertIn("Missing colon at end of line 42", result.response_text)
        self.assertIn("Add a colon after the def statement", result.response_text)

    def test_vision_skills_specialized_precedence(self) -> None:
        """15. Verify specialized VisionSkills has precedence over generic fallback."""
        matched_skill, _ = self.router._find_matching_skill(
            Intent(
                raw_command="what is on my screen",
                normalized_command="what is on my screen",
                intent_name="vision",
            )
        )
        self.assertIsNotNone(matched_skill)
        self.assertEqual(matched_skill.name, "vision")

    def test_no_duplicate_routing_or_events(self) -> None:
        """16. Verify single command.routed event published during voice interaction."""
        routed_events: list[Event] = []
        self.event_bus.subscribe("command.routed", routed_events.append)

        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")
        self.engine.listen_once()

        self.assertEqual(len(routed_events), 1)
        self.assertEqual(routed_events[0].payload["skill_name"], "vision")

    # =======================================================================
    # Group C: Security Boundary & Privacy Guardrails
    # =======================================================================

    def test_sensitive_password_manager_blocked(self) -> None:
        """17. Verify password manager window blocks capture without hardware GDI access."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Bitwarden - Vault", "bitwarden.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)
        self.assertEqual(self.mock_engine.capture_screen.call_count, 0)
        self.assertTrue(self.buffer_mgr.is_empty)

    def test_credential_window_blocked(self) -> None:
        """18. Verify credential dialog window blocks capture."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Windows Security - Smart Card Authentication", "credentialuibroker.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="read screen text", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)
        self.assertEqual(self.mock_engine.capture_screen.call_count, 0)

    def test_banking_window_blocked(self) -> None:
        """19. Verify banking / financial active window blocks capture."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Chase Bank - Online Banking", "chrome.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="explain this window", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)
        self.mock_gemini.generate.assert_not_called()

    def test_private_incognito_browsing_blocked(self) -> None:
        """20. Verify private/incognito browsing window blocks capture."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("New Incognito Tab - Google Chrome", "chrome.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="diagnose this error", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)
        self.mock_gemini.generate.assert_not_called()

    def test_security_block_never_touches_gdi_backend(self) -> None:
        """21. Invariant: backend capture count remains zero when blocked."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("1Password - Unlock Vault", "1password.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="look at my screen", language="en")

        initial_calls = self.mock_engine.capture_screen.call_count + self.mock_engine.capture_active_window.call_count
        self.engine.listen_once()

        self.assertEqual(
            self.mock_engine.capture_screen.call_count + self.mock_engine.capture_active_window.call_count,
            initial_calls,
        )

    def test_security_block_safe_result_returned(self) -> None:
        """22. Verify blocked capture error does not leak passwords or sensitive tokens."""
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Secret Vault - Master Key", "keepassxc.exe")
        )
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertNotIn("Master Key", str(result.error))

    # =======================================================================
    # Group D: Vision -> TTS & Audio Playback
    # =======================================================================

    def test_tts_receives_clean_speech_not_dataclass_repr(self) -> None:
        """23. Verify TTS synthesizer receives clean human text, never SystemSkillResult repr."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        self.engine.listen_once()

        self.mock_tts.synthesize.assert_called_once_with("Screen captured successfully.")
        synthesized_text = self.mock_tts.synthesize.call_args[0][0]
        self.assertNotIn("SystemSkillResult", synthesized_text)
        self.assertNotIn("duration_ms=", synthesized_text)

    def test_audio_player_plays_synthesized_response(self) -> None:
        """24. Verify AudioPlayer plays synthesized audio output."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        self.engine.listen_once(play_audio=True)

        self.mock_player.play.assert_called_once_with(self.mock_tts_output, block=True)

    def test_complete_turn_lifecycle_events(self) -> None:
        """25. Verify complete sequence of lifecycle events for Voice + Vision turn."""
        published_events: list[str] = []
        for topic in (
            EVENT_ENGINE_RECORDING,
            EVENT_ENGINE_TRANSCRIBING,
            EVENT_ENGINE_ROUTING,
            EVENT_ENGINE_SPEAKING,
            EVENT_ENGINE_COMPLETED,
        ):
            self.event_bus.subscribe(topic, lambda e, t=topic: published_events.append(t))

        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")
        result = self.engine.listen_once()

        self.assertTrue(result.success)
        self.assertEqual(
            published_events,
            [
                EVENT_ENGINE_RECORDING,
                EVENT_ENGINE_TRANSCRIBING,
                EVENT_ENGINE_ROUTING,
                EVENT_ENGINE_SPEAKING,
                EVENT_ENGINE_COMPLETED,
            ],
        )

    # =======================================================================
    # Group E: Interruption & State Recovery
    # =======================================================================

    def test_interrupt_during_speech_stops_playback(self) -> None:
        """26. Verify engine.interrupt() calls player.interrupt()."""
        self.engine.interrupt()
        self.mock_player.interrupt.assert_called_once()

    def test_interrupt_recovers_state_to_idle(self) -> None:
        """27. Verify interrupt transitions state from SPEAKING / EXECUTING to IDLE."""
        self.state_manager.transition_to(AssistantState.EXECUTING, status_message="Executing")
        self.state_manager.transition_to(AssistantState.SPEAKING, status_message="Reading screen")
        self.assertEqual(self.state_manager.get_snapshot().state, AssistantState.SPEAKING)

        self.engine.interrupt()

        self.assertEqual(self.state_manager.get_snapshot().state, AssistantState.IDLE)
        self.assertEqual(self.state_manager.get_snapshot().status_message, "Interrupted")

    def test_interrupt_during_executing_recovers_to_idle(self) -> None:
        """28. Verify interrupt transitions state from EXECUTING to IDLE."""
        self.state_manager.transition_to(AssistantState.EXECUTING, status_message="Analyzing vision")
        self.assertEqual(self.state_manager.get_snapshot().state, AssistantState.EXECUTING)

        self.engine.interrupt()

        self.assertEqual(self.state_manager.get_snapshot().state, AssistantState.IDLE)

    def test_interrupt_resets_amplitude_to_zero(self) -> None:
        """29. Verify interrupt resets state manager mic amplitude to 0.0."""
        self.state_manager.update_mic_amplitude(0.85)
        self.assertAlmostEqual(self.state_manager.get_snapshot().mic_amplitude, 0.85)

        self.engine.interrupt()

        self.assertAlmostEqual(self.state_manager.get_snapshot().mic_amplitude, 0.0)

    # =======================================================================
    # Group F: Privacy & Memory Invariants
    # =======================================================================

    def test_zero_raw_screenshot_bytes_in_response_or_events(self) -> None:
        """30. Invariant: zero raw image bytes or base64 payloads in voice results or events."""
        events_captured: list[Event] = []
        self.event_bus.subscribe("*", events_captured.append)

        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")
        result = self.engine.listen_once()

        # Check conversation result
        self.assertNotIn(b"\x00\xFF", str(result.response_text).encode("utf-8", errors="ignore"))
        self.assertNotIn("data:image", result.response_text)

        # Check all published event payloads
        for evt in events_captured:
            payload_str = str(evt.payload)
            self.assertNotIn("data:image", payload_str)
            self.assertNotIn("raw_data", payload_str)

    def test_ephemeral_buffer_cleanup(self) -> None:
        """31. Verify ScreenObservation stored in ephemeral buffer is cleared on demand."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")
        self.engine.listen_once()

        self.assertFalse(self.buffer_mgr.is_empty)
        self.buffer_mgr.clear()
        self.assertTrue(self.buffer_mgr.is_empty)

    # =======================================================================
    # Group G: Failure Recovery
    # =======================================================================

    def test_capture_failure_recovery(self) -> None:
        """32. Verify screen capture engine failure handled safely without unhandled exception."""
        self.mock_engine.capture_screen.side_effect = RuntimeError("GDI BitBlt failed")
        self.mock_engine.capture_active_window.side_effect = RuntimeError("GDI BitBlt failed")
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        failed_events: list[Event] = []
        self.event_bus.subscribe(EVENT_ENGINE_FAILED, failed_events.append)

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["phase"], "routing")

    def test_ocr_failure_recovery(self) -> None:
        """33. Verify OCR provider crash handled gracefully."""
        self.ocr_provider.extract_text = MagicMock(side_effect=RuntimeError("OCR engine out of memory"))
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="read text on screen", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertIn("OCR engine out of memory", str(result.error))

    def test_text_only_ai_provider_explain_window_failure(self) -> None:
        """34. Verify text-only AI provider raises clear error for explain_active_window."""
        self.mock_gemini.supports_multimodal = False
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="explain this window", language="en")

        result = self.engine.listen_once()

        self.assertFalse(result.success)
        self.assertIn("does not support multimodal vision", str(result.error))

    def test_tts_synthesis_failure_non_fatal(self) -> None:
        """35. Verify TTS synthesis failure is non-fatal to conversation turn result."""
        self.mock_tts.synthesize.side_effect = RuntimeError("TTS audio stream corrupted")
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        result = self.engine.listen_once()

        # Conversation succeeded in routing and generating response, even though TTS synthesis failed
        self.assertTrue(result.success)
        self.assertEqual(result.response_text, "Screen captured successfully.")
        self.assertIsNone(result.speech_result)

    def test_audio_playback_failure_handled(self) -> None:
        """36. Verify audio playback failure logs warning and recovers safely."""
        self.mock_player.play.side_effect = RuntimeError("Audio device disconnected")
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="capture screen", language="en")

        result = self.engine.listen_once(play_audio=True)

        self.assertTrue(result.success)
        self.assertAlmostEqual(self.state_manager.get_snapshot().mic_amplitude, 0.0)


if __name__ == "__main__":
    unittest.main()
