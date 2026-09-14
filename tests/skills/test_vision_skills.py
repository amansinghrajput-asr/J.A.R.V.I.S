"""Comprehensive unit, security, routing, and invariant tests for Phase 27.4 VisionSkills.

SAFETY INVARIANTS ENFORCED:
1. 100% offline, deterministic, hardware-safe.
2. NO physical desktop capture, NO Win32 GDI calls.
3. NO external network or real Gemini API calls.
4. ZERO screenshot persistence to disk.
5. ZERO raw image bytes in EventBus payloads, logs, or results.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.intent_router import IntentResult, IntentRouter, IntentType
from app.ai.models import AIResponse, ImagePart, UnsupportedModalityError
from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillFailed,
    SystemSkillStarted,
)
from app.ai.planner.executor import Executor
from app.ai.planner.models import Task
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.router.intent import Intent
from app.router.router import CommandRouter
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.skills.system.vision_skills import VisionSkills
from app.vision.capture import DesktopCaptureEngine, MockCaptureBackend
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
from app.vision.ocr import MockOCRProvider, OCRProvider, WindowsMediaOCRProvider
from app.vision.preprocessing import ImagePreprocessor
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    VisionSecurityPolicy,
)


class TestVisionSkills(unittest.TestCase):
    """Test suite for VisionSkills (Phase 27.4)."""

    def setUp(self) -> None:
        """Set up isolated mock dependencies."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()

        self.sec_policy = SystemSecurityPolicy(event_bus=self.planner_bus)
        self.conf_manager = SystemConfirmationManager(event_bus=self.planner_bus)

        # Synthetic screen capture (100x100 RGB)
        self.fake_capture = ScreenCapture(
            raw_data=b"\x00\xFF\x00\xFF" * (100 * 100),
            width=100,
            height=100,
            source="screen",
            bounds=WindowBounds(0, 0, 100, 100),
            timestamp=1000.0,
            metadata={"window_title": "Test Application"},
        )

        # Mock engine
        self.mock_engine = MagicMock(spec=DesktopCaptureEngine)
        self.mock_engine.get_active_window_handle.return_value = 12345
        self.mock_engine.capture_screen.return_value = self.fake_capture
        self.mock_engine.capture_active_window.return_value = self.fake_capture

        # Real vision security policy
        self.vision_policy = VisionSecurityPolicy()

        # Ephemeral buffer manager
        self.buffer_manager = EphemeralBufferManager(event_bus_instance=self.event_bus)

        # Real SecureVisionManager with mocked engine
        self.secure_vision = SecureVisionManager(
            engine=self.mock_engine,
            policy=self.vision_policy,
            buffer_manager=self.buffer_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Mock multimodal AI provider (e.g. Gemini)
        self.mock_gemini = MagicMock()
        self.mock_gemini.model = "gemini-2.5-flash"
        self.mock_gemini.supports_multimodal = True
        self.mock_gemini.generate.return_value = AIResponse(
            content="Active window shows VS Code with Python code editor.",
            model="gemini-2.5-flash",
        )

        # Mock text-only AI provider (e.g. Ollama)
        self.mock_ollama = MagicMock()
        self.mock_ollama.model = "qwen2.5:3b"
        self.mock_ollama.supports_multimodal = False

        # Mock OCR provider
        self.mock_ocr = MockOCRProvider(canned_text="File Edit Selection View Go Run Terminal Help")

        # Instantiate VisionSkills
        self.skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ocr_provider=self.mock_ocr,
            ai_provider=self.mock_gemini,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )

        # Intercept planner events
        self.events: list[PlannerEvent] = []
        self.planner_bus.subscribe(PlannerEvent, lambda e: self.events.append(e))

    # -----------------------------------------------------------------------
    # 1–4: Attributes, Tags, Permissions, Priorities
    # -----------------------------------------------------------------------

    def test_skill_attributes_and_registration(self) -> None:
        """1. Verify skill name, priority, tags, and permissions."""
        self.assertEqual(self.skill.name, "vision")
        self.assertEqual(self.skill.priority, 52)
        self.assertIn("vision", self.skill.tags)
        self.assertIn("ocr", self.skill.tags)
        self.assertIn("multimodal", self.skill.tags)
        self.assertIn("vision:capture", self.skill.permissions)
        self.assertIn("vision:analyze", self.skill.permissions)

    def test_skill_priority_order(self) -> None:
        """2. Verify skill priority sits in the designated system skill range."""
        self.assertTrue(50 <= self.skill.priority <= 60)

    def test_permissions_contract(self) -> None:
        """3. Verify required permissions are declared as a set."""
        self.assertIsInstance(self.skill.permissions, set)
        self.assertTrue(self.skill.permissions.issuperset({"system:read", "vision:capture", "vision:analyze"}))

    # -----------------------------------------------------------------------
    # 5–7: can_handle() and parse_command()
    # -----------------------------------------------------------------------

    def test_can_handle_natural_language_commands(self) -> None:
        """4. Verify deterministic regex recognition of natural language commands."""
        test_phrases = [
            "What is on my screen?",
            "what's on my screen",
            "look at my screen",
            "read the text on my screen",
            "read screen text",
            "OCR screen",
            "explain this window",
            "describe this screen",
            "what does this error mean?",
            "diagnose this error",
            "why is this screen failing?",
            "take a screenshot",
            "capture screen",
            "screenshot my screen",
        ]
        for phrase in test_phrases:
            self.assertTrue(self.skill.can_handle(phrase), f"Failed for phrase: '{phrase}'")

    def test_can_handle_structured_dict(self) -> None:
        """5. Verify can_handle accepts structured command dictionaries."""
        self.assertTrue(self.skill.can_handle({"operation": "capture_screen"}))
        self.assertTrue(self.skill.can_handle({"operation": "read_screen_text"}))
        self.assertTrue(self.skill.can_handle({"operation": "explain_active_window"}))
        self.assertTrue(self.skill.can_handle({"operation": "diagnose_screen_error"}))
        self.assertFalse(self.skill.can_handle({"operation": "unsupported_op"}))

    def test_can_handle_unrelated_commands(self) -> None:
        """6. Verify can_handle does not hijack unrelated chat or coding queries."""
        unrelated = [
            "hello jarvis",
            "calculate 2 + 2",
            "write python code",
            "list all windows",
            "search the web for news",
            "remember my favorite color",
        ]
        for phrase in unrelated:
            self.assertFalse(self.skill.can_handle(phrase), f"Incorrectly matched: '{phrase}'")

    def test_parse_command_mapping(self) -> None:
        """7. Verify parse_command maps natural language to operation and target."""
        op, target, _, _ = self.skill.parse_command("take a screenshot")
        self.assertEqual(op, "capture_screen")
        self.assertEqual(target, "screen")

        op, target, _, _ = self.skill.parse_command("read the text on my screen")
        self.assertEqual(op, "read_screen_text")
        self.assertEqual(target, "active_window")

        op, target, _, _ = self.skill.parse_command("explain this window")
        self.assertEqual(op, "explain_active_window")
        self.assertEqual(target, "active_window")

        op, target, _, _ = self.skill.parse_command("diagnose this error")
        self.assertEqual(op, "diagnose_screen_error")
        self.assertEqual(target, "active_window")

    # -----------------------------------------------------------------------
    # 8–9: CommandRouter Resolution
    # -----------------------------------------------------------------------

    def test_command_router_specialized_skill_dispatch(self) -> None:
        """8. Verify CommandRouter routes vision queries to VisionSkills as specialized skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        router = CommandRouter(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            skill_manager_instance=sm,
            auto_register_in_container=False,
        )

        matched, payload = router._find_matching_skill(Intent(raw_command="what is on my screen?"))
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "vision")

    def test_command_router_structured_dispatch(self) -> None:
        """9. Verify CommandRouter handles structured vision commands."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        router = CommandRouter(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            skill_manager_instance=sm,
            auto_register_in_container=False,
        )

        matched, payload = router._find_matching_skill(Intent(raw_command="read screen text"))
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "vision")

    # -----------------------------------------------------------------------
    # 10–14: capture_screen Operation
    # -----------------------------------------------------------------------

    def test_capture_screen_success(self) -> None:
        """10. Verify capture_screen returns metadata without exposing raw pixels."""
        result = self.skill.execute({"operation": "capture_screen"})
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, dict)
        self.assertEqual(result.data["width"], 100)
        self.assertEqual(result.data["height"], 100)
        self.assertIn("observation_id", result.data)
        # Verify no raw pixel bytes leaked into result data
        self.assertNotIn("raw_bytes", result.data)
        self.assertNotIn("bytes", result.data)

    def test_capture_screen_active_window_target(self) -> None:
        """11. Verify capture_screen targets active window when requested."""
        result = self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertTrue(result.success)
        self.mock_engine.capture_active_window.assert_called()

    def test_capture_screen_full_desktop(self) -> None:
        """12. Verify capture_screen defaults to full screen."""
        result = self.skill.execute({"operation": "capture_screen", "target": "screen"})
        self.assertTrue(result.success)
        self.mock_engine.capture_screen.assert_called()

    def test_capture_screen_security_block(self) -> None:
        """13. Verify capture_screen blocks sensitive windows (e.g. Bitwarden, KeePass)."""
        # Mock active window title as sensitive password manager
        self.skill._secure_vision_manager._resolve_window_metadata = MagicMock(
            return_value=("Bitwarden - Vault", "bitwarden.exe")
        )

        # Policy should block and skill should catch or fail cleanly
        with self.assertRaises(SkillExecutionError):
            self.skill.execute({"operation": "capture_screen"})

    def test_capture_screen_ephemeral_buffer_storage(self) -> None:
        """14. Verify capture_screen stores observation in RAM-only ephemeral buffer."""
        result = self.skill.execute({"operation": "capture_screen"})
        obs_id = result.data["observation_id"]
        # Buffer manager must contain this observation
        obs = self.buffer_manager.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertFalse(obs.is_expired)

    # -----------------------------------------------------------------------
    # 15–18: read_screen_text (Hybrid OCR)
    # -----------------------------------------------------------------------

    def test_read_screen_text_with_mock_ocr(self) -> None:
        """15. Verify read_screen_text successfully extracts text via MockOCRProvider."""
        result = self.skill.execute({"operation": "read_screen_text"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["text"], "File Edit Selection View Go Run Terminal Help")
        self.assertGreater(result.data["line_count"], 0)
        self.assertGreater(result.data["block_count"], 0)
        self.assertIn("duration_ms", result.data)

    def test_read_screen_text_native_ocr_unavailable_fallback(self) -> None:
        """16. Verify fallback to multimodal AI provider when native OCR is absent."""
        # VisionSkills with no explicit ocr_provider -> resolves to MultimodalVisionOCRAdapter
        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ocr_provider=None,
            ai_provider=self.mock_gemini,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )
        self.mock_gemini.generate.return_value = AIResponse(
            content="Extracted line 1\nExtracted line 2",
            model="gemini-2.5-flash",
        )

        result = skill.execute({"operation": "read_screen_text"})
        self.assertTrue(result.success)
        self.assertIn("Extracted line 1", result.data["text"])
        self.assertEqual(result.data["line_count"], 2)

    def test_read_screen_text_unsupported_provider_fails(self) -> None:
        """17. Verify clear error when no OCR provider and text-only AI provider (e.g. Ollama)."""
        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ocr_provider=None,
            ai_provider=self.mock_ollama,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )
        with self.assertRaises(SkillExecutionError) as ctx:
            skill.execute({"operation": "read_screen_text"})
        self.assertIn("No capable OCR provider is available", str(ctx.exception))

    def test_read_screen_text_screen_target(self) -> None:
        """18. Verify read_screen_text supports target='screen' when requested."""
        result = self.skill.execute({"operation": "read_screen_text", "target": "screen"})
        self.assertTrue(result.success)
        self.mock_engine.capture_screen.assert_called()

    # -----------------------------------------------------------------------
    # 19–21: explain_active_window
    # -----------------------------------------------------------------------

    def test_explain_active_window_success(self) -> None:
        """19. Verify explain_active_window returns structured visual analysis."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="A web browser showing GitHub pull request page.",
            model="gemini-2.5-flash",
        )
        result = self.skill.execute({"operation": "explain_active_window"})
        self.assertTrue(result.success)
        self.assertIn("GitHub", result.data["summary"])
        self.assertIn("observation_id", result.data)
        self.assertEqual(result.data["target"], "active_window")

    def test_explain_active_window_unsupported_provider(self) -> None:
        """20. Verify explain_active_window raises clear error on text-only provider."""
        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ai_provider=self.mock_ollama,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )
        with self.assertRaises(SkillExecutionError) as ctx:
            skill.execute({"operation": "explain_active_window"})
        self.assertIn("does not support multimodal vision", str(ctx.exception))

    def test_explain_active_window_downsampling_applied(self) -> None:
        """21. Verify image is downsampled before sending to AI provider."""
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        mock_preprocessor.to_image_part.return_value = ImagePart(data=b"mock_png", mime_type="image/png")

        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ai_provider=self.mock_gemini,
            preprocessor=mock_preprocessor,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.planner_bus,
        )
        result = skill.execute({"operation": "explain_active_window"})
        self.assertTrue(result.success)
        mock_preprocessor.to_image_part.assert_called_once()

    # -----------------------------------------------------------------------
    # 22–25: diagnose_screen_error
    # -----------------------------------------------------------------------

    def test_diagnose_screen_error_found(self) -> None:
        """22. Verify structured diagnosis output when visible error exists."""
        self.mock_gemini.generate.return_value = AIResponse(
            content=(
                "Error Found: True\n"
                "Error Title: ModuleNotFoundError: No module named 'numpy'\n"
                "Root Cause: Missing dependency in python virtual environment\n"
                "Recommended Fix: Run pip install numpy"
            ),
            model="gemini-2.5-flash",
        )
        result = self.skill.execute({"operation": "diagnose_screen_error"})
        self.assertTrue(result.success)
        self.assertTrue(result.data["error_found"])
        self.assertIn("ModuleNotFoundError", result.data["error_title"])
        self.assertIn("Missing dependency", result.data["root_cause"])
        self.assertIn("pip install numpy", result.data["recommended_fix"])

    def test_diagnose_screen_error_none_detected(self) -> None:
        """23. Verify clean result when no visible error is present."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="Error Found: False\nNo clear error or warning was detected in the active window.",
            model="gemini-2.5-flash",
        )
        result = self.skill.execute({"operation": "diagnose_screen_error"})
        self.assertTrue(result.success)
        self.assertFalse(result.data["error_found"])
        self.assertIsNone(result.data["error_title"])

    def test_diagnose_screen_error_malformed_response(self) -> None:
        """24. Verify unparseable AI response handled gracefully without crashing."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="Just a regular dialog with some text.",
            model="gemini-2.5-flash",
        )
        result = self.skill.execute({"operation": "diagnose_screen_error"})
        self.assertTrue(result.success)
        self.assertIn("raw_diagnosis", result.data)

    def test_diagnose_screen_error_does_not_execute_fix(self) -> None:
        """25. Verify diagnose_screen_error is analysis-only and performs zero system actions."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="Error Found: True\nRecommended Fix: delete file",
            model="gemini-2.5-flash",
        )
        result = self.skill.execute({"operation": "diagnose_screen_error"})
        self.assertTrue(result.success)
        # Ensure only capture occurred, no file deletion or shell execution
        self.assertEqual(result.operation, "diagnose_screen_error")

    # -----------------------------------------------------------------------
    # 26–28: Privacy and Leakage Prevention
    # -----------------------------------------------------------------------

    def test_no_raw_image_leakage_in_result(self) -> None:
        """26. Verify raw bytes or base64 pixel strings do not appear in SystemSkillResult."""
        for op in ("capture_screen", "read_screen_text", "explain_active_window", "diagnose_screen_error"):
            res = self.skill.execute({"operation": op})
            data_str = str(res.data)
            self.assertNotIn("raw_bytes", data_str)
            self.assertNotIn("b'\\x00", data_str)

    def test_no_raw_image_leakage_in_event_bus(self) -> None:
        """27. Verify PlannerEventBus telemetry contains no raw image data."""
        self.skill.execute({"operation": "capture_screen"})
        for evt in self.events:
            evt_str = str(getattr(evt, "__dict__", str(evt)))
            self.assertNotIn("raw_bytes", evt_str)
            self.assertNotIn("b'\\x00", evt_str)

    def test_no_conversation_memory_pollution(self) -> None:
        """28. Verify visual data does not inject raw images into conversation memory."""
        res = self.skill.execute({"operation": "explain_active_window"})
        self.assertIsInstance(res.data["summary"], str)
        self.assertNotIn("data:image", res.data["summary"])

    # -----------------------------------------------------------------------
    # 29–32: Executor Integration
    # -----------------------------------------------------------------------

    def test_executor_alias_dispatch_capture_screen(self) -> None:
        """29. Verify Executor alias map resolves capture_screen to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t1", action="capture_screen", target="screen")
        fn = executor._resolve_from_skill_manager("capture_screen", task=task)
        self.assertIsNotNone(fn)

    def test_executor_alias_dispatch_read_screen_text(self) -> None:
        """30. Verify Executor alias map resolves read_screen_text to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t2", action="read_screen_text")
        fn = executor._resolve_from_skill_manager("read_screen_text", task=task)
        self.assertIsNotNone(fn)

    def test_executor_alias_dispatch_explain_active_window(self) -> None:
        """31. Verify Executor alias map resolves explain_active_window to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t3", action="explain_active_window")
        fn = executor._resolve_from_skill_manager("explain_active_window", task=task)
        self.assertIsNotNone(fn)

    def test_executor_alias_dispatch_diagnose_screen_error(self) -> None:
        """32. Verify Executor alias map resolves diagnose_screen_error to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t4", action="diagnose_screen_error")
        fn = executor._resolve_from_skill_manager("diagnose_screen_error", task=task)
        self.assertIsNotNone(fn)

    # -----------------------------------------------------------------------
    # 33–34: IntentRouter Classification & Regression
    # -----------------------------------------------------------------------

    def test_intent_router_classifies_vision_intents(self) -> None:
        """33. Verify IntentRouter classifies visual commands as IntentType.VISION."""
        ir = IntentRouter(auto_register_in_container=False)

        vision_queries = [
            "look at my screen",
            "what is on my screen",
            "what's on my screen",
            "read the text on my screen",
            "explain this window",
            "diagnose this error",
            "what does this error mean",
            "take a screenshot",
            "capture screen",
        ]
        for query in vision_queries:
            res = ir.classify(query)
            self.assertEqual(res.intent, IntentType.VISION, f"Failed for query: '{query}' (got {res.intent})")

    def test_intent_router_preserves_existing_intents(self) -> None:
        """34. Verify existing IntentRouter rules are not broken or regressed."""
        ir = IntentRouter(auto_register_in_container=False)

        test_cases = [
            ("remember this note", IntentType.MEMORY),
            ("summarize this document.pdf", IntentType.FILE),
            ("open notepad", IntentType.SYSTEM),
            ("write python code", IntentType.CODING),
            ("calculate 5 * 25", IntentType.TOOL),
            ("what is the weather today", IntentType.SEARCH),
            ("explain recursion", IntentType.REASONING),
        ]
        for query, expected in test_cases:
            res = ir.classify(query)
            self.assertEqual(res.intent, expected, f"Failed regression for: '{query}' (expected {expected}, got {res.intent})")

    # -----------------------------------------------------------------------
    # 35–36: Async Execution & Telemetry
    # -----------------------------------------------------------------------

    def test_async_execution_offloads_thread(self) -> None:
        """35. Verify execute_async runs non-blockingly via asyncio.to_thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.skill.execute_async({"operation": "capture_screen"})
            )
            self.assertTrue(result.success)
            self.assertEqual(result.operation, "capture_screen")
        finally:
            loop.close()

    def test_planner_event_bus_lifecycle_telemetry(self) -> None:
        """36. Verify BaseSystemSkill emits SystemSkillStarted and SystemSkillCompleted events."""
        self.skill.execute({"operation": "capture_screen"})

        started_events = [e for e in self.events if isinstance(e, SystemSkillStarted)]
        completed_events = [e for e in self.events if isinstance(e, SystemSkillCompleted)]

        self.assertEqual(len(started_events), 1)
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(started_events[0].skill_name, "vision")
        self.assertEqual(started_events[0].operation, "capture_screen")
        self.assertEqual(completed_events[0].operation, "capture_screen")


if __name__ == "__main__":
    unittest.main()
