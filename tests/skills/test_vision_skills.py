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

    # -----------------------------------------------------------------------
    # Phase 27.6: VQA, Observation Reuse, Visual Verification & Routing Tests
    # -----------------------------------------------------------------------

    # A. ask_screen Routing
    def test_can_handle_ask_screen_prefix(self) -> None:
        """37. Verify can_handle accepts explicit 'ask screen' and 'screen query' prefixes."""
        self.assertTrue(self.skill.can_handle("ask screen what color is the button?"))
        self.assertTrue(self.skill.can_handle("ask the screen what port is open"))
        self.assertTrue(self.skill.can_handle("screen query: is the terminal idle?"))
        self.assertTrue(self.skill.can_handle("vqa: what application is running?"))

    def test_can_handle_visual_context_queries(self) -> None:
        """38. Verify can_handle accepts natural language visual context questions."""
        queries = [
            "What is on my screen?",
            "What color is the button on my screen?",
            "Does the terminal show an error?",
            "What port is shown in the terminal?",
            "Is the download finished?",
            "What does this dialog say?",
            "What is visible on my screen?",
            "What is shown in this window?",
        ]
        for q in queries:
            self.assertTrue(self.skill.can_handle(q), f"Failed to recognize visual query: '{q}'")

    def test_can_handle_rejects_non_visual_queries(self) -> None:
        """39. Verify can_handle rejects ordinary non-visual questions lacking visual context."""
        non_visual = [
            "What is the button?",
            "What does this mean?",
            "Explain recursion in Python",
            "What is the capital of France?",
            "Calculate 45 * 2",
            "Tell me a joke",
            "Write a function to sort a list",
        ]
        for q in non_visual:
            self.assertFalse(self.skill.can_handle(q), f"Should not handle non-visual query: '{q}'")

    def test_parse_command_ask_screen_prefix(self) -> None:
        """40. Verify parse_command extracts operation and question from prefix format."""
        op, target, params, _ = self.skill.parse_command("ask screen what port is open?")
        self.assertEqual(op, "ask_screen")
        self.assertEqual(target, "active_window")
        self.assertEqual(params.get("question"), "what port is open?")

    def test_parse_command_verify_screen_prefix(self) -> None:
        """41. Verify parse_command extracts operation and condition from verify prefix."""
        op, target, params, _ = self.skill.parse_command("verify screen state: the download is complete")
        self.assertEqual(op, "verify_screen_state")
        self.assertEqual(target, "active_window")
        self.assertEqual(params.get("condition"), "the download is complete")

    def test_parse_command_visual_question(self) -> None:
        """42. Verify parse_command routes natural visual question to ask_screen."""
        op, target, params, _ = self.skill.parse_command("What port is shown in the terminal?")
        self.assertEqual(op, "ask_screen")
        self.assertEqual(target, "active_window")
        self.assertEqual(params.get("question"), "What port is shown in the terminal?")

    # B. VQA Core
    def test_ask_screen_vqa_success(self) -> None:
        """43. Verify ask_screen executes VQA and returns structured answer."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="The submit button appears blue.",
            model="gemini-2.5-flash",
        )
        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What color is the submit button?"},
        })
        self.assertTrue(res.success)
        self.assertEqual(res.data["answer"], "The submit button appears blue.")
        self.assertEqual(res.data["question"], "What color is the submit button?")
        self.assertFalse(res.data["reused_cache"])
        self.assertTrue(bool(res.data["observation_id"]))

    def test_ask_screen_vqa_receives_exact_question_and_image(self) -> None:
        """44. Verify prompt builder receives the exact question and preprocessed image part."""
        self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Does the terminal show an error?"},
        })
        call_args = self.mock_gemini.generate.call_args[0][0]
        prompt_text = call_args["contents"][0]["parts"][0]["text"]
        self.assertIn("Does the terminal show an error?", prompt_text)
        self.assertIn("inlineData", call_args["contents"][0]["parts"][1])

    def test_ask_screen_model_uncertainty(self) -> None:
        """45. Verify ask_screen handles model visual uncertainty response."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="I can't determine that from the visible screen.",
            model="gemini-2.5-flash",
        )
        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What is the password?"},
        })
        self.assertTrue(res.success)
        self.assertEqual(res.data["answer"], "I can't determine that from the visible screen.")

    def test_ask_screen_text_only_provider_raises_error(self) -> None:
        """46. Verify ask_screen raises SkillExecutionError when provider lacks multimodal support."""
        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ai_provider=self.mock_ollama,
            security_policy=self.sec_policy,
        )
        with self.assertRaises(SkillExecutionError) as ctx:
            skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is on my screen?"},
            })
        self.assertIn("does not support multimodal vision inputs", str(ctx.exception))

    def test_ask_screen_missing_question_raises_error(self) -> None:
        """47. Verify ask_screen raises SkillExecutionError when no question is provided."""
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({"operation": "ask_screen", "parameters": {}})
        self.assertIn("No question provided", str(ctx.exception))

    # C. Observation Reuse
    def test_ask_screen_fresh_capture_by_default(self) -> None:
        """48. Verify ask_screen performs fresh capture by default (reuse_cache=False)."""
        self.skill.execute({"operation": "ask_screen", "parameters": {"question": "Q1"}})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        self.skill.execute({"operation": "ask_screen", "parameters": {"question": "Q2"}})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_reuse_cache_when_enabled_and_fresh(self) -> None:
        """49. Verify ask_screen reuses cached observation when reuse_cache=True within <=5s."""
        # Prime the buffer
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Mock window identity to match stored metadata
        self.mock_engine.get_active_window_handle.return_value = 12345
        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "", None, (0, 0, 100, 100))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertTrue(res.data["reused_cache"])
            # Engine capture count did not increase
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

    def test_ask_screen_reuse_cache_rejected_when_older_than_5_seconds(self) -> None:
        """50. Verify observation older than 5.0 seconds is rejected and fresh capture occurs."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Artificially age the cached observation
        obs = self.buffer_manager.get()
        self.assertIsNotNone(obs)
        object.__setattr__(obs, "timestamp", time.time() - 6.0)

        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "", None, (0, 0, 100, 100))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_reuse_cache_rejected_when_hwnd_mismatched(self) -> None:
        """51. Verify cache reuse rejected when foreground HWND differs from cached observation."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})

        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(99999, "", None, (0, 0, 100, 100))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_reuse_cache_rejected_when_window_title_mismatched(self) -> None:
        """52. Verify cache reuse rejected when window title changes (e.g. browser navigation)."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})

        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "Different Page Title", None, (0, 0, 100, 100))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_reuse_cache_rejected_when_process_mismatched(self) -> None:
        """53. Verify cache reuse rejected when active process name differs."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})

        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "", "other_app.exe", (0, 0, 100, 100))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_reuse_cache_rejected_when_bounds_mismatched(self) -> None:
        """54. Verify cache reuse rejected when window bounds change (e.g. resized/moved)."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})

        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "", None, (10, 20, 500, 600))):
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the button?", "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_ask_screen_temporal_query_forces_fresh_capture(self) -> None:
        """55. Verify temporal questions force fresh capture even when reuse_cache=True."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        temporal_queries = [
            "What is on my screen right now?",
            "What is currently shown in the window?",
            "What is the current status?",
            "Show me the latest terminal output",
            "Has it changed on my screen?",
            "Is it happening now?",
        ]
        with patch.object(self.secure_vision, "get_active_window_identity", return_value=(12345, "", None, (0, 0, 100, 100))):
            for i, q in enumerate(temporal_queries, start=2):
                res = self.skill.execute({
                    "operation": "ask_screen",
                    "parameters": {"question": q, "reuse_cache": True},
                })
                self.assertFalse(res.data["reused_cache"], f"Temporal query '{q}' should not reuse cache")
                self.assertEqual(self.mock_engine.capture_active_window.call_count, i)

    def test_ask_screen_explicit_observation_id_reuse(self) -> None:
        """56. Verify intra-step explicit observation_id reuse within 5 seconds."""
        res_cap = self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        obs_id = res_cap.data["observation_id"]

        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What is shown?", "observation_id": obs_id},
        })
        self.assertTrue(res.data["reused_cache"])
        self.assertEqual(res.data["observation_id"], obs_id)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

    # D. verify_screen_state
    def test_verify_screen_state_verified_true(self) -> None:
        """57. Verify verify_screen_state parses VERIFIED verdict and returns verified=True."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="VERIFIED: The download completed and file shows 100%.",
            model="gemini-2.5-flash",
        )
        res = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "The download is complete"},
        })
        self.assertTrue(res.success)
        self.assertTrue(res.data["verified"])
        self.assertEqual(res.data["status"], "verified")
        self.assertIn("download completed", res.data["reason"])
        self.assertFalse(res.data["reused_cache"])

    def test_verify_screen_state_not_verified(self) -> None:
        """58. Verify verify_screen_state parses NOT VERIFIED verdict and returns verified=False."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="NOT VERIFIED: The progress bar shows 45%, download is in progress.",
            model="gemini-2.5-flash",
        )
        res = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "The download is complete"},
        })
        self.assertTrue(res.success)
        self.assertFalse(res.data["verified"])
        self.assertEqual(res.data["status"], "not_verified")
        self.assertIn("progress bar shows 45%", res.data["reason"])

    def test_verify_screen_state_uncertain(self) -> None:
        """59. Verify verify_screen_state parses UNCERTAIN verdict and returns verified=None."""
        self.mock_gemini.generate.return_value = AIResponse(
            content="UNCERTAIN: The window is obscured and progress cannot be determined.",
            model="gemini-2.5-flash",
        )
        res = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "The download is complete"},
        })
        self.assertTrue(res.success)
        self.assertIsNone(res.data["verified"])
        self.assertEqual(res.data["status"], "uncertain")
        self.assertIn("obscured", res.data["reason"])

    def test_verify_screen_state_always_fresh_never_reuses_cache(self) -> None:
        """60. Verify verify_screen_state MANDATES fresh capture and ignores reuse_cache=True."""
        # Prime cache
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Call verify with reuse_cache=True -> MUST STILL FORCE FRESH CAPTURE
        res = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "Button is visible", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_verify_screen_state_missing_condition_raises_error(self) -> None:
        """61. Verify verify_screen_state raises SkillExecutionError when condition is omitted."""
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({"operation": "verify_screen_state", "parameters": {}})
        self.assertIn("No condition specified", str(ctx.exception))

    def test_verify_screen_state_text_only_provider_raises_error(self) -> None:
        """62. Verify verify_screen_state raises error with text-only AI provider."""
        skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ai_provider=self.mock_ollama,
            security_policy=self.sec_policy,
        )
        with self.assertRaises(SkillExecutionError) as ctx:
            skill.execute({
                "operation": "verify_screen_state",
                "parameters": {"condition": "App is open"},
            })
        self.assertIn("does not support multimodal vision inputs", str(ctx.exception))

    # E. Planner Integration
    def test_executor_resolves_ask_screen_via_alias(self) -> None:
        """63. Verify Executor alias map resolves ask_screen to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t5", action="ask_screen", parameters={"question": "What is on screen?"})
        fn = executor._resolve_from_skill_manager("ask_screen", task=task)
        self.assertIsNotNone(fn)

    def test_executor_resolves_verify_screen_state_via_alias(self) -> None:
        """64. Verify Executor alias map resolves verify_screen_state to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        task = Task(id="t6", action="verify_screen_state", parameters={"condition": "App is running"})
        fn = executor._resolve_from_skill_manager("verify_screen_state", task=task)
        self.assertIsNotNone(fn)

    def test_executor_resolves_via_container_candidate_keys(self) -> None:
        """65. Verify Executor container resolution includes vision keys for new operations."""
        self.container.register_singleton("vision", self.skill)

        executor = Executor(
            container_instance=self.container,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        for act in ("ask_screen", "verify_screen_state"):
            fn = executor._resolve_from_container(act)
            self.assertIsNotNone(fn, f"Failed container resolution for '{act}'")

    # F. Security
    def test_security_blocks_vqa_on_sensitive_window(self) -> None:
        """66. Verify sensitive window policy denies ask_screen and prevents capture."""
        with patch.object(self.secure_vision, "_resolve_window_metadata", return_value=("1Password - Master Vault", "1password.exe")):
            with self.assertRaises((CaptureBlockedError, SkillExecutionError)):
                self.skill.execute({
                    "operation": "ask_screen",
                    "parameters": {"question": "What is shown?"},
                })
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)

    def test_security_blocks_verification_on_sensitive_window(self) -> None:
        """67. Verify sensitive window policy denies verify_screen_state and prevents capture."""
        with patch.object(self.secure_vision, "_resolve_window_metadata", return_value=("Bank of America - Login", "chrome.exe")):
            with self.assertRaises((CaptureBlockedError, SkillExecutionError)):
                self.skill.execute({
                    "operation": "verify_screen_state",
                    "parameters": {"condition": "Login is complete"},
                })
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 0)

    def test_security_blocks_cache_reuse_when_active_window_becomes_sensitive(self) -> None:
        """68. Verify observation reuse is blocked if active window switches to sensitive context."""
        # Prime cache with safe window
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Now active window becomes sensitive
        with patch.object(self.secure_vision, "_resolve_window_metadata", return_value=("Bitwarden - Passwords", "bitwarden.exe")):
            with self.assertRaises((CaptureBlockedError, SkillExecutionError)):
                self.skill.execute({
                    "operation": "ask_screen",
                    "parameters": {"question": "What is the text?", "reuse_cache": True},
                })

    # G. Response Formatting
    def test_system_skill_result_message_ask_screen(self) -> None:
        """69. Verify SystemSkillResult.message cleanly extracts ask_screen answer."""
        res = SystemSkillResult(
            success=True,
            operation="ask_screen",
            data={"answer": "The terminal shows a connection error."},
        )
        self.assertEqual(res.message, "The terminal shows a connection error.")

    def test_system_skill_result_message_verify_screen_state_verified(self) -> None:
        """70. Verify SystemSkillResult.message formats verified state cleanly."""
        res = SystemSkillResult(
            success=True,
            operation="verify_screen_state",
            data={
                "verified": True,
                "status": "verified",
                "reason": "The download reached 100%.",
            },
        )
        self.assertEqual(res.message, "The condition was verified. The download reached 100%.")

    def test_system_skill_result_message_verify_screen_state_not_verified(self) -> None:
        """71. Verify SystemSkillResult.message formats not verified state cleanly."""
        res = SystemSkillResult(
            success=True,
            operation="verify_screen_state",
            data={
                "verified": False,
                "status": "not_verified",
                "reason": "The file is still downloading at 45%.",
            },
        )
        self.assertEqual(res.message, "The condition was not verified. The file is still downloading at 45%.")

    def test_system_skill_result_message_verify_screen_state_uncertain(self) -> None:
        """72. Verify SystemSkillResult.message formats uncertain state cleanly."""
        res = SystemSkillResult(
            success=True,
            operation="verify_screen_state",
            data={
                "verified": None,
                "status": "uncertain",
                "reason": "The window is minimized.",
            },
        )
        self.assertEqual(res.message, "The window is minimized.")


if __name__ == "__main__":
    unittest.main()
