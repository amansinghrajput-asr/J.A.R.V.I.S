"""Comprehensive unit, security, routing, privacy, and closed-loop tests for Phase 27.7.

Tests:
A. Follow-up routing (with context vs. without context vs. non-visual)
B. Observation reuse & TTL
C. Compound identity mismatch (HWND, title, process, bounds)
D. Privacy & security re-authorization on foreground change
E. Temporal queries forcing fresh observation
F. Verification questions forcing fresh observation
G. Planner closed loop (OBSERVE -> ACT -> VERIFY)
H. 3-way verification semantics (VERIFIED, NOT VERIFIED, UNCERTAIN)
I. Memory privacy (zero raw pixels/buffers in memory)
J. Voice compatibility and speakable SystemSkillResult.message
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.models import AIResponse, ImagePart
from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillFailed,
    SystemSkillStarted,
)
from app.ai.planner.executor import Executor
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord
from app.ai.planner.models import Task, TaskStatus
from app.ai.planner.planner import KNOWN_ACTIONS, Planner
from app.core.container import ServiceContainer
from app.core.event_bus import Event, EventBus
from app.memory.models import ConversationMemory
from app.router.intent import Intent
from app.router.router import CommandRouter
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    SystemConfirmationManager,
    SystemSecurityPolicy,
)
from app.skills.system.vision_skills import VisionSkills
from app.vision.capture import DesktopCaptureEngine
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    ScreenCapture,
    ScreenObservation,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    VisionSecurityPolicy,
)


class TestVisionContextPhase277(unittest.TestCase):
    """Phase 27.7 Visual Context & Observation Intelligence test suite."""

    def setUp(self) -> None:
        """Set up isolated mock dependencies."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.planner_bus = PlannerEventBus()

        self.sec_policy = SystemSecurityPolicy(event_bus=self.planner_bus)
        self.conf_manager = SystemConfirmationManager(event_bus=self.planner_bus)

        # Synthetic screen capture (100x100)
        self.fake_capture = ScreenCapture(
            raw_data=b"\x00\xFF\x00\xFF" * (100 * 100),
            width=100,
            height=100,
            source="screen",
            bounds=WindowBounds(0, 0, 100, 100),
            timestamp=time.time(),
            metadata={"window_title": "Editor Window"},
        )

        # Mock capture engine
        self.mock_engine = MagicMock(spec=DesktopCaptureEngine)
        self.mock_engine.get_active_window_handle.return_value = 1111
        self.mock_engine.capture_screen.return_value = self.fake_capture
        self.mock_engine.capture_active_window.return_value = self.fake_capture
        self.mock_engine.get_window_bounds.return_value = WindowBounds(0, 0, 100, 100)

        # Real vision security policy
        self.vision_policy = VisionSecurityPolicy()

        # Ephemeral buffer manager
        self.buffer_manager = EphemeralBufferManager(event_bus_instance=self.event_bus)

        # SecureVisionManager
        self.secure_vision = SecureVisionManager(
            engine=self.mock_engine,
            policy=self.vision_policy,
            buffer_manager=self.buffer_manager,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Default window metadata for safe editor
        self.default_hwnd = 1111
        self.default_title = "Editor Window"
        self.default_proc = "code.exe"
        self.default_bounds = (0, 0, 100, 100)
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=(self.default_title, self.default_proc)
        )
        self.secure_vision.get_active_window_identity = MagicMock(
            return_value=(self.default_hwnd, self.default_title, self.default_proc, self.default_bounds)
        )

        # Mock multimodal AI provider
        self.mock_ai = MagicMock()
        self.mock_ai.model = "gemini-2.5-flash"
        self.mock_ai.supports_multimodal = True
        self.mock_ai.generate.return_value = AIResponse(
            content="Active screen shows a login dialog with submit button.",
            model="gemini-2.5-flash",
        )

        # Mock OCR
        self.mock_ocr = MockOCRProvider(canned_text="Sign In Username Password Submit")

        # VisionSkills instance
        self.skill = VisionSkills(
            secure_vision_manager=self.secure_vision,
            ocr_provider=self.mock_ocr,
            ai_provider=self.mock_ai,
            security_policy=self.sec_policy,
            confirmation_manager=self.conf_manager,
            container=self.container,
            event_bus=self.event_bus,
        )
        self.skill.bind_system_services(planner_event_bus=self.planner_bus)

    # -----------------------------------------------------------------------
    # A. Follow-up Routing
    # -----------------------------------------------------------------------

    def test_followup_without_active_observation_rejected(self) -> None:
        """A1. Follow-up questions without an active observation in cache are rejected by can_handle."""
        follow_ups = [
            "What about the button on the right?",
            "What does that say?",
            "Where is that button?",
            "Can you see the error?",
            "Is that the login button?",
            "And the text below it?",
        ]
        for q in follow_ups:
            self.assertFalse(
                self.skill.can_handle(q),
                f"Should NOT handle follow-up '{q}' when no observation exists in cache",
            )

    def test_followup_with_valid_observation_accepted(self) -> None:
        """A2. Follow-up questions with a recent valid observation are handled and parsed as ask_screen with reuse."""
        # Prime the cache with an observation
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertIsNotNone(self.secure_vision.get_latest_observation())

        follow_ups = [
            "What about the button on the right?",
            "What does that say?",
            "Where is that button?",
            "Can you see the error?",
            "Is that the login button?",
            "And the text below it?",
        ]
        for q in follow_ups:
            self.assertTrue(
                self.skill.can_handle(q),
                f"Should handle follow-up '{q}' when valid observation exists in cache",
            )
            op, target, params, _ = self.skill.parse_command(q)
            self.assertEqual(op, "ask_screen")
            self.assertTrue(params.get("reuse_cache"))
            self.assertTrue(params.get("is_followup"))

    def test_unrelated_normal_queries_not_routed_to_vision_even_with_cache(self) -> None:
        """A3. Unrelated normal conversational queries do not route to VisionSkills even when cache is active."""
        # Prime cache
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})

        unrelated = [
            "How about tomorrow?",
            "Does it work?",
            "What about the weather tomorrow?",
            "Calculate 15 + 27",
            "Tell me a joke",
            "What is Python?",
        ]
        for q in unrelated:
            self.assertFalse(
                self.skill.can_handle(q),
                f"Should NOT route unrelated query '{q}' to VisionSkills",
            )

    def test_intent_object_handled_in_can_handle_and_parse(self) -> None:
        """A4. CommandRouter Intent objects are safely parsed by can_handle and parse_command."""
        intent = Intent(raw_command="what is on my screen?")
        self.assertTrue(self.skill.can_handle(intent))
        op, target, _, _ = self.skill.parse_command(intent)
        self.assertEqual(op, "explain_active_window")

    # -----------------------------------------------------------------------
    # B. Observation Reuse & TTL
    # -----------------------------------------------------------------------

    def test_valid_observation_reused_without_recapturing(self) -> None:
        """B1. Valid recent observation is reused without invoking capture engine bitblt again."""
        # Initial capture
        res1 = self.skill.execute({"operation": "ask_screen", "parameters": {"question": "What is on screen?"}})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)
        self.assertFalse(res1.data["reused_cache"])

        # Follow-up with reuse_cache=True
        res2 = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What about the button on the right?", "reuse_cache": True},
        })
        self.assertTrue(res2.data["reused_cache"])
        # Call count must still be 1 (zero unnecessary recapture)
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)
        self.assertEqual(res1.data["observation_id"], res2.data["observation_id"])

    def test_observation_older_than_ttl_forces_fresh_capture(self) -> None:
        """B2. Observation older than 5.0s reuse TTL forces a fresh screen capture."""
        # Prime observation
        res1 = self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Artificially age observation past 5.0 seconds
        obs = self.secure_vision.get_latest_observation()
        assert obs is not None
        obs.timestamp = time.time() - 6.0

        # Ask question requesting reuse -> MUST force fresh capture
        res2 = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What does that say?", "reuse_cache": True},
        })
        self.assertFalse(res2.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_invalidated_cache_forces_fresh_capture(self) -> None:
        """B3. Explicitly invalidated observation cache forces fresh capture on subsequent query."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Invalidate cache explicitly
        self.skill.invalidate_observation_cache(reason="test_invalidation")
        self.assertIsNone(self.secure_vision.get_latest_observation())

        # Next query must perform fresh capture
        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Where is that button?", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    # -----------------------------------------------------------------------
    # C. Compound Identity Mismatch
    # -----------------------------------------------------------------------

    def test_hwnd_change_prevents_reuse(self) -> None:
        """C1. Change in HWND between observations prevents reuse and forces fresh capture."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Simulate active window switch (different HWND)
        self.secure_vision.get_active_window_identity = MagicMock(
            return_value=(9999, self.default_title, self.default_proc, self.default_bounds)
        )

        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Where is the button?", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_title_change_prevents_reuse(self) -> None:
        """C2. Change in window title prevents reuse and forces fresh capture."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Title changed
        self.secure_vision.get_active_window_identity = MagicMock(
            return_value=(self.default_hwnd, "Different Document - Code", self.default_proc, self.default_bounds)
        )

        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Where is the button?", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_process_change_prevents_reuse(self) -> None:
        """C3. Change in process name prevents reuse and forces fresh capture."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Process changed
        self.secure_vision.get_active_window_identity = MagicMock(
            return_value=(self.default_hwnd, self.default_title, "other_process.exe", self.default_bounds)
        )

        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Where is the button?", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    def test_bounds_change_prevents_reuse(self) -> None:
        """C4. Change in window bounds/geometry prevents reuse and forces fresh capture."""
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Bounds changed (window moved/resized)
        self.secure_vision.get_active_window_identity = MagicMock(
            return_value=(self.default_hwnd, self.default_title, self.default_proc, (100, 100, 500, 500))
        )

        res = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "Where is the button?", "reuse_cache": True},
        })
        self.assertFalse(res.data["reused_cache"])
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 2)

    # -----------------------------------------------------------------------
    # D. Privacy & Security Re-authorization
    # -----------------------------------------------------------------------

    def test_foreground_switch_to_sensitive_blocks_reuse_and_raises(self) -> None:
        """D1. Switching foreground to a sensitive window prevents cache reuse and raises CaptureBlockedError."""
        # Prime cache with safe window
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

        # Foreground window switches to password vault
        self.secure_vision._resolve_window_metadata = MagicMock(
            return_value=("Bitwarden - Vault", "bitwarden.exe")
        )

        with self.assertRaises((CaptureBlockedError, SkillExecutionError)):
            self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": "What is the password?", "reuse_cache": True},
            })
        # Capture engine must NOT be invoked for sensitive window
        self.assertEqual(self.mock_engine.capture_active_window.call_count, 1)

    def test_zero_raw_screenshot_data_in_skill_result(self) -> None:
        """D2. SystemSkillResult data contains zero raw pixels, byte buffers, or base64 data."""
        res = self.skill.execute({"operation": "ask_screen", "parameters": {"question": "What is on screen?"}})
        self.assertTrue(res.success)
        for k, v in res.data.items():
            self.assertNotEqual(k, "raw_data")
            self.assertNotEqual(k, "pixels")
            self.assertNotEqual(k, "base64")
            self.assertNotIsInstance(v, bytes)

    # -----------------------------------------------------------------------
    # E. Temporal Queries Force Fresh Capture
    # -----------------------------------------------------------------------

    def test_temporal_queries_force_fresh_capture(self) -> None:
        """E1. Temporal questions ('what changed', 'what is happening now', 'is it still there') force fresh capture."""
        temporal_queries = [
            "what changed?",
            "what just changed?",
            "what is happening now?",
            "is it still there?",
            "has anything changed right now?",
        ]
        for idx, q in enumerate(temporal_queries, start=1):
            # Prime cache
            self.skill.execute({"operation": "capture_screen", "target": "active_window"})
            prev_count = self.mock_engine.capture_active_window.call_count

            # Temporal query with reuse_cache=True must STILL force fresh capture
            res = self.skill.execute({
                "operation": "ask_screen",
                "parameters": {"question": q, "reuse_cache": True},
            })
            self.assertFalse(res.data["reused_cache"], f"Temporal query '{q}' should NOT reuse cache")
            self.assertEqual(self.mock_engine.capture_active_window.call_count, prev_count + 1)

    # -----------------------------------------------------------------------
    # F. Verification Questions Force Fresh Capture
    # -----------------------------------------------------------------------

    def test_verification_questions_route_to_verify_and_force_fresh(self) -> None:
        """F1. Questions like 'did that work', 'did it open', 'did the page open' route to verify_screen_state and force fresh capture."""
        ver_questions = [
            "did that work?",
            "did it open?",
            "did the page open?",
            "verify screen state",
        ]
        for q in ver_questions:
            self.assertTrue(self.skill.can_handle(q), f"Should handle verification query '{q}'")
            op, target, params, _ = self.skill.parse_command(q)
            self.assertEqual(op, "verify_screen_state")

            prev_count = self.mock_engine.capture_active_window.call_count
            res = self.skill.execute({"operation": op, "parameters": params})
            self.assertTrue(res.success)
            self.assertFalse(res.data["reused_cache"])
            self.assertEqual(self.mock_engine.capture_active_window.call_count, prev_count + 1)

    # -----------------------------------------------------------------------
    # G. Planner Closed-Loop & Event Invalidation
    # -----------------------------------------------------------------------

    def test_planner_known_actions_contains_vision_actions(self) -> None:
        """G1. Planner KNOWN_ACTIONS and prompt schemas include ask_screen and verify_screen_state."""
        self.assertIn("ask_screen", KNOWN_ACTIONS)
        self.assertIn("verify_screen_state", KNOWN_ACTIONS)

        planner = Planner()
        self.assertIn("ask_screen", planner._allowed_actions)
        self.assertIn("verify_screen_state", planner._allowed_actions)

    def test_executor_resolves_vision_actions(self) -> None:
        """G2. Executor resolves ask_screen and verify_screen_state to vision skill."""
        sm = SkillManager(container_instance=self.container, event_bus_instance=self.event_bus)
        sm.register(self.skill)

        executor = Executor(
            container_instance=self.container,
            skill_manager_instance=sm,
            event_bus_instance=self.planner_bus,
            auto_register_in_container=False,
        )

        for act in ("ask_screen", "verify_screen_state"):
            t = Task(id="t1", action=act, parameters={"question": "Test", "condition": "Test"})
            handler = executor._resolve_from_skill_manager(act, task=t)
            self.assertIsNotNone(handler, f"Executor failed to resolve '{act}'")

    def test_ui_mutating_event_invalidates_cache_in_closed_loop(self) -> None:
        """G3. UI-mutating action in planner/skills emits event that invalidates vision cache before verify."""
        # 1. OBSERVE: Prime observation cache
        self.skill.execute({"operation": "capture_screen", "target": "active_window"})
        self.assertIsNotNone(self.secure_vision.get_latest_observation())

        # 2. ACT: Simulate execution of a UI mutating operation (e.g. open_app)
        self.planner_bus.publish(
            SystemSkillCompleted(
                execution_id="exec_1",
                skill_name="app",
                operation="open_app",
                target="notepad",
                duration=0.2,
                success=True,
            )
        )

        # Cache must be automatically invalidated!
        self.assertIsNone(self.secure_vision.get_latest_observation())

        # 3. VERIFY: Visual state verification occurs with fresh observation
        self.mock_ai.generate.return_value = AIResponse(
            content="VERIFIED: Notepad window is open and visible.",
            model="gemini-2.5-flash",
        )
        res_verify = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "Notepad is open"},
        })
        self.assertTrue(res_verify.data["verified"])
        self.assertEqual(res_verify.data["status"], "verified")
        self.assertFalse(res_verify.data["reused_cache"])

    # -----------------------------------------------------------------------
    # H. 3-Way Verification Semantics
    # -----------------------------------------------------------------------

    def test_three_way_verification_semantics(self) -> None:
        """H1. Three-way verification verdicts remain distinct: VERIFIED (True), NOT VERIFIED (False), UNCERTAIN (None)."""
        # 1. VERIFIED
        self.mock_ai.generate.return_value = AIResponse(
            content="VERIFIED: The document was saved successfully.",
            model="gemini-2.5-flash",
        )
        res_v = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "Document is saved"},
        })
        self.assertIs(res_v.data["verified"], True)
        self.assertEqual(res_v.data["status"], "verified")

        # 2. NOT VERIFIED
        self.mock_ai.generate.return_value = AIResponse(
            content="NOT VERIFIED: The document tab still shows an unsaved asterisk.",
            model="gemini-2.5-flash",
        )
        res_nv = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "Document is saved"},
        })
        self.assertIs(res_nv.data["verified"], False)
        self.assertEqual(res_nv.data["status"], "not_verified")

        # 3. UNCERTAIN
        self.mock_ai.generate.return_value = AIResponse(
            content="UNCERTAIN: The window title bar is truncated and cannot confirm state.",
            model="gemini-2.5-flash",
        )
        res_u = self.skill.execute({
            "operation": "verify_screen_state",
            "parameters": {"condition": "Document is saved"},
        })
        self.assertIsNone(res_u.data["verified"])
        self.assertEqual(res_u.data["status"], "uncertain")

    # -----------------------------------------------------------------------
    # I. Memory Privacy Invariants
    # -----------------------------------------------------------------------

    def test_memory_privacy_zero_raw_data_in_conversation_memory(self) -> None:
        """I1. ConversationMemory stores only clean text and safe metadata; zero raw pixels or base64."""
        res = self.skill.execute({"operation": "ask_screen", "parameters": {"question": "What is on screen?"}})
        mem = ConversationMemory(
            content=res.message,
            role="assistant",
            metadata={"observation_id": res.data["observation_id"], "target": res.data["target"]},
        )
        serialized = mem.to_dict()
        raw_json = str(serialized)
        self.assertNotIn("raw_data", raw_json)
        self.assertNotIn("pixels", raw_json)
        self.assertNotIn("base64", raw_json)

    def test_memory_privacy_zero_raw_data_in_execution_memory(self) -> None:
        """I2. ExecutionMemory from execution result contains zero raw image buffers or base64."""
        res = self.skill.execute({"operation": "ask_screen", "parameters": {"question": "What is on screen?"}})
        exec_mem = ExecutionMemory(
            records=[
                TaskExecutionRecord(
                    task_id="t_vision",
                    action="ask_screen",
                    target=None,
                    status=TaskStatus.COMPLETED,
                    duration=0.1,
                    output=res.data,
                )
            ]
        )
        d = exec_mem.to_dict()
        raw_str = str(d)
        self.assertNotIn("raw_data", raw_str)
        self.assertNotIn("pixels", raw_str)
        self.assertNotIn("base64", raw_str)

    # -----------------------------------------------------------------------
    # J. Voice & SystemSkillResult Compatibility
    # -----------------------------------------------------------------------

    def test_system_skill_result_message_formatting(self) -> None:
        """J1. SystemSkillResult.message cleanly formats follow-up Q&A and verification messages into speakable text."""
        res_qa = SystemSkillResult(
            success=True,
            operation="ask_screen",
            data={"answer": "The submit button is located in the bottom right corner."},
        )
        self.assertEqual(res_qa.message, "The submit button is located in the bottom right corner.")

        res_ver = SystemSkillResult(
            success=True,
            operation="verify_screen_state",
            data={"verified": True, "status": "verified", "reason": "The login succeeded."},
        )
        self.assertEqual(res_ver.message, "The condition was verified. The login succeeded.")

    def test_contextual_prompt_injected_on_reused_followup(self) -> None:
        """J2. Reusing an observation for a follow-up question injects prior visual context into multimodal prompt."""
        # 1. Initial screen question
        self.mock_ai.generate.return_value = AIResponse(
            content="I see a web browser with a blue submit button on the right.",
            model="gemini-2.5-flash",
        )
        res1 = self.skill.execute({"operation": "ask_screen", "parameters": {"question": "What is on my screen?"}})
        self.assertFalse(res1.data["reused_cache"])

        # Check prompt sent in initial turn
        initial_payload = self.mock_ai.generate.call_args[0][0]
        initial_prompt = str(initial_payload)
        self.assertIn("What is on my screen?", initial_prompt)
        self.assertNotIn("Previous Visual Discussion", initial_prompt)

        # 2. Conversational follow-up question about the same observation
        self.mock_ai.generate.return_value = AIResponse(
            content="The button on the right submits the active form.",
            model="gemini-2.5-flash",
        )
        res2 = self.skill.execute({
            "operation": "ask_screen",
            "parameters": {"question": "What about the button on the right?", "reuse_cache": True},
        })
        self.assertTrue(res2.data["reused_cache"])
        self.assertTrue(res2.data["has_context"])

        # Check prompt sent in follow-up turn
        followup_payload = self.mock_ai.generate.call_args[0][0]
        followup_prompt = str(followup_payload)
        self.assertIn("Previous Visual Discussion", followup_prompt)
        self.assertIn("What is on my screen?", followup_prompt)
        self.assertIn("blue submit button on the right", followup_prompt)
        self.assertIn("What about the button on the right?", followup_prompt)


if __name__ == "__main__":
    unittest.main()

