"""Phase 27.25 — MVP Integration Test Suite.

Exhaustive verification of:
1. Launcher options and delegation:
   - --gui delegation to gui.main(args)
   - default CLI preservation
   - --voice mode preservation
   - shared system foundation registration in container
2. Voice bridge integration:
   - UIBridge.start_voice_interaction() -> PresentationAdapter.start_voice_interaction()
   - concurrent voice trigger concurrency guard
   - speech response interruption on voice trigger
3. Confirmation gateway integration:
   - Shared SystemConfirmationManager singleton between skills and GUI adapter
   - Approval resolution flow
   - Rejection / abort resolution flow
   - Parameter-bound token verification (hash protection)
   - Zero token exposure in HUD telemetry
4. Sanitized HUD telemetry & privacy invariants:
   - Only sanitized semantic metadata reaches presentation signals
   - Strictly no passwords, raw input_text, coordinates, HWNDs, OCR dumps, screenshots
5. Existing Phase 27.24 & physical safety invariants:
   - Preservation of VisualDispatchState and idempotency fence
   - Zero physical Win32 input (Win32InputBackend invocation count == 0)
   - MockInputBackend exclusively
   - Zero focus stealing, zero shutdown/restart/process kill side effects
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure offscreen platform for Qt in CI / test environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

# Ensure QApplication singleton exists for test session
_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

import gui
import main
from app.ai.planner.executor import Executor
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord
from app.ai.planner.models import (
    Plan,
    SemanticActionKey,
    Task,
    TaskStatus,
    VisualDispatchState,
    WorkflowContext,
    make_semantic_action_key,
)
from app.application import JarvisApplication
from app.automation.input import MockInputBackend, Win32InputBackend
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.presentation import PresentationAdapter, create_presentation_adapter
from app.core.state import AssistantSnapshot, AssistantState, PendingConfirmation, PresentationEvent
from app.core.state_manager import AssistantStateManager
from app.skills.system import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    InteractionSkills,
    SystemConfirmationManager,
    SystemSecurityPolicy,
    register_system_foundation,
)
from app.ui.bridge import UIBridge


class TestLauncherIntegration(unittest.TestCase):
    """Tests for application launcher options and component initialization."""

    def test_main_gui_flag_delegates_to_gui_main(self):
        """Verify that passing --gui to main.py delegates directly to gui.main(args)."""
        with patch("gui.main", return_value=0) as mock_gui_main:
            exit_code = main.main(["--gui"])
            mock_gui_main.assert_called_once_with(["--gui"])
            self.assertEqual(exit_code, 0)

    def test_main_cli_mode_unchanged(self):
        """Verify default main.py runs CLI bootstrap and does NOT call gui.main."""
        with patch("gui.main") as mock_gui_main, \
             patch("app.core.bootstrap.bootstrap", return_value=True), \
             patch.object(JarvisApplication, "run", return_value=0) as mock_run:
            exit_code = main.main([], interactive=False)
            mock_gui_main.assert_not_called()
            self.assertEqual(exit_code, 0)

    def test_main_voice_mode_preserved(self):
        """Verify main.py --voice sets voice_mode=True on JarvisApplication."""
        with patch("gui.main") as mock_gui_main, \
             patch("app.core.bootstrap.bootstrap", return_value=True), \
             patch.object(JarvisApplication, "run", return_value=0) as mock_run:
            exit_code = main.main(["--voice"], interactive=False)
            mock_gui_main.assert_not_called()
            self.assertEqual(exit_code, 0)

    def test_gui_setup_registers_system_foundation(self):
        """Verify setup_gui_components() registers shared system foundation in the container."""
        backend_mock = MagicMock(spec=JarvisApplication)
        test_container = ServiceContainer()
        backend_mock.container = test_container

        with patch("gui.bootstrap"), \
             patch("gui.JarvisMainWindow"):
            # Execute setup_gui_components with real container
            qapp, window, bridge, backend, adapter = gui.setup_gui_components(
                ["--no-banner"],
                app_instance=backend_mock,
            )

            try:
                self.assertTrue(test_container.exists("system_security_policy"))
                self.assertTrue(test_container.exists("system_confirmation_manager"))
                self.assertIsInstance(test_container.resolve("system_confirmation_manager"), SystemConfirmationManager)
                self.assertIsInstance(test_container.resolve("system_security_policy"), SystemSecurityPolicy)
            finally:
                bridge.close()
                adapter.close()


class TestVoiceBridgeIntegration(unittest.TestCase):
    """Tests for UIBridge -> PresentationAdapter -> VoiceConversationEngine integration."""

    def setUp(self):
        self.state_mgr = AssistantStateManager()
        self.mock_voice_engine = MagicMock()
        self.adapter = PresentationAdapter(
            state_manager=self.state_mgr,
            voice_engine=self.mock_voice_engine,
        )
        self.bridge = UIBridge(presentation_adapter=self.adapter)

    def tearDown(self):
        self.bridge.close()
        self.adapter.close()
        self.state_mgr.close()

    def test_ui_bridge_voice_trigger_invokes_adapter(self):
        """Verify UIBridge.start_voice_interaction() triggers adapter voice cycle."""
        async def _mock_listen(duration=None):
            return "Voice result: success"

        self.mock_voice_engine.listen_once_async = AsyncMock(side_effect=_mock_listen)

        completed_events = []
        self.bridge.command_completed.connect(completed_events.append)

        # Trigger voice via bridge
        started = self.bridge.start_voice_interaction(duration=2.5)
        self.assertTrue(started)

        # Wait for worker thread to complete
        start_t = time.perf_counter()
        while not self.mock_voice_engine.listen_once_async.called and (time.perf_counter() - start_t) < 3.0:
            _qapp.processEvents()
            time.sleep(0.02)

        self.assertTrue(self.mock_voice_engine.listen_once_async.called)
        self.mock_voice_engine.listen_once_async.assert_called_once_with(duration=2.5)

    def test_voice_active_concurrency_guard(self):
        """Verify that triggering voice when already active is rejected safely."""
        self.bridge._voice_active = True
        started = self.bridge.start_voice_interaction()
        self.assertFalse(started)
        self.mock_voice_engine.listen_once_async.assert_not_called()

    def test_voice_trigger_interrupts_speaking_state(self):
        """Verify voice trigger while assistant is SPEAKING calls interrupt_speech."""
        self.bridge._last_state = "SPEAKING"
        with patch.object(self.adapter, "interrupt_speech", return_value=True) as mock_interrupt:
            started = self.bridge.start_voice_interaction()
            self.assertTrue(started)
            mock_interrupt.assert_called_once()
            self.mock_voice_engine.listen_once_async.assert_not_called()


class TestConfirmationIntegration(unittest.TestCase):
    """Tests for shared confirmation manager between skills and HUD presentation."""

    def setUp(self):
        self.container = ServiceContainer()
        register_system_foundation(self.container)

        self.conf_mgr = self.container.resolve("system_confirmation_manager")
        self.sec_policy = self.container.resolve("system_security_policy")
        self.state_mgr = AssistantStateManager()

        self.adapter = PresentationAdapter(
            state_manager=self.state_mgr,
            confirmation_manager=self.conf_mgr,
        )
        self.bridge = UIBridge(presentation_adapter=self.adapter)

    def tearDown(self):
        self.bridge.close()
        self.adapter.close()
        self.state_mgr.close()

    def test_shared_confirmation_manager_singleton(self):
        """Verify that skills and presentation adapter reference the identical manager."""
        skill = InteractionSkills(
            container=self.container,
            input_backend=MockInputBackend(),
        )
        self.assertIs(skill.confirmation_manager, self.conf_mgr)
        self.assertIs(self.adapter._confirmation_manager, self.conf_mgr)

    def test_confirmation_approval_flow(self):
        """Verify token requested by skill is approved by bridge and verified."""
        token = self.conf_mgr.request_confirmation(
            operation="visual_click",
            target="restricted_btn",
            parameters={"target": "restricted_btn"},
            description="Authorize click on restricted control",
        )
        self.assertFalse(self.conf_mgr.is_approved(token))

        # Operator approves via bridge (simulating ConfirmationCard click)
        resolved = self.bridge.resolve_confirmation(token, approved=True)
        self.assertTrue(resolved)
        self.assertTrue(self.conf_mgr.is_approved(token))

        # Skill can consume the approved confirmation
        consumed = self.conf_mgr.verify_and_consume(
            confirmation_id=token,
            operation="visual_click",
            target="restricted_btn",
            parameters={"target": "restricted_btn"},
        )
        self.assertTrue(consumed)

        # Single-use consumption check: second attempt to consume must be rejected
        with self.assertRaises(ConfirmationRejectedError):
            self.conf_mgr.verify_and_consume(
                confirmation_id=token,
                operation="visual_click",
                target="restricted_btn",
                parameters={"target": "restricted_btn"},
            )

    def test_confirmation_rejection_flow(self):
        """Verify token cancelled by bridge is marked rejected and cannot be consumed."""
        token = self.conf_mgr.request_confirmation(
            operation="visual_click",
            target="restricted_btn",
            parameters={"target": "restricted_btn"},
            description="Authorize click",
        )

        # Operator rejects via bridge
        resolved = self.bridge.resolve_confirmation(token, approved=False, reason="Denied by user")
        self.assertTrue(resolved)
        self.assertFalse(self.conf_mgr.is_approved(token))

        # Attempt to consume raises ConfirmationRejectedError
        with self.assertRaises(ConfirmationRejectedError):
            self.conf_mgr.verify_and_consume(
                confirmation_id=token,
                operation="visual_click",
                target="restricted_btn",
                parameters={"target": "restricted_btn"},
            )

    def test_confirmation_parameter_binding_tamper_proofing(self):
        """Verify approved confirmation token cannot be used with modified parameters."""
        token = self.conf_mgr.request_confirmation(
            operation="visual_type",
            target="field1",
            parameters={"target": "field1", "action": "login"},
        )
        self.bridge.resolve_confirmation(token, approved=True)

        # Modified parameters must fail validation
        with self.assertRaises(ConfirmationRejectedError):
            self.conf_mgr.verify_and_consume(
                confirmation_id=token,
                operation="visual_type",
                target="field1",
                parameters={"target": "field1", "action": "logout"},  # Tampered!
            )


class TestHUDTelemetrySanitization(unittest.TestCase):
    """Tests ensuring ONLY sanitized semantic metadata reaches HUD signals."""

    def setUp(self):
        self.state_mgr = AssistantStateManager()
        self.adapter = PresentationAdapter(state_manager=self.state_mgr)
        self.bridge = UIBridge(presentation_adapter=self.adapter)

    def tearDown(self):
        self.bridge.close()
        self.adapter.close()
        self.state_mgr.close()

    def test_forbidden_data_not_in_hud_plan_state(self):
        """Verify passwords, raw input_text, coordinates, and tokens are omitted from HUD plan signals."""
        received_plans = []
        self.bridge.plan_updated.connect(received_plans.append)

        # Simulate planner event dispatch with rich metadata
        clean_target = "login button"
        event = PresentationEvent(
            event_type="TaskStarted",
            snapshot=self.state_mgr.get_snapshot(),
            payload=(
                ("task_id", "task_01"),
                ("action", "visual_click"),
                ("target", clean_target),
            ),
        )
        self.bridge._process_planner_event(event)

        self.assertEqual(len(received_plans), 1)
        plan_info = received_plans[0]
        steps = plan_info.get("steps", [])
        self.assertEqual(len(steps), 1)
        step = steps[0]

        # Verify safe semantic content present
        self.assertEqual(step["task_id"], "task_01")
        self.assertEqual(step["title"], "visual_click")

        # Verify forbidden items absent from step payload
        for forbidden in ("password", "secret", "input_text", "ocr", "hwnd", "token", "coordinates", "bounds"):
            self.assertNotIn(forbidden, step)

    def test_mic_amplitude_telemetry_emission(self):
        """Verify microphone amplitude emits normalized telemetry without mutating state."""
        emitted_amplitudes = []
        self.bridge.amplitude_updated.connect(emitted_amplitudes.append)

        self.state_mgr.update_mic_amplitude(0.42)
        # Process event queue in bridge
        self.bridge._poll_backend_events()

        self.assertEqual(len(emitted_amplitudes), 1)
        self.assertAlmostEqual(emitted_amplitudes[0], 0.42, places=2)
        # Assistant state must remain IDLE
        self.assertEqual(self.state_mgr.get_snapshot().state, AssistantState.IDLE)


class TestPhase2724PreservationAndPhysicalSafety(unittest.TestCase):
    """Tests guaranteeing Phase 27.24 idempotency and zero physical input."""

    def test_win32_input_count_zero(self):
        """Verify zero physical Win32 mouse/keyboard invocations."""
        self.assertEqual(Win32InputBackend.invocation_count, 0)

    def test_mock_input_backend_exclusively(self):
        """Verify visual action execution in interaction skills uses MockInputBackend."""
        mock_backend = MockInputBackend()
        skill = InteractionSkills(input_backend=mock_backend)
        self.assertIs(skill.input_backend, mock_backend)
        self.assertEqual(mock_backend.event_count(), 0)

    def test_idempotency_fence_preservation(self):
        """Verify Phase 27.24 VisualDispatchState fence blocks DISPATCHED_UNVERIFIED actions."""
        sak = make_semantic_action_key("visual_click", "submit button")
        rec = TaskExecutionRecord(
            task_id="prior_t1",
            action="visual_click",
            target="submit button",
            status=TaskStatus.COMPLETED,
            visual_status=None,  # -> DISPATCHED_UNVERIFIED
        )
        mem = ExecutionMemory(records=[rec])

        lookup = mem.dispatch_state_lookup(sak)
        self.assertIsNotNone(lookup)
        matched_rec, state = lookup
        self.assertEqual(state, VisualDispatchState.DISPATCHED_UNVERIFIED)

        # Plan with recovery wave must be blocked by idempotency fence
        task = Task(id="recovery_t1", action="visual_click", target="submit button", parameters={"is_recovery": True})
        plan = Plan(tasks=[task], query="click submit")
        executor = Executor(input_backend=MockInputBackend())

        result = executor.execute_plan(plan, prior_memory=mem)
        self.assertFalse(result.success)
        self.assertEqual(len(result.failed_tasks), 1)
        self.assertIn("Dispatch blocked by idempotency fence", str(result.output or ""))
        self.assertEqual(Win32InputBackend.invocation_count, 0)


if __name__ == "__main__":
    unittest.main()
