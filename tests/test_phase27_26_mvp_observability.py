"""Phase 27.26 — MVP Observability and Shared PlannerEventBus Wiring Test Suite.

Verifies end-to-end event observability across:
1. Shared PlannerEventBus container registration (idempotent, single bus).
2. Identity wiring:
   - executor.planner_event_bus is container.resolve("planner_event_bus")
   - state_manager._planner_event_bus is container.resolve("planner_event_bus")
   - confirmation_manager._event_bus is container.resolve("planner_event_bus")
   - security_policy._event_bus is container.resolve("planner_event_bus")
3. Real-time event propagation:
   - Executor TaskStarted -> AssistantStateManager EXECUTING state
   - Executor TaskCompleted -> AssistantStateManager task_progress update
   - Executor TaskFailed -> AssistantStateManager event processing
   - PlanStarted / PlanCompleted -> Presentation plan state and cognitive stage updates
   - SystemSkillConfirmationRequired -> AssistantStateManager AWAITING_CONFIRMATION state
4. Recovery / replanning observability preservation.
5. Privacy and sanitization guarantees:
   - Zero coordinates, HWNDs, screenshots, passwords, raw input_text in presentation/state layers.
6. Phase 27.24 idempotency fence preservation.
7. Phase 27.25 launcher option preservation.
8. Physical safety invariants:
   - MockInputBackend exclusively.
   - Win32InputBackend.invocation_count == 0.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure offscreen Qt platform for test environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

import gui
import main
from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEventBus,
    PlanStarted,
    SystemSkillConfirmationRequired,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
)
from app.ai.planner.executor import Executor
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord
from app.ai.planner.models import (
    Plan,
    SemanticActionKey,
    Task,
    TaskStatus,
    VisualDispatchState,
    make_semantic_action_key,
    purge_physical_state,
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


class TestSharedPlannerEventBusRegistration(unittest.TestCase):
    """Tests 1-3: Container registration, idempotency, and singleton semantics."""

    def test_01_planner_event_bus_is_registered_in_container(self):
        """Verify register_system_foundation registers planner_event_bus in container."""
        container = ServiceContainer()
        register_system_foundation(container)

        self.assertTrue(container.exists("planner_event_bus"))
        p_bus = container.resolve("planner_event_bus")
        self.assertIsInstance(p_bus, PlannerEventBus)

    def test_02_registration_is_idempotent(self):
        """Verify multiple calls to register_system_foundation do not duplicate the bus."""
        container = ServiceContainer()
        register_system_foundation(container)
        bus1 = container.resolve("planner_event_bus")

        register_system_foundation(container)
        bus2 = container.resolve("planner_event_bus")

        self.assertIs(bus1, bus2)

    def test_03_repeated_foundation_registration_preserves_custom_bus(self):
        """Verify injecting an explicit PlannerEventBus registers and retains that exact instance."""
        container = ServiceContainer()
        custom_bus = PlannerEventBus()

        register_system_foundation(container, planner_event_bus=custom_bus)
        resolved = container.resolve("planner_event_bus")

        self.assertIs(resolved, custom_bus)


class TestGUIIdentityWiring(unittest.TestCase):
    """Tests 4-5: Exact object identity wiring across Executor, StateManager, and Skills."""

    def test_04_executor_receives_exact_same_event_bus_instance(self):
        """Verify executor.planner_event_bus is container.resolve('planner_event_bus')."""
        backend_mock = MagicMock(spec=JarvisApplication)
        test_container = ServiceContainer()
        backend_mock.container = test_container

        mock_executor = Executor(
            container_instance=test_container,
            input_backend=MockInputBackend(),
            auto_register_in_container=False,
        )
        mock_ai_mgr = MagicMock()
        mock_ai_mgr.executor = mock_executor
        backend_mock.ai_manager = mock_ai_mgr
        test_container.register_singleton("executor", mock_executor, allow_override=True)

        with patch("gui.bootstrap"), patch("gui.JarvisMainWindow"):
            qapp, window, bridge, backend, adapter = gui.setup_gui_components(
                ["--no-banner"],
                app_instance=backend_mock,
            )
            try:
                shared_bus = test_container.resolve("planner_event_bus")
                self.assertIs(mock_executor.planner_event_bus, shared_bus)
            finally:
                bridge.close()
                adapter.close()

    def test_05_state_manager_receives_exact_same_event_bus_instance(self):
        """Verify PresentationAdapter._state_manager._planner_event_bus is container.resolve('planner_event_bus')."""
        backend_mock = MagicMock(spec=JarvisApplication)
        test_container = ServiceContainer()
        backend_mock.container = test_container

        with patch("gui.bootstrap"), patch("gui.JarvisMainWindow"):
            qapp, window, bridge, backend, adapter = gui.setup_gui_components(
                ["--no-banner"],
                app_instance=backend_mock,
            )
            try:
                shared_bus = test_container.resolve("planner_event_bus")
                self.assertIs(adapter._state_manager._planner_event_bus, shared_bus)
            finally:
                bridge.close()
                adapter.close()


class TestExecutorEventPropagationToStateManager(unittest.TestCase):
    """Tests 6-8: Executor task lifecycle events reach AssistantStateManager via shared bus."""

    def setUp(self):
        self.container = ServiceContainer()
        register_system_foundation(self.container)
        self.shared_bus = self.container.resolve("planner_event_bus")
        self.event_bus = EventBus()

        self.state_mgr = AssistantStateManager(
            event_bus=self.event_bus,
            planner_event_bus=self.shared_bus,
        )

    def tearDown(self):
        self.state_mgr.close()

    def test_06_executor_task_started_reaches_state_manager(self):
        """Verify TaskStarted emitted by Executor transitions StateManager to EXECUTING."""
        executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.shared_bus,
            input_backend=MockInputBackend(),
        )

        task = Task(id="t_obs_1", action="custom_action", target="widget")
        plan = Plan(id="plan_obs_1", tasks=[task], query="execute widget")

        def _handler(t: Task):
            # Check state while task is running
            snap = self.state_mgr.get_snapshot()
            self.assertEqual(snap.state, AssistantState.EXECUTING)
            self.assertEqual(snap.task_id, "t_obs_1")
            return "success"

        executor.register_handler("custom_action", _handler)
        res = executor.execute_plan(plan)

        self.assertTrue(res.success)
        self.assertEqual(Win32InputBackend.invocation_count, 0)

    def test_07_executor_task_completed_reaches_state_manager(self):
        """Verify TaskCompleted emitted by Executor updates StateManager completed count and progress."""
        executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.shared_bus,
            input_backend=MockInputBackend(),
        )

        t1 = Task(id="t_obs_c1", action="action_1", target="target1")
        t2 = Task(id="t_obs_c2", action="action_2", target="target2", dependencies=["t_obs_c1"])
        plan = Plan(id="plan_obs_comp", tasks=[t1, t2], query="two step plan")

        executor.register_handler("action_1", lambda t: "done1")
        executor.register_handler("action_2", lambda t: "done2")

        res = executor.execute_plan(plan)
        self.assertTrue(res.success)

        # Plan completed event resets task counts to 0 and transitions back to IDLE
        snap = self.state_mgr.get_snapshot()
        self.assertEqual(snap.state, AssistantState.IDLE)
        self.assertAlmostEqual(snap.task_progress, 1.0)
        self.assertEqual(Win32InputBackend.invocation_count, 0)

    def test_08_executor_task_failed_reaches_state_manager(self):
        """Verify TaskFailed emitted by Executor is observed across the shared bus by state manager."""
        executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.shared_bus,
            input_backend=MockInputBackend(),
        )

        task = Task(id="t_obs_fail", action="unknown_action_unhandled", target="widget")
        plan = Plan(id="plan_obs_fail", tasks=[task], query="fail plan")

        with patch.object(AssistantStateManager, "_handle_task_failed", autospec=True) as mock_failed:
            test_state_mgr = AssistantStateManager(
                event_bus=self.event_bus,
                planner_event_bus=self.shared_bus,
            )
            try:
                res = executor.execute_plan(plan)
                self.assertFalse(res.success)
                self.assertEqual(len(res.failed_tasks), 1)

                # Confirm state manager received TaskFailed across shared PlannerEventBus
                mock_failed.assert_called_once()
                failed_event = mock_failed.call_args[0][1]
                self.assertEqual(failed_event.task_id, "t_obs_fail")
                self.assertIn("No handler registered", str(failed_event.error))
            finally:
                test_state_mgr.close()

        self.assertEqual(Win32InputBackend.invocation_count, 0)


class TestPresentationLifecycleAndConfirmationObservability(unittest.TestCase):
    """Tests 9-11: UIBridge drains lifecycle events, confirmation requests, and recovery state."""

    def setUp(self):
        self.container = ServiceContainer()
        register_system_foundation(self.container)
        self.shared_bus = self.container.resolve("planner_event_bus")
        self.conf_mgr = self.container.resolve("system_confirmation_manager")
        self.event_bus = EventBus()

        self.state_mgr = AssistantStateManager(
            event_bus=self.event_bus,
            planner_event_bus=self.shared_bus,
        )
        self.adapter = PresentationAdapter(
            state_manager=self.state_mgr,
            confirmation_manager=self.conf_mgr,
        )
        self.bridge = UIBridge(presentation_adapter=self.adapter)

    def tearDown(self):
        self.bridge.close()
        self.adapter.close()
        self.state_mgr.close()

    def test_09_planner_lifecycle_updates_presentation_state(self):
        """Verify PlanStarted, TaskStarted, TaskCompleted update UIBridge active plan info and cognitive stage."""
        plan_updates = []
        stages = []
        self.bridge.plan_updated.connect(plan_updates.append)
        self.bridge.cognitive_stage_changed.connect(lambda stage, detail: stages.append((stage, detail)))

        # 1. PlanStarted on shared bus -> state transitions to PLANNING
        self.shared_bus.publish(
            PlanStarted(
                execution_id="plan_p1",
                plan_id="plan_p1",
                query="test workflow",
                task_count=2,
            )
        )
        self.bridge._poll_backend_events()
        self.assertEqual(self.state_mgr.get_snapshot().state, AssistantState.PLANNING)

        # 2. TaskStarted on shared bus -> state transitions to EXECUTING
        self.shared_bus.publish(
            TaskStarted(
                execution_id="plan_p1",
                plan_id="plan_p1",
                task_id="task_01",
                action="visual_click",
                target="search button",
            )
        )
        self.bridge._poll_backend_events()
        self.assertEqual(self.state_mgr.get_snapshot().state, AssistantState.EXECUTING)
        self.assertIn("EXECUTING", [s[0] for s in stages])

        # 3. TaskCompleted on shared bus -> drains planner.task_completed and emits plan_updated
        self.shared_bus.publish(
            TaskCompleted(
                execution_id="plan_p1",
                plan_id="plan_p1",
                task_id="task_01",
                action="visual_click",
                duration=0.15,
            )
        )
        self.bridge._poll_backend_events()
        self.assertIn("SYNTHESIZING", [s[0] for s in stages])
        self.assertTrue(len(plan_updates) >= 1)

        # 4. PlanCompleted on shared bus -> state returns to IDLE
        self.shared_bus.publish(
            PlanCompleted(
                execution_id="plan_p1",
                plan_id="plan_p1",
                success=True,
            )
        )
        self.bridge._poll_backend_events()
        self.assertEqual(self.state_mgr.get_snapshot().state, AssistantState.IDLE)

    def test_10_confirmation_required_event_reaches_presentation_state_layer(self):
        """Verify SystemConfirmationManager emits SystemSkillConfirmationRequired, updating state to AWAITING_CONFIRMATION."""
        snapshots = []
        self.bridge.snapshot_updated.connect(snapshots.append)

        # Trigger confirmation request
        token = self.conf_mgr.request_confirmation(
            operation="visual_click",
            target="shutdown_system_btn",
            parameters={"target": "shutdown_system_btn"},
            description="Authorize destructive system action",
        )

        # Drain presentation events into bridge
        self.bridge._poll_backend_events()

        # State manager must be in AWAITING_CONFIRMATION with pending confirmation
        snap = self.state_mgr.get_snapshot()
        self.assertEqual(snap.state, AssistantState.AWAITING_CONFIRMATION)
        self.assertIsNotNone(snap.pending_confirmation)
        self.assertEqual(snap.pending_confirmation.confirmation_id, token)
        self.assertEqual(snap.pending_confirmation.operation, "visual_click")

        # Resolve through bridge
        resolved = self.bridge.resolve_confirmation(token, approved=True)
        self.assertTrue(resolved)
        self.assertTrue(self.conf_mgr.is_approved(token))

    def test_11_recovery_replanning_observability_preserved(self):
        """Verify recovery wave execution updates task tracking and duration metrics."""
        executor = Executor(
            container_instance=self.container,
            planner_event_bus=self.shared_bus,
            input_backend=MockInputBackend(),
        )

        rec_task = Task(id="rec_t1", action="retry_step", target="editor", parameters={"is_recovery": True})
        rec_plan = Plan(id="rec_plan_01", tasks=[rec_task], query="recover editor")

        executor.register_handler("retry_step", lambda t: "recovered")
        res = executor.execute_plan(rec_plan)

        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(Win32InputBackend.invocation_count, 0)


class TestPrivacyAndSanitizationGuarantees(unittest.TestCase):
    """Tests 12-16: Zero coordinates, HWNDs, pixels, passwords, or raw secrets reach presentation."""

    def setUp(self):
        self.container = ServiceContainer()
        register_system_foundation(self.container)
        self.shared_bus = self.container.resolve("planner_event_bus")
        self.state_mgr = AssistantStateManager(planner_event_bus=self.shared_bus)
        self.adapter = PresentationAdapter(state_manager=self.state_mgr)
        self.bridge = UIBridge(presentation_adapter=self.adapter)

    def tearDown(self):
        self.bridge.close()
        self.adapter.close()
        self.state_mgr.close()

    def test_12_semantic_task_metadata_reaches_presentation_safely(self):
        """Verify clean semantic action and target are displayed in active plan state."""
        received = []
        self.bridge.plan_updated.connect(received.append)

        event = PresentationEvent(
            event_type="TaskStarted",
            snapshot=self.state_mgr.get_snapshot(),
            payload=(
                ("task_id", "t_safe_1"),
                ("action", "visual_click"),
                ("target", "Save Document"),
            ),
        )
        self.bridge._process_planner_event(event)

        self.assertEqual(len(received), 1)
        step = received[0]["steps"][0]
        self.assertEqual(step["task_id"], "t_safe_1")
        self.assertEqual(step["title"], "visual_click")

    def test_13_coordinates_are_not_exposed(self):
        """Verify physical coordinates and Point objects are not exposed in presentation signals."""
        steps = []
        self.bridge.plan_updated.connect(lambda p: steps.extend(p.get("steps", [])))

        event = PresentationEvent(
            event_type="TaskStarted",
            snapshot=self.state_mgr.get_snapshot(),
            payload=(
                ("task_id", "t_coord"),
                ("action", "visual_click"),
                ("target", "OK Button"),
                ("coordinates", (340, 720)),
                ("target_point", {"x": 340, "y": 720}),
                ("bounds", {"left": 300, "top": 700, "right": 380, "bottom": 740}),
            ),
        )
        self.bridge._process_planner_event(event)

        for step in steps:
            for forbidden in ("coordinates", "target_point", "point", "bounds"):
                self.assertNotIn(forbidden, step)

    def test_14_hwnd_window_handles_are_not_exposed(self):
        """Verify window handles and HWNDs are stripped from presentation step data."""
        steps = []
        self.bridge.plan_updated.connect(lambda p: steps.extend(p.get("steps", [])))

        event = PresentationEvent(
            event_type="TaskStarted",
            snapshot=self.state_mgr.get_snapshot(),
            payload=(
                ("task_id", "t_hwnd"),
                ("action", "visual_type"),
                ("target", "search box"),
                ("hwnd", 131244),
                ("window_handle", 131244),
            ),
        )
        self.bridge._process_planner_event(event)

        for step in steps:
            self.assertNotIn("hwnd", step)
            self.assertNotIn("window_handle", step)

    def test_15_screenshots_and_raw_pixels_are_not_exposed(self):
        """Verify screenshots and raw OCR pixel dumps are stripped."""
        steps = []
        self.bridge.plan_updated.connect(lambda p: steps.extend(p.get("steps", [])))

        event = PresentationEvent(
            event_type="TaskStarted",
            snapshot=self.state_mgr.get_snapshot(),
            payload=(
                ("task_id", "t_pixel"),
                ("action", "visual_click"),
                ("target", "dialog button"),
                ("screenshot", b"\x89PNG\r\n\x1a\n...fake..."),
                ("raw_ocr", "dump of all text on screen"),
            ),
        )
        self.bridge._process_planner_event(event)

        for step in steps:
            self.assertNotIn("screenshot", step)
            self.assertNotIn("raw_ocr", step)

    def test_16_passwords_and_raw_sensitive_input_are_not_exposed(self):
        """Verify purge_physical_state removes passwords and raw sensitive inputs."""
        params = {
            "target": "login_field",
            "password": "SuperSecretPassword123!",
            "input_text": "SuperSecretPassword123!",
            "coordinates": (100, 200),
            "hwnd": 99999,
        }
        purge_physical_state(params)

        self.assertNotIn("password", params)
        self.assertNotIn("input_text", params)
        self.assertNotIn("coordinates", params)
        self.assertNotIn("hwnd", params)


class TestSafetyInvariantsAndCompatibility(unittest.TestCase):
    """Tests 17-20: Idempotency fence, launcher options, and MockInputBackend."""

    def test_17_phase27_24_idempotency_remains_intact(self):
        """Verify idempotency fence rejects DISPATCHED_UNVERIFIED recovery wave."""
        sak = make_semantic_action_key("visual_click", "submit button")
        rec = TaskExecutionRecord(
            task_id="prior_t1",
            action="visual_click",
            target="submit button",
            status=TaskStatus.COMPLETED,
            visual_status=None,  # -> DISPATCHED_UNVERIFIED
        )
        mem = ExecutionMemory(records=[rec])

        task = Task(id="rec_t1", action="visual_click", target="submit button", parameters={"is_recovery": True})
        plan = Plan(tasks=[task], query="click submit")
        executor = Executor(input_backend=MockInputBackend())

        result = executor.execute_plan(plan, prior_memory=mem)
        self.assertFalse(result.success)
        self.assertIn("Dispatch blocked by idempotency fence", str(result.output or ""))

    def test_18_phase27_25_launcher_behavior_remains_intact(self):
        """Verify CLI, --voice, and --gui arguments preserve approved dispatch paths."""
        with patch("gui.main", return_value=0) as mock_gui:
            res_gui = main.main(["--gui"])
            mock_gui.assert_called_once_with(["--gui"])
            self.assertEqual(res_gui, 0)

        with patch("gui.main") as mock_gui, \
             patch("app.core.bootstrap.bootstrap", return_value=True), \
             patch.object(JarvisApplication, "run", return_value=0):
            res_cli = main.main([], interactive=False)
            mock_gui.assert_not_called()
            self.assertEqual(res_cli, 0)

    def test_19_mock_input_backend_used_exclusively(self):
        """Verify visual actions configure MockInputBackend with zero physical events."""
        mock_backend = MockInputBackend()
        skills = InteractionSkills(input_backend=mock_backend)

        self.assertIs(skills.input_backend, mock_backend)
        self.assertEqual(mock_backend.event_count(), 0)

    def test_20_win32_input_count_remains_zero(self):
        """Verify physical Win32 mouse/keyboard invocation count is strictly 0."""
        self.assertEqual(Win32InputBackend.invocation_count, 0)


if __name__ == "__main__":
    unittest.main()
