"""Tests for Phase 27.19: End-to-End Closed-Loop Visual Action Orchestration.

Verifies:
1. IntentRouter classification of visual interaction commands into IntentType.VISION.
2. Planner decomposition of atomic visual interaction actions (click, type, toggle, etc.).
3. AIManager single-task visual action dispatch to Executor vs. conversational query preservation.
4. Executor resolving InteractionSkills via alias_map for visual actions.
5. InteractionSkills grounding semantic target strings via VisionSkills.
6. Backward compatibility for pre-grounded VisualActionTarget.
7. Preflight TOCTOU failures (stale coordinates, window mismatch, modal changed).
8. MockInputBackend usage and strict Win32InputBackend.invocation_count == 0.
9. Visual verification outcome and failure handling.
10. SystemSkillResult.to_user_message() human-readable translations and secret redaction.
11. FailureClassifier mapping to VISUAL_TOCTOU_FAILURE, VISUAL_VERIFICATION_FAILURE, VISUAL_PRECONDITION_FAILURE.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
import pytest

from app.ai.intent_router import IntentRouter, IntentType
from app.ai.planner.planner import Planner, KNOWN_ACTIONS
from app.ai.planner.executor import Executor
from app.ai.planner.memory import FailureCategory
from app.ai.planner.models import Task, Plan, ExecutionResult
from app.ai.planner.failure_classifier import FailureClassifier
from app.ai.manager import AIManager, EXECUTABLE_VISUAL_ACTIONS
from app.automation.input import MockInputBackend, Win32InputBackend
from app.automation.visual_action_adapter import (
    VisualActionAdapter,
    VisualActionType,
    VisualActionResultStatus,
    ActionPreflightValidator,
)
from app.skills.system.interaction_skills import InteractionSkills
from app.skills.system.base_system_skill import SystemSkillResult
from app.vision.models import (
    Point,
    WindowBounds,
    VisualActionSafetyTier,
    VisualActionFeasibilityStatus,
    VisualActionTarget,
    VisualGoalSpec,
)


@pytest.fixture(autouse=True)
def reset_win32_invocations():
    """Strict test safeguard: Win32 real input is never invoked."""
    Win32InputBackend.reset_invocation_count()
    Win32InputBackend.enable_test_safety_guard()
    yield
    assert Win32InputBackend.invocation_count == 0, (
        f"CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked {Win32InputBackend.invocation_count} times!"
    )


def make_target(
    action_type: VisualActionType = VisualActionType.CLICK,
    name: str = "Submit Button",
    point: Optional[Point] = Point(160, 220),
    bounds: Optional[WindowBounds] = WindowBounds(100, 200, 220, 240),
    safety_tier: VisualActionSafetyTier = VisualActionSafetyTier.SAFE,
    feasibility: VisualActionFeasibilityStatus = VisualActionFeasibilityStatus.FEASIBLE,
    requires_confirmation: bool = False,
    confidence: float = 0.95,
    window_handle: Optional[int] = 12345,
    grounded_at: Optional[float] = None,
    input_text: Optional[str] = None,
    expected_outcome: Optional[VisualGoalSpec] = None,
) -> VisualActionTarget:
    """Helper creating test VisualActionTarget instances."""
    return VisualActionTarget(
        target_id="act_test_001",
        action_type=action_type,
        target_element_name=name,
        target_point=point,
        bounds=bounds,
        safety_tier=safety_tier,
        feasibility=feasibility,
        requires_confirmation=requires_confirmation,
        confidence=confidence,
        reason="Target is feasible.",
        expected_outcome=expected_outcome,
        metadata={"element_name": name},
        window_handle=window_handle,
        grounded_at=grounded_at if grounded_at is not None else time.monotonic(),
        input_text=input_text,
    )


# ==============================================================================
# 1. IntentRouter Visual Interaction Tests
# ==============================================================================

class TestIntentRouterVisualInteractions:
    """Verify IntentRouter routes visual interaction commands to IntentType.VISION."""

    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.mark.parametrize("query", [
        "click the login button",
        "double click the submit icon",
        "type admin into username field",
        "clear and type test@example.com into email",
        "toggle the dark mode switch",
        "select option dark from settings",
        "dismiss the cookie modal",
        "close the dialog",
    ])
    def test_visual_interaction_queries_route_to_vision(self, router, query):
        result = router.classify(query)
        assert result.intent == IntentType.VISION, f"Expected VISION for '{query}', got {result.intent}"
        assert result.confidence >= 0.85

    @pytest.mark.parametrize("query", [
        "what is the weather today?",
        "tell me a joke",
        "how does quantum computing work",
    ])
    def test_conversational_queries_do_not_route_to_vision(self, router, query):
        result = router.classify(query)
        assert result.intent != IntentType.VISION


# ==============================================================================
# 2. Planner Visual Action Decomposition Tests
# ==============================================================================

class TestPlannerVisualActions:
    """Verify Planner handles visual interaction actions."""

    @pytest.fixture
    def planner(self):
        p = Planner()
        p._provider = MagicMock()
        return p

    def test_known_actions_contains_all_visual_actions(self):
        expected = {
            "visual_click",
            "visual_double_click",
            "visual_type",
            "visual_clear_and_type",
            "visual_select",
            "visual_toggle",
            "visual_dismiss_modal",
            "visual_interact",
        }
        for act in expected:
            assert act in KNOWN_ACTIONS, f"Action '{act}' missing from KNOWN_ACTIONS"

    def test_parse_single_action_click(self, planner):
        task = planner._parse_single_action("click the login button")
        assert task is not None
        assert task.action == "visual_click"
        assert "login" in task.target.lower()

    def test_parse_single_action_double_click(self, planner):
        task = planner._parse_single_action("double click the desktop folder")
        assert task is not None
        assert task.action == "visual_double_click"
        assert "folder" in task.target.lower()

    def test_parse_single_action_type(self, planner):
        task = planner._parse_single_action("type hello world into search input")
        assert task is not None
        assert task.action == "visual_type"
        assert task.parameters.get("input_text") == "hello world"
        assert "search" in task.target.lower()

    def test_parse_single_action_clear_and_type(self, planner):
        task = planner._parse_single_action("clear and type admin into username")
        assert task is not None
        assert task.action == "visual_clear_and_type"
        assert task.parameters.get("input_text") == "admin"
        assert "username" in task.target.lower()

    def test_parse_single_action_toggle(self, planner):
        task = planner._parse_single_action("toggle the dark mode switch")
        assert task is not None
        assert task.action == "visual_toggle"
        assert "dark mode" in task.target.lower()

    def test_parse_single_action_dismiss_modal(self, planner):
        task = planner._parse_single_action("dismiss modal")
        assert task is not None
        assert task.action == "visual_dismiss_modal"
        assert "modal" in task.target.lower()


# ==============================================================================
# 3. AIManager Single-Task Dispatch Invariants
# ==============================================================================

class TestAIManagerDispatchInvariants:
    """Verify AIManager dispatches single-task visual actions to Executor while preserving conversational flow."""

    def test_executable_visual_actions_defined(self):
        assert "visual_click" in EXECUTABLE_VISUAL_ACTIONS
        assert "visual_type" in EXECUTABLE_VISUAL_ACTIONS
        assert "visual_interact" in EXECUTABLE_VISUAL_ACTIONS

    def test_single_task_visual_action_dispatches_to_executor(self):
        ai = AIManager.__new__(AIManager)
        ai._lock = threading.RLock()
        ai._planner = MagicMock()
        ai._executor = MagicMock()
        ai._intent_router = MagicMock()
        ai._container = None
        ai._llm_service = MagicMock()
        ai._memory_manager = None
        ai._logger = MagicMock()
        ai._event_bus = MagicMock()
        ai._last_plan = None
        ai._auto_replan = False
        ai._max_replans = 0

        # Planner returns a plan with 1 task: visual_click
        visual_task = Task(id="1", action="visual_click", target="login")
        plan = Plan(query="click login", tasks=[visual_task])
        ai._planner.create_plan.return_value = plan

        # Executor mock
        exec_result = ExecutionResult(
            success=True,
            completed_tasks=[visual_task],
            output="Clicked 'login' successfully",
        )
        ai._executor.execute_plan.return_value = exec_result

        # Call generate
        resp = ai.generate("click the login button")
        assert resp.model == "executor"
        assert "Visual Click login" in resp.content
        ai._executor.execute_plan.assert_called_once_with(plan)

    def test_ordinary_conversational_query_preserves_conversational_path(self):
        ai = AIManager.__new__(AIManager)
        ai._lock = threading.RLock()
        ai._planner = MagicMock()
        ai._executor = MagicMock()
        ai._intent_router = MagicMock()
        ai._container = None
        ai._llm_service = MagicMock()
        ai._memory_manager = None
        ai._logger = MagicMock()
        ai._event_bus = MagicMock()
        ai._prompt_builder = MagicMock()
        ai._provider_router = MagicMock()
        mock_provider = MagicMock()
        mock_provider.model = "mock-model"
        mock_provider.generate.return_value = MagicMock(
            content="Hello! How can I assist you today?",
            model="mock-model",
            total_tokens=10,
            duration=0.05,
            metadata={},
        )
        ai._provider = mock_provider
        ai._provider_router.resolve_provider.return_value = mock_provider
        ai._config = MagicMock()
        ai._last_plan = None
        ai._auto_replan = False
        ai._max_replans = 0

        # Planner returns empty tasks
        plan = Plan(query="hello jarvis", tasks=[])
        ai._planner.create_plan.return_value = plan

        # Intent router classifies as chat
        ai._intent_router.classify.return_value = MagicMock(intent=IntentType.CHAT)

        resp = ai.generate("hello jarvis")
        assert "Hello! How can I assist you today?" in resp.content
        # Executor must NOT be called for empty tasks / conversational
        ai._executor.execute_plan.assert_not_called()


# ==============================================================================
# 4. Executor Resolution of InteractionSkills via alias_map
# ==============================================================================

class TestExecutorVisualSkillResolution:
    """Verify Executor resolves visual interaction actions to InteractionSkills."""

    def test_executor_skill_manager_alias_resolution(self):
        mock_sm = MagicMock(spec=["get"])
        interaction_skill = MagicMock()
        mock_sm.get.side_effect = lambda name: interaction_skill if name == "interaction" else None

        executor = Executor(skill_manager_instance=mock_sm)
        handler = executor._resolve_from_skill_manager("visual_click")
        assert handler is not None
        mock_sm.get.assert_any_call("interaction")

    def test_executor_container_resolution_and_dispatch(self):
        from app.core.container import ServiceContainer
        container = ServiceContainer()
        interaction_skill = MagicMock()
        interaction_skill.execute.return_value = SystemSkillResult(
            operation="visual_click",
            success=True,
            data={"status": VisualActionResultStatus.SUCCESS.value, "action_type": "click"},
        )
        container.register_singleton("interaction", interaction_skill)

        executor = Executor(container_instance=container)
        task = Task(id="1", action="visual_click", target="login")
        plan = Plan(
            query="click login",
            tasks=[task],
        )
        result = executor.execute_plan(plan)
        assert result.success is True
        interaction_skill.execute.assert_called_once()


# ==============================================================================
# 5. Semantic Grounding & Observation Reuse in InteractionSkills
# ==============================================================================

class TestInteractionSkillsSemanticGrounding:
    """Verify InteractionSkills grounds semantic targets via VisionSkills without duplicating perception."""

    @pytest.fixture
    def mock_vision_skills(self):
        vs = MagicMock()
        target = make_target(point=Point(160, 220), bounds=WindowBounds(100, 200, 220, 240))
        vs.execute.return_value = {"success": True, "target": target}
        return vs

    @pytest.fixture
    def mock_backend(self):
        return MockInputBackend()

    @pytest.fixture
    def preflight(self):
        return ActionPreflightValidator()

    @pytest.fixture
    def adapter(self, mock_backend, preflight):
        return VisualActionAdapter(input_backend=mock_backend, preflight_validator=preflight)

    def test_interaction_skills_grounds_semantic_string(self, mock_vision_skills, adapter, mock_backend):
        skill = InteractionSkills(action_adapter=adapter, vision_skills=mock_vision_skills)

        res = skill.execute({
            "action_type": "click",
            "target": "login button",
        })

        assert res.success is True
        mock_vision_skills.execute.assert_called_once()
        call_args = mock_vision_skills.execute.call_args[0][0]
        assert call_args["operation"] == "ground_visual_action"
        assert call_args["target"] == "login button"

        # Check mock backend recorded click event
        events = mock_backend.get_events()
        assert len(events) == 1
        assert events[0].event_type == "click"
        assert events[0].point.x == 160
        assert events[0].point.y == 220

    def test_interaction_skills_pregrounded_target_backward_compatibility(self, adapter, mock_backend):
        mock_vs = MagicMock()
        skill = InteractionSkills(action_adapter=adapter, vision_skills=mock_vs)

        pregrounded = make_target(point=Point(75, 75), bounds=WindowBounds(50, 50, 100, 100))

        res = skill.execute({
            "action_type": "click",
            "action_target": pregrounded,
        })

        assert res.success is True
        # VisionSkills should not be invoked when target is already pre-grounded
        mock_vs.execute.assert_not_called()
        events = mock_backend.get_events()
        assert len(events) == 1
        assert events[0].point.x == 75
        assert events[0].point.y == 75

    def test_interaction_skills_grounding_failure(self, adapter):
        failing_vs = MagicMock()
        failing_vs.execute.return_value = {"success": False, "error": "Element 'unicorn' not found on screen"}
        skill = InteractionSkills(action_adapter=adapter, vision_skills=failing_vs)

        res = skill.execute({
            "action_type": "click",
            "target": "unicorn",
        })

        assert res.success is False
        assert res.data["status"] == VisualActionResultStatus.GROUNDING_FAILED.value


# ==============================================================================
# 6. Preflight TOCTOU Failures
# ==============================================================================

class TestPreflightTOCTOUFailures:
    """Verify preflight TOCTOU failures prevent input dispatch and report exact statuses."""

    @pytest.fixture
    def mock_backend(self):
        return MockInputBackend()

    @pytest.fixture
    def preflight(self):
        return ActionPreflightValidator()

    @pytest.fixture
    def adapter(self, mock_backend, preflight):
        return VisualActionAdapter(input_backend=mock_backend, preflight_validator=preflight)

    def test_stale_coordinates_failure(self, adapter, mock_backend):
        # Target with expired TTL
        invalid_target = make_target(
            bounds=WindowBounds(100, 100, 200, 200),
            point=Point(150, 150),
            grounded_at=time.monotonic() - 100.0,
        )
        result = adapter.execute_target(invalid_target)
        assert result.success is False
        assert result.status == VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES
        assert mock_backend.event_count() == 0

    def test_window_mismatch_failure(self, adapter, mock_backend):
        target = make_target(window_handle=99999)
        # Pass different active window handle to preflight context
        result = adapter.execute_target(
            target,
            current_window_info={"hwnd": 11111, "title": "Different Window"},
        )
        assert result.success is False
        assert result.status == VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH
        assert mock_backend.event_count() == 0


# ==============================================================================
# 7. FailureClassifier Mapping Tests
# ==============================================================================

class TestFailureClassifierVisualMappings:
    """Verify FailureClassifier maps visual errors to specific FailureCategory members."""

    @pytest.fixture
    def classifier(self):
        return FailureClassifier()

    def test_classify_visual_toctou_failure(self, classifier):
        task = Task(action="visual_click", target="login")
        cat = classifier.classify(task, "Preflight TOCTOU check failed: PREFLIGHT_STALE_COORDINATES")
        assert cat == FailureCategory.VISUAL_TOCTOU_FAILURE

        cat2 = classifier.classify(task, "Target window changed: PREFLIGHT_WINDOW_MISMATCH")
        assert cat2 == FailureCategory.VISUAL_TOCTOU_FAILURE

        cat3 = classifier.classify(task, "Modal appeared: PREFLIGHT_MODAL_CHANGED")
        assert cat3 == FailureCategory.VISUAL_TOCTOU_FAILURE

    def test_classify_visual_verification_failure(self, classifier):
        task = Task(action="visual_click", target="submit")
        cat = classifier.classify(task, "Visual verification failed: post-condition check unsatisfied")
        assert cat == FailureCategory.VISUAL_VERIFICATION_FAILURE

        cat2 = classifier.classify(task, "VERIFICATION_FAILED: element still present")
        assert cat2 == FailureCategory.VISUAL_VERIFICATION_FAILURE

    def test_classify_visual_precondition_failure(self, classifier):
        task = Task(action="visual_type", target="password")
        cat = classifier.classify(task, "PRECONDITION_FAILED: Target element not visible")
        assert cat == FailureCategory.VISUAL_PRECONDITION_FAILURE

        cat2 = classifier.classify(task, "Visual grounding failed: element not found")
        assert cat2 == FailureCategory.VISUAL_PRECONDITION_FAILURE


# ==============================================================================
# 8. Human-Friendly Translation & Secret Redaction
# ==============================================================================

class TestUserMessageFormattingAndPrivacy:
    """Verify SystemSkillResult.to_user_message() handles all visual statuses and protects secrets."""

    def test_success_message(self):
        res = SystemSkillResult(
            operation="visual_click",
            success=True,
            data={"status": VisualActionResultStatus.SUCCESS.value, "action_type": "click", "target": "Login Button"},
        )
        msg = res.to_user_message()
        assert "Successfully clicked 'Login Button'." in msg

    def test_stale_coordinates_message(self):
        res = SystemSkillResult(
            operation="visual_click",
            success=False,
            data={"status": VisualActionResultStatus.PREFLIGHT_STALE_COORDINATES.value, "action_type": "click", "target": "Submit"},
        )
        msg = res.to_user_message()
        assert "Target 'Submit' changed position or disappeared" in msg

    def test_window_mismatch_message(self):
        res = SystemSkillResult(
            operation="visual_click",
            success=False,
            data={"status": VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH.value, "action_type": "click"},
        )
        msg = res.to_user_message()
        assert "active window changed unexpectedly" in msg

    def test_modal_changed_message(self):
        res = SystemSkillResult(
            operation="visual_click",
            success=False,
            data={"status": VisualActionResultStatus.PREFLIGHT_MODAL_CHANGED.value, "action_type": "click"},
        )
        msg = res.to_user_message()
        assert "unexpected modal or dialog appeared" in msg

    def test_verification_failed_message(self):
        res = SystemSkillResult(
            operation="visual_click",
            success=False,
            data={"status": VisualActionResultStatus.VERIFICATION_FAILED.value, "action_type": "click", "target": "Checkbox"},
        )
        msg = res.to_user_message()
        assert "could not verify the expected screen changes" in msg

    def test_secret_redaction_in_user_message(self):
        # Even if data has secret/text, it must NEVER appear in user message
        res = SystemSkillResult(
            operation="visual_type",
            success=True,
            data={
                "status": VisualActionResultStatus.SUCCESS.value,
                "action_type": "type",
                "target": "Password Field",
                "input_text": "SuperSecretPassword123!",
            },
            error="Internal raw msg with SuperSecretPassword123!",
        )
        msg = res.to_user_message()
        assert "SuperSecretPassword123!" not in msg
        assert "Successfully typed into 'Password Field'." in msg
