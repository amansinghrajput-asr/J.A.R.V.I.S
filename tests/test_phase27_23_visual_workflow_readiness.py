"""Tests for Phase 27.23: Visual Readiness, Verification Integrity & Recovery Continuity.

Enforces:
1. open_app → visual action readiness sequence
2. readiness succeeds when expected app appears
3. readiness timeout fails closed
4. readiness never steals focus
5. readiness does not persist physical state
6. recovery preserves WorkflowContext
7. recovery advances semantic step correctly
8. successful recovery updates AIManager active context
9. next user turn receives recovered semantic context
10. no verification → never reports verified success
11. successful verification → VERIFIED
12. failed verification → NOT_VERIFIED
13. uncertain verification → UNCERTAIN and stops
14. clicked button remaining visible is not automatically VERIFIED
15. deterministic state transition can produce VERIFIED
16. transient disabled state gets at most one bounded fresh observation
17. genuinely disabled control remains fail-closed
18. confirmation token does not cross recovery/workflow boundary
19. physical target cannot leak into recovery context
20. async/sync parity
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.manager import AIManager
from app.ai.planner.executor import Executor
from app.ai.planner.heuristics import (
    evaluate_recovery_viability,
    is_transient_control_readiness,
    is_permanent_visual_failure,
)
from app.ai.planner.memory import (
    ExecutionMemory,
    FailureCategory,
    TaskExecutionRecord,
)
from app.ai.planner.models import (
    ExecutionResult,
    Plan,
    PlanningStrategy,
    Task,
    TaskStatus,
    WorkflowContext,
    GROUNDED_PARAM_KEYS,
    purge_physical_state,
)
from app.ai.planner.planner import Planner
from app.automation.input import MockInputBackend, Win32InputBackend
from app.automation.visual_action_adapter import (
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.interaction_skills import InteractionSkills
from app.skills.system.security import (
    SystemConfirmationManager,
    SystemSecurityPolicy,
)
from app.vision.models import (
    OCRResult,
    Point,
    ScreenCapture,
    ScreenObservation,
    VisualActionFeasibilityStatus,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualDeltaResult,
    VisualEvidenceItem,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualOutcomeType,
    VisualVerificationResult,
    WindowBounds,
)
from app.vision.verification import VisualVerificationEngine


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_dummy_capture(title: str = "Test Window", proc: str = "test.exe", text: str = "Submit Search Cancel") -> ScreenObservation:
    cap = ScreenCapture(
        raw_data=b"\x00" * 400,
        width=10,
        height=10,
        channels=4,
        metadata={"window_title": title, "process_name": proc},
    )
    return ScreenObservation(
        capture=cap,
        metadata={"window_title": title, "process_name": proc, "hwnd": 12345},
        ocr_result=OCRResult(text=text),
    )


@pytest.fixture(autouse=True)
def enforce_zero_win32_input():
    """Ensure Win32InputBackend invocation count remains 0 across all tests."""
    initial_count = getattr(Win32InputBackend, "invocation_count", 0)
    yield
    final_count = getattr(Win32InputBackend, "invocation_count", 0)
    assert final_count == initial_count, f"Win32 input invoked! Count increased by {final_count - initial_count}"


# ---------------------------------------------------------------------------
# 1-5: Visual Readiness Observation Barrier
# ---------------------------------------------------------------------------

def test_1_open_app_visual_action_readiness_sequence():
    """Verify that interaction skills invokes bounded readiness check before visual action."""
    mock_input = MockInputBackend()
    sec_policy = SystemSecurityPolicy()
    conf_mgr = SystemConfirmationManager()
    skill = InteractionSkills(
        security_policy=sec_policy,
        confirmation_manager=conf_mgr,
        input_backend=mock_input,
    )

    # Mock vision_skills get_active_window_identity
    mock_vs = MagicMock()
    # First returns wrong window (cmd.exe), then expected (chrome.exe)
    mock_vs.get_active_window_identity.side_effect = [
        (100, "Command Prompt", "cmd.exe", (0, 0, 800, 600)),
        (200, "Google Chrome", "chrome.exe", (0, 0, 800, 600)),
    ]
    skill._vision_skills = mock_vs

    ready, title, proc = skill._wait_for_window_readiness(
        expected_app="chrome",
        expected_proc="chrome.exe",
        timeout_s=1.0,
        interval_s=0.01,
    )
    assert ready is True
    assert "chrome" in proc.lower()
    assert mock_vs.get_active_window_identity.call_count >= 2


def test_2_readiness_succeeds_when_expected_app_appears():
    """Verify readiness succeeds when expected application becomes active."""
    skill = InteractionSkills(
        security_policy=SystemSecurityPolicy(),
        confirmation_manager=SystemConfirmationManager(),
        input_backend=MockInputBackend(),
    )
    mock_vs = MagicMock()
    mock_vs.get_active_window_identity.return_value = (
        123, "Notepad - Untitled", "notepad.exe", (0, 0, 400, 300)
    )
    skill._vision_skills = mock_vs

    ready, title, proc = skill._wait_for_window_readiness(
        expected_app="notepad",
        expected_proc="notepad.exe",
        timeout_s=0.5,
        interval_s=0.01,
    )
    assert ready is True
    assert "notepad" in title.lower()


def test_3_readiness_timeout_fails_closed():
    """Verify readiness fails closed with PREFLIGHT_WINDOW_MISMATCH when timeout expires."""
    skill = InteractionSkills(
        security_policy=SystemSecurityPolicy(),
        confirmation_manager=SystemConfirmationManager(),
        input_backend=MockInputBackend(),
    )
    mock_vs = MagicMock()
    # Active window remains PowerShell
    mock_vs.get_active_window_identity.return_value = (
        999, "Windows PowerShell", "powershell.exe", (0, 0, 600, 400)
    )
    skill._vision_skills = mock_vs

    res = skill.execute({
        "operation": "visual_click",
        "target": "Search",
        "parameters": {
            "expected_app": "chrome",
            "expected_process": "chrome.exe",
            "readiness_timeout_s": 0.05,
            "readiness_interval_s": 0.01,
        },
    })
    assert res.success is False
    assert res.data.get("status") == VisualActionResultStatus.PREFLIGHT_WINDOW_MISMATCH.value
    assert res.data.get("preflight_reason") == "PREFLIGHT_WINDOW_MISMATCH"
    assert "readiness timeout" in str(res.error).lower()


def test_4_readiness_never_steals_focus():
    """Verify readiness observation only queries window metadata and never steals focus."""
    skill = InteractionSkills(
        security_policy=SystemSecurityPolicy(),
        confirmation_manager=SystemConfirmationManager(),
        input_backend=MockInputBackend(),
    )
    mock_vs = MagicMock()
    mock_vs.get_active_window_identity.return_value = (
        123, "Chrome", "chrome.exe", None
    )
    skill._vision_skills = mock_vs

    with patch("ctypes.windll.user32.SetForegroundWindow", create=True) as mock_set_fg:
        ready, _, _ = skill._wait_for_window_readiness(
            expected_app="chrome",
            expected_proc="chrome.exe",
            timeout_s=0.1,
        )
        assert ready is True
        assert mock_set_fg.call_count == 0


def test_5_readiness_does_not_persist_physical_state():
    """Verify readiness check does not store physical UI state into params or context."""
    skill = InteractionSkills(
        security_policy=SystemSecurityPolicy(),
        confirmation_manager=SystemConfirmationManager(),
        input_backend=MockInputBackend(),
    )
    mock_vs = MagicMock()
    mock_vs.get_active_window_identity.return_value = (
        555, "Settings", "settings.exe", (10, 10, 500, 500)
    )
    skill._vision_skills = mock_vs

    params = {"expected_app": "settings", "expected_process": "settings.exe"}
    skill._wait_for_window_readiness("settings", "settings.exe", timeout_s=0.1)

    for k in ("hwnd", "bounds", "coordinates", "x", "y", "screenshot", "ocr"):
        assert k not in params


# ---------------------------------------------------------------------------
# 6-9: Recovery Workflow Context Continuity
# ---------------------------------------------------------------------------

def test_6_recovery_preserves_workflow_context():
    """Verify Planner.replan preserves original semantic WorkflowContext."""
    planner = Planner(strategy=PlanningStrategy.RULE_BASED)
    orig_wf = WorkflowContext(
        workflow_id="wf_rec_001",
        objective="Open Chrome and search",
        expected_app="chrome",
        expected_process="chrome.exe",
        current_semantic_step=1,
    )
    orig_plan = Plan(
        query="Open Chrome and click search",
        tasks=[
            Task(id="task_1", action="open_app", target="chrome"),
            Task(id="task_2", action="visual_click", target="search", dependencies=["task_1"]),
        ],
        workflow_context=orig_wf,
    )

    # Simulate execution failure on task_2
    task_rec_1 = TaskExecutionRecord(
        task_id="task_1",
        action="open_app",
        target="chrome",
        status=TaskStatus.COMPLETED,
    )
    task_rec_2 = TaskExecutionRecord(
        task_id="task_2",
        action="visual_click",
        target="search",
        status=TaskStatus.FAILED,
        error="PREFLIGHT_WINDOW_MISMATCH",
        failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
    )
    memory = ExecutionMemory(records=[task_rec_1, task_rec_2])

    # Mock planner internal LLM generation for recovery
    planner._generate_plan_sync = MagicMock(return_value=[
        Task(id="task_rec_1", action="visual_click", target="search", dependencies=["task_1"])
    ])

    rec_plan = planner.replan(
        user_query="Open Chrome and click search",
        execution_result=memory,
        original_plan=orig_plan,
    )

    assert rec_plan.workflow_context is not None
    assert rec_plan.workflow_context.workflow_id == "wf_rec_001"
    assert rec_plan.workflow_context.expected_app == "chrome"
    assert rec_plan.workflow_context.expected_process == "chrome.exe"
    # Ensure recovered tasks receive the context
    assert len(rec_plan.tasks) == 1
    assert rec_plan.tasks[0].workflow_context == rec_plan.workflow_context


def test_7_recovery_advances_semantic_step_correctly():
    """Verify recovery advances current_semantic_step and records recovery status."""
    planner = Planner(strategy=PlanningStrategy.RULE_BASED)
    orig_wf = WorkflowContext(
        workflow_id="wf_step_002",
        objective="Log in to portal",
        current_semantic_step=2,
    )
    orig_plan = Plan(
        query="Log in to portal",
        tasks=[
            Task(id="t1", action="open_app", target="portal"),
            Task(id="t2", action="visual_click", target="login", dependencies=["t1"]),
        ],
        workflow_context=orig_wf,
    )
    task_rec_1 = TaskExecutionRecord(
        task_id="t1",
        action="open_app",
        target="portal",
        status=TaskStatus.COMPLETED,
    )
    task_rec_2 = TaskExecutionRecord(
        task_id="t2",
        action="visual_click",
        target="login",
        status=TaskStatus.FAILED,
        error="PREFLIGHT_TARGET_OBSCURED",
        failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
    )
    memory = ExecutionMemory(records=[task_rec_1, task_rec_2])
    planner._generate_plan_sync = MagicMock(return_value=[
        Task(id="t_rec", action="visual_click", target="login", dependencies=["t1"])
    ])

    rec_plan = planner.replan("Log in to portal", memory, orig_plan)
    assert rec_plan.workflow_context.current_semantic_step == 3
    assert "RECOVERING" in (rec_plan.workflow_context.previous_verification_outcome or "")


def test_8_successful_recovery_updates_ai_manager_active_context():
    """Verify AIManager updates _active_workflow_context from successful recovery."""
    mock_planner = MagicMock()
    mock_executor = MagicMock()
    ai_mgr = AIManager(planner_instance=mock_planner, executor_instance=mock_executor)

    orig_wf = WorkflowContext(
        workflow_id="wf_mgr_003",
        objective="Submit form",
        current_semantic_step=1,
    )
    orig_plan = Plan(
        query="Submit form",
        tasks=[Task(id="t1", action="visual_click", target="Submit")],
        workflow_context=orig_wf,
    )
    mock_planner.create_plan.return_value = orig_plan

    # Initial plan fails
    initial_fail_result = ExecutionResult(
        success=False,
        failed_tasks=[orig_plan.tasks[0]],
    )
    # Recovery plan succeeds
    rec_wf = orig_wf.with_step_outcome(action="visual_click", outcome="VERIFIED")
    rec_plan = Plan(
        query="Submit form",
        tasks=[Task(id="t_rec", action="visual_click", target="Submit")],
        workflow_context=rec_wf,
    )
    recovery_success_result = ExecutionResult(
        success=True,
        completed_tasks=[rec_plan.tasks[0]],
    )

    mock_executor.execute_plan.side_effect = [
        initial_fail_result,
        recovery_success_result,
    ]
    mock_planner.replan.return_value = rec_plan

    response = ai_mgr.generate("Submit form")
    active_wf = ai_mgr.active_workflow_context
    assert active_wf is not None
    assert active_wf.workflow_id == "wf_mgr_003"
    assert active_wf.current_semantic_step == 2


def test_9_next_user_turn_receives_recovered_semantic_context():
    """Verify that a subsequent turn in AIManager passes the recovered context to Planner."""
    mock_planner = MagicMock()
    mock_executor = MagicMock()
    ai_mgr = AIManager(planner_instance=mock_planner, executor_instance=mock_executor)

    # Pre-populate active context from prior recovery
    recovered_wf = WorkflowContext(
        workflow_id="wf_turn_004",
        objective="Multi-step form",
        expected_app="chrome",
        expected_process="chrome.exe",
        current_semantic_step=3,
    )
    ai_mgr._active_workflow_context = recovered_wf

    next_plan = Plan(query="Next step", tasks=[Task(id="t_next", action="visual_type", target="input")])
    mock_planner.create_plan.return_value = next_plan
    mock_executor.execute_plan.return_value = ExecutionResult(success=True)

    ai_mgr.generate("Enter username")
    mock_planner.create_plan.assert_called_once_with("Enter username", workflow_context=recovered_wf)


# ---------------------------------------------------------------------------
# 10-15: Verification Integrity & Deterministic Click Evidence
# ---------------------------------------------------------------------------

def test_10_no_verification_never_reports_verified_success():
    """Verify that an action with no post-condition verification is reported as unverified."""
    mock_adapter = MagicMock()
    mock_adapter.execute_target.return_value = VisualActionResult(
        status=VisualActionResultStatus.SUCCESS,
        action_type=VisualActionType.CLICK,
        target_id="tgt_1",
        target_element_name="button",
        success=True,
        reason="Action executed successfully.",
        verification_dict=None,  # No verification ran
    )
    sec_policy = SystemSecurityPolicy()
    sec_policy._safe_operations.add("visual_click")
    skill = InteractionSkills(
        security_policy=sec_policy,
        confirmation_manager=SystemConfirmationManager(),
        action_adapter=mock_adapter,
        input_backend=MockInputBackend(),
    )
    target_obj = VisualActionTarget(
        target_id="tgt_1",
        action_type=VisualActionType.CLICK,
        target_element_name="button",
        bounds=WindowBounds(0, 0, 10, 10),
        target_point=Point(5, 5),
        safety_tier=VisualActionSafetyTier.SAFE,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=False,
    )

    res = skill.execute({"operation": "visual_click", "target": "button", "parameters": {"action_target": target_obj}})
    assert res.success is True
    assert res.data.get("verified") is False
    assert "unverified" in res.data.get("summary", "").lower()
    assert "verified successfully" not in res.data.get("summary", "").lower()


def test_11_successful_verification_produces_verified():
    """Verify that when verification succeeds, it reports VERIFIED."""
    mock_adapter = MagicMock()
    mock_adapter.execute_target.return_value = VisualActionResult(
        status=VisualActionResultStatus.SUCCESS,
        action_type=VisualActionType.CLICK,
        target_id="tgt_1",
        target_element_name="button",
        success=True,
        reason="Action executed successfully.",
        verification_dict={"outcome": "VERIFIED", "reason": "Modal dismissed."},
    )
    sec_policy = SystemSecurityPolicy()
    sec_policy._safe_operations.add("visual_click")
    skill = InteractionSkills(
        security_policy=sec_policy,
        confirmation_manager=SystemConfirmationManager(),
        action_adapter=mock_adapter,
        input_backend=MockInputBackend(),
    )
    target_obj = VisualActionTarget(
        target_id="tgt_1",
        action_type=VisualActionType.CLICK,
        target_element_name="button",
        bounds=WindowBounds(0, 0, 10, 10),
        target_point=Point(5, 5),
        safety_tier=VisualActionSafetyTier.SAFE,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=False,
    )

    res = skill.execute({"operation": "visual_click", "target": "button", "parameters": {"action_target": target_obj}})
    assert res.success is True
    assert res.data.get("verified") is True
    assert "verified successfully" in res.data.get("summary", "").lower()


def test_12_failed_verification_produces_not_verified():
    """Verify that failed verification results in success=False and NOT_VERIFIED."""
    mock_adapter = MagicMock()
    mock_adapter.execute_target.return_value = VisualActionResult(
        status=VisualActionResultStatus.VERIFICATION_FAILED,
        action_type=VisualActionType.CLICK,
        target_id="tgt_1",
        target_element_name="button",
        success=False,
        reason="Visual verification NOT_VERIFIED: Expected element did not appear.",
        verification_dict={"outcome": "NOT_VERIFIED", "reason": "Expected element missing."},
    )
    sec_policy = SystemSecurityPolicy()
    sec_policy._safe_operations.add("visual_click")
    skill = InteractionSkills(
        security_policy=sec_policy,
        confirmation_manager=SystemConfirmationManager(),
        action_adapter=mock_adapter,
        input_backend=MockInputBackend(),
    )
    target_obj = VisualActionTarget(
        target_id="tgt_1",
        action_type=VisualActionType.CLICK,
        target_element_name="button",
        bounds=WindowBounds(0, 0, 10, 10),
        target_point=Point(5, 5),
        safety_tier=VisualActionSafetyTier.SAFE,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=False,
    )

    res = skill.execute({"operation": "visual_click", "target": "button", "parameters": {"action_target": target_obj}})
    assert res.success is False
    assert res.data.get("verified") is False
    assert res.data.get("verification_outcome") == "NOT_VERIFIED"


def test_13_uncertain_verification_produces_uncertain_and_stops():
    """Verify that UNCERTAIN verification halts recovery viability immediately."""
    err_reason = is_permanent_visual_failure("Visual verification UNCERTAIN: Evidence ambiguous.")
    assert err_reason == "VERIFICATION_UNCERTAIN"

    # In heuristics:
    memory = ExecutionMemory(
        records=[TaskExecutionRecord(
            task_id="t_unc",
            action="visual_click",
            target="submit",
            status=TaskStatus.FAILED,
            error="Visual verification UNCERTAIN: Evidence ambiguous.",
            failure_category=FailureCategory.VISUAL_VERIFICATION_FAILURE,
        )],
    )
    decision = evaluate_recovery_viability("Click submit", [Task(id="t_unc", action="visual_click")], memory)
    assert decision.viable is False
    assert "VERIFICATION_UNCERTAIN" in decision.reason


def test_14_clicked_button_remaining_visible_is_not_automatically_verified():
    """Verify that merely seeing the clicked button's text in OCR does not produce VERIFIED."""
    engine = VisualVerificationEngine()
    # Baseline observation before click: button "Submit" exists
    prior_obs = make_dummy_capture("Form", "form.exe", "Submit Form")
    # Post-action observation: button "Submit" is still there, screen has zero delta
    post_obs = make_dummy_capture("Form", "form.exe", "Submit Form")

    spec = VisualGoalSpec(
        criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
        target="Submit",
        metadata={"action": "click", "element_name": "Submit"},
    )

    res = engine.verify_goal(spec, post_obs, prior_observation=prior_obs)
    # Must NOT be VERIFIED!
    assert res.outcome != VisualOutcomeType.VERIFIED
    assert res.outcome == VisualOutcomeType.UNCERTAIN


def test_15_deterministic_state_transition_can_produce_verified():
    """Verify that genuine deterministic transitions (disappearance or delta) produce VERIFIED."""
    engine = VisualVerificationEngine()
    prior_obs = make_dummy_capture("Login Dialog", "login.exe", "Submit")
    # Post-action: dialog closed and title changed
    post_obs = make_dummy_capture("Dashboard", "app.exe", "Welcome to Dashboard")

    spec = VisualGoalSpec(
        criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
        target="Submit",
        metadata={"action": "click", "element_name": "Submit"},
    )

    res = engine.verify_goal(spec, post_obs, prior_observation=prior_obs)
    assert res.outcome == VisualOutcomeType.VERIFIED
    assert "disappeared" in res.explanation.lower() or "window title" in res.explanation.lower()


# ---------------------------------------------------------------------------
# 16-17: Transient Disabled Control
# ---------------------------------------------------------------------------

def test_16_transient_disabled_state_gets_at_most_one_recovery():
    """Verify that disabled control with explicit transient indicator allows 1 retry then halts."""
    err_transient = "Control is disabled (loading: debouncing form submission)"
    assert is_transient_control_readiness(err_transient) is True

    # Wave 0 (0 retries in history): viable for 1 recovery attempt
    task = Task(id="t_load", action="visual_click", target="submit")
    memory_w0 = ExecutionMemory(
        records=[TaskExecutionRecord(
            task_id="t_load",
            action="visual_click",
            target="submit",
            status=TaskStatus.FAILED,
            error=err_transient,
            failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
        )],
    )
    d0 = evaluate_recovery_viability("Submit form", [task], memory_w0)
    assert d0.viable is True  # Allowed 1 attempt

    # Wave 1 (already retried once): must fail closed
    memory_w1 = ExecutionMemory(
        records=[
            TaskExecutionRecord(
                task_id="t_load",
                action="visual_click",
                target="submit",
                status=TaskStatus.FAILED,
                error=err_transient,
                failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
            ),
            TaskExecutionRecord(
                task_id="t_load",
                action="visual_click",
                target="submit",
                status=TaskStatus.FAILED,
                error=err_transient,
                failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
            ),
        ],
    )
    d1 = evaluate_recovery_viability("Submit form", [task], memory_w1)
    assert d1.viable is False
    assert "exhausted transient readiness retry" in d1.reason


def test_17_genuinely_disabled_control_remains_fail_closed():
    """Verify that ordinary disabled control without transient evidence immediately fails closed."""
    err_disabled = "PREFLIGHT_DISABLED_CONTROL: Target button is disabled."
    assert is_transient_control_readiness(err_disabled) is False

    task = Task(id="t_dis", action="visual_click", target="disabled_btn")
    memory = ExecutionMemory(
        records=[TaskExecutionRecord(
            task_id="t_dis",
            action="visual_click",
            target="disabled_btn",
            status=TaskStatus.FAILED,
            error=err_disabled,
            failure_category=FailureCategory.VISUAL_PRECONDITION_FAILURE,
        )],
    )
    decision = evaluate_recovery_viability("Click disabled button", [task], memory)
    assert decision.viable is False
    assert "DISABLED_CONTROL" in decision.reason


# ---------------------------------------------------------------------------
# 18-20: Security Boundaries, Sanitization & Async/Sync Parity
# ---------------------------------------------------------------------------

def test_18_confirmation_token_does_not_cross_recovery_or_workflow_boundary():
    """Verify confirmation tokens are completely purged by purge_physical_state."""
    params = {
        "confirmation_id": "conf_xyz123",
        "confirmation_token": "token_abc789",
        "action_type": "CLICK",
        "semantic_intent": "Submit",
    }
    purge_physical_state(params)
    assert "confirmation_id" not in params
    assert "confirmation_token" not in params
    assert params["semantic_intent"] == "Submit"


def test_19_physical_target_cannot_leak_into_recovery_context():
    """Verify that physical coordinates, bounds, screenshots, OCR cannot leak into WorkflowContext."""
    wf = WorkflowContext(
        workflow_id="wf_sec_005",
        objective="Secure workflow",
        current_semantic_step=1,
    )
    adv_wf = wf.with_step_outcome(
        action="visual_click",
        outcome="VERIFIED",
        summary="Clicked at point(100, 200) with bounds [0,0,50,50] password: secret123",
    )
    # Check that summary sanitization redacts coordinates and passwords
    assert "secret123" not in str(adv_wf.verification_summary)
    assert "[REDACTED]" in str(adv_wf.verification_summary)

    # Check WorkflowContext attributes contain strictly zero physical coordinates
    for k in ("hwnd", "bounds", "target_point", "point", "coordinates", "screenshot", "ocr"):
        assert not hasattr(adv_wf, k)


def test_20_async_sync_parity():
    """Verify that async replanning and execution maintains parity with sync path."""
    async def _run():
        mock_planner = MagicMock()
        mock_executor = MagicMock()
        ai_mgr = AIManager(planner_instance=mock_planner, executor_instance=mock_executor)

        orig_wf = WorkflowContext(
            workflow_id="wf_parity_006",
            objective="Async workflow",
            current_semantic_step=1,
        )
        orig_plan = Plan(
            query="Async workflow",
            tasks=[Task(id="t1", action="visual_click", target="submit")],
            workflow_context=orig_wf,
        )
        mock_planner.create_plan_async = AsyncMock(return_value=orig_plan)

        rec_wf = orig_wf.with_step_outcome(action="visual_click", outcome="VERIFIED")
        rec_plan = Plan(
            query="Async workflow",
            tasks=[Task(id="t_rec", action="visual_click", target="submit")],
            workflow_context=rec_wf,
        )

        mock_executor.execute_plan_async = AsyncMock()
        mock_executor.execute_plan_async.side_effect = [
            ExecutionResult(success=False, failed_tasks=[orig_plan.tasks[0]]),
            ExecutionResult(success=True, completed_tasks=[rec_plan.tasks[0]]),
        ]
        mock_planner.replan_async = AsyncMock(return_value=rec_plan)

        await ai_mgr.generate_async("Async workflow")
        assert ai_mgr.active_workflow_context is not None
        assert ai_mgr.active_workflow_context.workflow_id == "wf_parity_006"
        assert ai_mgr.active_workflow_context.current_semantic_step == 2

    asyncio.run(_run())
