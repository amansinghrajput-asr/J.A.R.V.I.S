"""Tests for Phase 27.22: Visual Multi-Step Workflow Context & Semantic Continuity.

Verifies:
1. WorkflowContext creation and immutability.
2. Semantic context survives Task N -> Task N+1.
3. Physical target does NOT survive Task N -> Task N+1.
4. target_point cannot cross task boundary.
5. bounds cannot cross task boundary.
6. HWND cannot cross task boundary.
7. grounded_at cannot cross task boundary.
8. confirmation token cannot cross task boundary.
9. Task N+1 always performs fresh grounding.
10. Expected application semantic constraint works.
11. Unexpected application is rejected/replanned, never auto-refocused.
12. Expected modal continuation.
13. Unexpected modal recovery.
14. VERIFIED outcome propagates semantically.
15. VERIFICATION_UNCERTAIN stops safely.
16. Verification failure uses existing Phase 27.20 recovery.
17. Recovery discards physical state.
18. Composite visual actions are sequential.
19. Non-visual DAG behavior remains unchanged.
20. "Open login page" -> "Now enter username" conversational continuation.
21. Invalid workflow expires/terminates.
22. Password/input_text never appears in workflow context.
23. Confirmation token never appears in workflow context.
24. Coordinates never appear in workflow context.
25. Sensitive window context is sanitized.
26. Ordinary non-visual tasks remain backward compatible.

All tests strictly use MockInputBackend and enforce Win32InputBackend.invocation_count == 0.
"""

from __future__ import annotations

import json
import time
import pytest
from unittest.mock import MagicMock, patch

from tests.test_planner import MockPlanningProvider

from app.automation.input import MockInputBackend, Win32InputBackend
from app.automation.visual_action_adapter import (
    ActionPreflightValidator,
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)
from app.ai.manager import AIManager
from app.ai.planner.executor import Executor
from app.ai.planner.memory import (
    ExecutionMemory,
    FailureCategory,
    TaskExecutionRecord,
    sanitize_sensitive_data,
)
from app.ai.planner.models import (
    GROUNDED_PARAM_KEYS,
    ExecutionResult,
    Plan,
    PlanningStrategy,
    Task,
    TaskStatus,
    WorkflowContext,
    purge_physical_state,
)
from app.ai.planner.planner import Planner
from app.skills.system.interaction_skills import InteractionSkills
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.vision.models import (
    Point,
    VisualActionFeasibilityStatus,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualOutcomeType,
    VisualVerificationResult,
    WindowBounds,
)


@pytest.fixture(autouse=True)
def assert_zero_win32_invocations():
    """Ensure no real Win32 input calls happen during tests."""
    initial_count = Win32InputBackend.invocation_count
    yield
    assert Win32InputBackend.invocation_count == initial_count, "Real Win32 input must never be invoked in tests!"


@pytest.fixture
def mock_backend():
    return MockInputBackend()


@pytest.fixture
def confirmation_mgr():
    return SystemConfirmationManager()


@pytest.fixture
def security_policy():
    return SystemSecurityPolicy()


def _make_sample_target(name: str = "btn", x: int = 250, y: int = 350, hwnd: int = 12345) -> VisualActionTarget:
    return VisualActionTarget(
        target_id=f"tgt_{name}",
        action_type=VisualActionType.CLICK,
        target_element_name=name,
        target_point=Point(x, y),
        bounds=WindowBounds(x - 20, y - 10, x + 20, y + 10),
        safety_tier=VisualActionSafetyTier.SAFE,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=False,
        confidence=0.95,
        window_handle=hwnd,
        grounded_at=time.monotonic(),
    )


# ==============================================================================
# 1. WorkflowContext Model & Serialization Tests
# ==============================================================================

def test_01_workflow_context_creation_and_immutability():
    """WorkflowContext holds semantic fields and is strictly immutable."""
    wf = WorkflowContext(
        workflow_id="wf_101",
        objective="login to system",
        expected_app="chrome",
        expected_process="chrome.exe",
    )
    assert wf.workflow_id == "wf_101"
    assert wf.objective == "login to system"
    assert wf.expected_app == "chrome"
    assert wf.current_semantic_step == 1
    assert wf.is_valid is True

    # Immutability
    with pytest.raises(Exception):
        wf.current_semantic_step = 2  # type: ignore


def test_02_workflow_context_serialization_zero_physical_state():
    """WorkflowContext serialization contains zero physical visual state."""
    wf = WorkflowContext(
        workflow_id="wf_102",
        objective="submit form",
        expected_app="notepad",
        expected_process="notepad.exe",
        current_semantic_step=2,
        previous_action="visual_click",
        previous_verification_outcome="VERIFIED",
        verification_summary="Button clicked successfully",
    )
    data = wf.to_dict()
    for forbidden in ("hwnd", "HWND", "target_point", "point", "bounds", "target", "token", "password", "input_text"):
        assert forbidden not in data, f"Physical/secret key '{forbidden}' leaked into WorkflowContext serialization!"

    reconstituted = WorkflowContext.from_dict(data)
    assert reconstituted.workflow_id == wf.workflow_id
    assert reconstituted.objective == wf.objective
    assert reconstituted.previous_verification_outcome == "VERIFIED"


def test_03_workflow_context_ttl_expiration():
    """WorkflowContext expires after ttl_seconds."""
    wf = WorkflowContext(
        workflow_id="wf_103",
        objective="test ttl",
        created_at=time.monotonic() - 65.0,
        ttl_seconds=60.0,
    )
    assert wf.is_expired() is True

    wf_fresh = WorkflowContext(
        workflow_id="wf_104",
        objective="test fresh",
        created_at=time.monotonic(),
        ttl_seconds=60.0,
    )
    assert wf_fresh.is_expired() is False


# ==============================================================================
# 2. Task Boundary & Physical State Purge Tests
# ==============================================================================

def test_04_semantic_context_survives_task_boundary(mock_backend):
    """Semantic context survives Task 1 -> Task 2 in Executor DAG execution."""
    exec_inst = Executor()

    wf = WorkflowContext(
        workflow_id="wf_201",
        objective="multi-step test",
        expected_app="calc",
        expected_process="calc.exe",
    )
    t1 = Task(id="task_1", action="open_app", target="calc")
    t2 = Task(id="task_2", action="visual_click", target="One Button", dependencies=["task_1"])

    plan = Plan(query="open calc and click one", tasks=[t1, t2], workflow_context=wf)

    # Register handlers
    exec_inst.register_handler("open_app", lambda t: "Opened calc")
    exec_inst.register_handler("visual_click", lambda t: "Clicked one")

    res = exec_inst.execute_plan(plan)
    assert res.success is True
    assert t2.workflow_context is not None
    assert t2.workflow_context.previous_action == "open_app"
    assert t2.workflow_context.current_semantic_step >= 2


def test_05_physical_target_does_not_survive_task_boundary(mock_backend):
    """Physical VisualActionTarget cannot cross task boundaries."""
    exec_inst = Executor()

    target1 = _make_sample_target("OK Button", x=111, y=222)
    t1 = Task(
        id="t1",
        action="visual_click",
        target="OK Button",
        parameters={"action_target": target1, "point": (111, 222)},
    )
    t2 = Task(
        id="t2",
        action="visual_click",
        target="Next Button",
        dependencies=["t1"],
        parameters={"action_target": target1, "point": (111, 222)},
    )
    plan = Plan(query="click ok then next", tasks=[t1, t2])

    captured_params = []

    def mock_handler(task: Task):
        captured_params.append(dict(task.parameters))
        return "OK"

    exec_inst.register_handler("visual_click", mock_handler)
    res = exec_inst.execute_plan(plan)
    assert res.success is True

    # Both tasks must have had physical parameters purged before handler execution
    for p in captured_params:
        assert "action_target" not in p
        assert "point" not in p
        assert "target_point" not in p


def test_06_target_point_cannot_cross_task_boundary():
    """target_point parameter is purged from parameters before execution."""
    params = {"target_point": Point(500, 600), "point": (500, 600), "text": "hello"}
    purge_physical_state(params)
    assert "target_point" not in params
    assert "point" not in params
    assert "text" in params


def test_07_bounds_cannot_cross_task_boundary():
    """WindowBounds and bounds parameter are purged from parameters."""
    params = {"bounds": WindowBounds(10, 20, 30, 40), "valid_param": 123}
    purge_physical_state(params)
    assert "bounds" not in params
    assert params["valid_param"] == 123


def test_08_hwnd_cannot_cross_task_boundary():
    """HWND and window_handle are purged from parameters."""
    params = {"hwnd": 0x1234, "window_handle": 0x1234, "app": "test"}
    purge_physical_state(params)
    assert "hwnd" not in params
    assert "window_handle" not in params
    assert params["app"] == "test"


def test_09_grounded_at_cannot_cross_task_boundary():
    """grounded_at timestamp is purged from parameters."""
    params = {"grounded_at": time.monotonic(), "step": "next"}
    purge_physical_state(params)
    assert "grounded_at" not in params
    assert params["step"] == "next"


def test_10_confirmation_token_cannot_cross_task_boundary():
    """confirmation_token and confirmation_id are purged from parameters."""
    params = {"confirmation_token": "token_abc123", "confirmation_id": "conf_001", "keep": True}
    purge_physical_state(params)
    assert "confirmation_token" not in params
    assert "confirmation_id" not in params
    assert params["keep"] is True


# ==============================================================================
# 3. Fresh Grounding & Application Continuity Tests
# ==============================================================================

def test_11_task_n_plus_1_always_performs_fresh_grounding(mock_backend):
    """InteractionSkills always invokes fresh grounding when VisionSkills is present."""
    mock_vs = MagicMock()
    mock_vs.execute.return_value = {
        "target": _make_sample_target("Fresh Button", x=300, y=400),
        "status": "FEASIBLE",
        "summary": "Target found",
    }
    skill = InteractionSkills(input_backend=mock_backend, vision_skills=mock_vs)

    stale_target = _make_sample_target("Stale Button", x=100, y=100)
    cmd = {
        "operation": "visual_click",
        "target": "Submit Button",
        "parameters": {"target": stale_target},  # Attempted stale target injection
    }
    res = skill.execute(cmd)
    assert res.success is True
    # Vision skills must have been called to perform fresh grounding
    assert mock_vs.execute.called
    call_args = mock_vs.execute.call_args[0][0]
    assert call_args["operation"] == "ground_visual_action"
    assert call_args["target"] == "Submit Button"


def test_12_expected_application_semantic_constraint(mock_backend):
    """Application continuity constraint succeeds when window process matches."""
    mock_vs = MagicMock()
    mock_vs.execute.return_value = {
        "target": _make_sample_target("Login Button", x=300, y=400),
        "status": "FEASIBLE",
        "window_info": {"process_name": "chrome.exe", "window_title": "Google Accounts"},
    }
    skill = InteractionSkills(input_backend=mock_backend, vision_skills=mock_vs)

    cmd = {
        "operation": "visual_click",
        "target": "Login Button",
        "parameters": {"expected_app": "chrome", "expected_process": "chrome.exe"},
    }
    res = skill.execute(cmd)
    assert res.success is True


def test_13_unexpected_application_rejected_no_auto_refocus(mock_backend):
    """When active window mismatches expected application, rejects with PREFLIGHT_WINDOW_MISMATCH without auto-refocus."""
    mock_vs = MagicMock()
    mock_vs.execute.return_value = {
        "target": _make_sample_target("Login Button", x=300, y=400),
        "status": "FEASIBLE",
        "window_info": {"process_name": "slack.exe", "window_title": "Slack Notifications"},
    }
    skill = InteractionSkills(input_backend=mock_backend, vision_skills=mock_vs)

    cmd = {
        "operation": "visual_click",
        "target": "Login Button",
        "parameters": {"expected_app": "chrome", "expected_process": "chrome.exe"},
    }
    res = skill.execute(cmd)
    assert res.success is False
    assert res.data.get("status") == "PREFLIGHT_WINDOW_MISMATCH"
    assert "mismatch" in str(res.error).lower()
    # Confirms no click dispatch occurred to auto-refocus
    assert mock_backend.event_count("click") == 0


def test_14_expected_modal_continuation(mock_backend):
    """Modal continuation progresses when window owner matches expected application."""
    mock_vs = MagicMock()
    mock_vs.execute.return_value = {
        "target": _make_sample_target("Save As Button", x=300, y=400),
        "status": "FEASIBLE",
        "window_info": {"process_name": "chrome.exe", "window_title": "Save File - Google Chrome"},
    }
    skill = InteractionSkills(input_backend=mock_backend, vision_skills=mock_vs)

    cmd = {
        "operation": "visual_click",
        "target": "Save As Button",
        "parameters": {"expected_app": "chrome"},
    }
    res = skill.execute(cmd)
    assert res.success is True


def test_15_unexpected_modal_recovery(mock_backend):
    """Unexpected modal failure produces visual recovery recommendation."""
    mock_prov = MockPlanningProvider(
        json.dumps({
            "tasks": [
                {"id": "t_modal", "action": "visual_dismiss_modal", "target": "popup"}
            ]
        })
    )
    planner = Planner(strategy=PlanningStrategy.LLM, provider_instance=mock_prov)
    failed_task = Task(
        id="t_click",
        action="visual_click",
        target="submit button",
        parameters={},
    )
    res = ExecutionResult(
        success=False,
        failed_tasks=[failed_task],
        output="PREFLIGHT_MODAL_CHANGED: Obstructed by unexpected popup modal",
    )
    memory = ExecutionMemory.from_execution_result(res, wave=1)
    orig_plan = Plan(query="click submit", tasks=[failed_task])

    recovery_plan = planner.replan("click submit", memory, orig_plan)
    assert recovery_plan is not None
    assert any(t.action == "visual_dismiss_modal" for t in recovery_plan.tasks)


# ==============================================================================
# 4. Verification & Recovery Propagation Tests
# ==============================================================================

def test_16_verified_outcome_propagates_semantically(mock_backend):
    """Successful verification outcome updates workflow context semantically."""
    exec_inst = Executor()

    wf = WorkflowContext(workflow_id="wf_v1", objective="verify flow")
    t1 = Task(id="t1", action="open_app", target="chrome")
    t2 = Task(id="t2", action="visual_click", target="Search", dependencies=["t1"])

    plan = Plan(query="open chrome then search", tasks=[t1, t2], workflow_context=wf)

    exec_inst.register_handler("open_app", lambda t: "Opened chrome")
    exec_inst.register_handler("visual_click", lambda t: "Clicked search")

    res = exec_inst.execute_plan(plan)
    assert res.success is True
    assert plan.workflow_context is not None
    assert plan.workflow_context.previous_verification_outcome == "VERIFIED"
    assert plan.workflow_context.expected_app == "chrome"


def test_17_verification_uncertain_stops_safely(mock_backend):
    """VERIFICATION_UNCERTAIN stops execution without blind retries."""
    exec_inst = Executor()

    class MockUncertainResult:
        success = False
        error = "VERIFICATION_UNCERTAIN: Visual delta ambiguous"
        data = {"status": "VERIFICATION_UNCERTAIN", "reason": "Visual delta ambiguous"}

    t1 = Task(id="t1", action="visual_click", target="Proceed")
    plan = Plan(query="click proceed", tasks=[t1])

    call_count = 0
    def handler(t):
        nonlocal call_count
        call_count += 1
        return MockUncertainResult()

    exec_inst.register_handler("visual_click", handler)
    res = exec_inst.execute_plan(plan)
    assert res.success is False
    assert call_count == 1  # Exactly 1 call: no blind retry loop


def test_18_verification_failure_uses_phase27_20_recovery(mock_backend):
    """Verification failure allows single fresh recovery attempt."""
    mock_prov = MockPlanningProvider(
        json.dumps({
            "tasks": [
                {"id": "t_retry", "action": "visual_click", "target": "submit"}
            ]
        })
    )
    planner = Planner(strategy=PlanningStrategy.LLM, provider_instance=mock_prov)
    failed_task = Task(id="t_sub", action="visual_click", target="submit")
    res = ExecutionResult(
        success=False,
        failed_tasks=[failed_task],
        output="VERIFICATION_FAILED: Expected dashboard did not appear",
    )
    memory = ExecutionMemory.from_execution_result(res, wave=1)
    orig_plan = Plan(query="click submit", tasks=[failed_task])

    recovery_plan = planner.replan("click submit", memory, orig_plan)
    assert recovery_plan is not None
    assert len(recovery_plan.tasks) == 1
    assert recovery_plan.tasks[0].action == "visual_click"
    # Physical state must be clean
    assert "target_point" not in recovery_plan.tasks[0].parameters


def test_19_recovery_discards_physical_state(mock_backend):
    """Recovery tasks discard physical state keys from parameters."""
    mock_prov = MockPlanningProvider(
        json.dumps({
            "tasks": [
                {
                    "id": "t_retry",
                    "action": "visual_click",
                    "target": "save",
                    "parameters": {"action_target": "bad", "point": [10, 20], "hwnd": 999},
                }
            ]
        })
    )
    planner = Planner(strategy=PlanningStrategy.LLM, provider_instance=mock_prov)
    stale_t = Task(
        id="t_stale",
        action="visual_click",
        target="save",
        parameters={"action_target": "bad", "point": (10, 20), "hwnd": 999},
    )
    res = ExecutionResult(
        success=False,
        failed_tasks=[stale_t],
        output="TIMEOUT",
    )
    memory = ExecutionMemory.from_execution_result(res, wave=1)
    orig_plan = Plan(query="click save", tasks=[stale_t])

    rec_plan = planner.replan("click save", memory, orig_plan)
    assert rec_plan is not None
    assert len(rec_plan.tasks) == 1
    for k in GROUNDED_PARAM_KEYS:
        assert k not in rec_plan.tasks[0].parameters


# ==============================================================================
# 5. Composite Visual Task Dependencies & DAG
# ==============================================================================

def test_20_composite_visual_actions_are_sequential():
    """Conjunction decomposition chains sequential dependencies for visual tasks."""
    planner = Planner(strategy=PlanningStrategy.RULE_BASED)
    plan = planner.create_plan("open chrome and click login")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].action == "open_app"
    assert plan.tasks[1].action == "visual_click"
    # Task 2 MUST depend on Task 1
    assert plan.tasks[1].dependencies == [plan.tasks[0].id]


def test_21_non_visual_dag_behavior_unchanged():
    """Non-visual independent tasks created directly in a plan maintain independent DAG execution."""
    exec_inst = Executor()

    t1 = Task(id="t1", action="calculate", target="2+2")
    t2 = Task(id="t2", action="calculate", target="3+3")
    plan = Plan(query="calc both", tasks=[t1, t2])  # Both have dependencies=[]

    exec_inst.register_handler("calculate", lambda t: "4")
    res = exec_inst.execute_plan(plan)
    assert res.success is True
    assert len(res.completed_tasks) == 2


# ==============================================================================
# 6. Conversational Continuity & Anaphora
# ==============================================================================

def test_22_conversational_continuation_now_enter_username():
    """Planner strips conversational prefixes like 'now' to match visual actions."""
    planner = Planner(strategy=PlanningStrategy.RULE_BASED)
    wf = WorkflowContext(
        workflow_id="wf_conv_1",
        objective="login",
        expected_app="chrome",
        expected_process="chrome.exe",
    )
    plan = planner.create_plan("now enter my username", workflow_context=wf)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].action == "visual_type"
    assert "username" in plan.tasks[0].target
    assert plan.workflow_context == wf


def test_23_invalid_workflow_expires_terminates():
    """Expired or invalidated workflow context is purged and not attached to new plans."""
    planner = Planner(strategy=PlanningStrategy.RULE_BASED)
    expired_wf = WorkflowContext(
        workflow_id="wf_exp",
        objective="old flow",
        created_at=time.monotonic() - 100.0,
        ttl_seconds=60.0,
    )
    plan = planner.create_plan("open chrome", workflow_context=expired_wf)
    assert plan.workflow_context is not None
    assert plan.workflow_context.workflow_id != "wf_exp"  # Fresh workflow generated


# ==============================================================================
# 7. Privacy & Security Invariants
# ==============================================================================

def test_24_password_input_text_never_appears_in_workflow_context():
    """Passwords or input text typed during action are never stored in WorkflowContext."""
    wf = WorkflowContext(workflow_id="wf_sec", objective="login")
    updated = wf.with_step_outcome(
        action="visual_type",
        outcome="VERIFIED",
        summary="Entered secret password123 successfully",
    )
    assert "password123" not in str(updated.verification_summary)
    d = updated.to_dict()
    assert "password" not in str(d)
    assert "password123" not in str(d)


def test_25_coordinates_never_appear_in_workflow_context():
    """Coordinates are redacted from WorkflowContext verification summaries."""
    wf = WorkflowContext(workflow_id="wf_coord", objective="test")
    updated = wf.with_step_outcome(
        action="visual_click",
        outcome="VERIFIED",
        summary="Clicked element at (542, 891)",
    )
    assert "(542, 891)" not in updated.verification_summary
    assert "[COORDINATES_REDACTED]" in updated.verification_summary


def test_26_backward_compatibility_non_visual_tasks(mock_backend):
    """Ordinary non-visual tasks remain fully backward compatible."""
    exec_inst = Executor()

    t = Task(id="calc_1", action="calculate", target="10*10")
    plan = Plan(query="calculate 10*10", tasks=[t])

    exec_inst.register_handler("calculate", lambda t: "100")
    res = exec_inst.execute_plan(plan)
    assert res.success is True
    assert res.completed_tasks[0].id == "calc_1"
    assert "100" in res.output
