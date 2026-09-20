"""Tests for Phase 27.20: Visual Action Recovery and Adaptive Replanning.

Covers invariants A through U:
A. Stale target is discarded during recovery
B. Old coordinates are never reused
C. Old window handle is never reused
D. Old grounded_at is never reused
E. Old confirmation token is never reused
F. Visual max retry = 1
G. Visual max replans <= 2
H. SECURITY_BLOCKED stops recovery
I. SENSITIVE_PROTECTED stops recovery
J. DISABLED_CONTROL stops recovery
K. CONFIRMATION_REQUIRED stops and requires fresh confirmation
L. VERIFICATION_UNCERTAIN stops
M. VERIFICATION_FAILED allows only one fresh recovery attempt
N. PREFLIGHT_WINDOW_MISMATCH causes fresh grounding
O. PREFLIGHT_STALE_COORDINATES causes fresh grounding
P. PREFLIGHT_MODAL_CHANGED allows semantic modal-resolution recovery
Q. Semantic recovery task contains no raw coordinates
R. Visual recovery memory summary contains no input_text
S. Password/secret payloads never enter recovery prompts
T. Timeout does not blindly repeat side effects
U. Existing non-visual recovery behavior remains unchanged

All tests use MockInputBackend and verify Win32InputBackend.invocation_count == 0.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

from app.automation.input import MockInputBackend, Win32InputBackend
from app.automation.visual_action_adapter import (
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)
from app.ai.manager import AIManager, EXECUTABLE_VISUAL_ACTIONS
from app.ai.planner.heuristics import (
    evaluate_recovery_viability,
    is_permanent_visual_failure,
    VISUAL_ACTIONS,
    PERMANENT_VISUAL_FAILURES,
)
from app.ai.planner.memory import (
    ExecutionMemory,
    FailureCategory,
    TaskExecutionRecord,
)
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.models import ExecutionResult, Plan, PlanningStrategy, Task, TaskStatus
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


def _make_stale_target(name: str = "submit_btn", hwnd: int = 12345, x: int = 100, y: int = 200) -> VisualActionTarget:
    """Create a simulated stale VisualActionTarget."""
    return VisualActionTarget(
        target_id="target_stale_01",
        action_type=VisualActionType.CLICK,
        target_element_name=name,
        target_point=Point(x, y),
        bounds=WindowBounds(x - 20, y - 10, x + 20, y + 10),
        safety_tier=VisualActionSafetyTier.SAFE,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=False,
        confidence=0.95,
        window_handle=hwnd,
        grounded_at=time.monotonic() - 10.0,  # 10s old
    )


def _make_fresh_target(
    name: str = "submit_btn",
    hwnd: int = 54321,
    x: int = 300,
    y: int = 400,
    requires_confirmation: bool = False,
    safety_tier: VisualActionSafetyTier = VisualActionSafetyTier.SAFE,
) -> VisualActionTarget:
    """Create a simulated newly grounded VisualActionTarget."""
    return VisualActionTarget(
        target_id="target_fresh_99",
        action_type=VisualActionType.CLICK,
        target_element_name=name,
        target_point=Point(x, y),
        bounds=WindowBounds(x - 25, y - 12, x + 25, y + 12),
        safety_tier=safety_tier,
        feasibility=VisualActionFeasibilityStatus.FEASIBLE,
        requires_confirmation=requires_confirmation,
        confidence=0.98,
        window_handle=hwnd,
        grounded_at=time.monotonic(),
    )


# ==============================================================================
# Requirement A, B, C, D: Stale Target, Coordinates, Window Handle, Grounded_At Discarded
# ==============================================================================

def test_a_b_c_d_stale_target_discarded_and_fresh_grounding(mock_backend, confirmation_mgr, security_policy):
    """A, B, C, D: Stale target, coordinates, HWND, and grounded_at are discarded in recovery wave."""
    stale_target = _make_stale_target(name="save_button", hwnd=1111, x=50, y=60)
    fresh_target = _make_fresh_target(name="save_button", hwnd=2222, x=250, y=360)

    mock_vision = MagicMock()
    mock_vision.execute.return_value = MagicMock(data={"target": fresh_target, "status": "FEASIBLE"})

    skill = InteractionSkills(
        input_backend=mock_backend,
        confirmation_manager=confirmation_mgr,
        security_policy=security_policy,
        vision_skills=mock_vision,
    )

    # Simulate wave 2 recovery execution passing stale target in parameters
    cmd = {
        "operation": "visual_click",
        "target": "save_button",
        "parameters": {
            "is_recovery": True,
            "wave": 2,
            "target": stale_target,
            "target_point": {"x": 50, "y": 60},
            "window_handle": 1111,
        },
    }

    res = skill.execute(cmd)

    # Fresh grounding must have been called with force_fresh=True
    assert mock_vision.execute.called
    call_args = mock_vision.execute.call_args[0][0]
    assert call_args["parameters"].get("force_fresh") is True
    assert call_args["parameters"].get("bypass_cache") is True

    # The executed target in data must be the fresh one, not the stale one
    assert res.data["target_point"]["x"] == 250
    assert res.data["target_point"]["y"] == 360
    assert res.data["target_id"] == "target_fresh_99"


# ==============================================================================
# Requirement E: Old Confirmation Token Is Never Reused
# ==============================================================================

def test_e_old_confirmation_token_invalidated_across_recovery_boundary(mock_backend, confirmation_mgr, security_policy):
    """E: Old confirmation token cannot be carried over or reused in recovery wave."""
    # Issue a confirmation token for wave 1
    token = confirmation_mgr.request_confirmation(
        operation="visual_click",
        target="delete_account",
        parameters={"action": "delete"},
        description="Destructive action",
    )
    # Approve it
    confirmation_mgr.resolve_confirmation(token, approved=True)

    fresh_target = _make_fresh_target(
        name="delete_account",
        hwnd=2222,
        x=200,
        y=300,
        requires_confirmation=True,
        safety_tier=VisualActionSafetyTier.DESTRUCTIVE,
    )

    mock_vision = MagicMock()
    mock_vision.execute.return_value = MagicMock(data={"target": fresh_target, "status": "FEASIBLE"})

    skill = InteractionSkills(
        input_backend=mock_backend,
        confirmation_manager=confirmation_mgr,
        security_policy=security_policy,
        vision_skills=mock_vision,
    )

    # In recovery wave 2, passing the old approved token must be rejected/ignored
    # and a new confirmation required error must be raised
    cmd = {
        "operation": "visual_click",
        "target": "delete_account",
        "parameters": {
            "is_recovery": True,
            "wave": 2,
            "confirmation_id": token,
            "action": "delete",
        },
    }

    with pytest.raises(ConfirmationRequiredError) as exc_info:
        skill.execute(cmd)

    # The newly requested confirmation ID must differ from the old token
    assert exc_info.value.confirmation_id != token


# ==============================================================================
# Requirement F: Visual Max Retry = 1
# ==============================================================================

def test_f_visual_max_retry_exhaustion():
    """F: Visual tasks are limited to 1 recovery attempt (total attempts limit = 2)."""
    task = Task(id="t_click", action="visual_click", target="login")
    plan = Plan(query="click login", tasks=[task])

    # Wave 1 failure
    mem = ExecutionMemory()
    res1 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="PREFLIGHT_STALE_COORDINATES",
    )
    mem = mem.record_execution(res1, wave=1)

    # Attempt 1: 1 recovery attempt should be viable
    decision = evaluate_recovery_viability("click login", plan, mem)
    assert decision.viable is True

    # Wave 2 failure (this is the 1 recovery attempt)
    res2 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="PREFLIGHT_STALE_COORDINATES",
    )
    mem = mem.record_execution(res2, wave=2)

    # Attempt 2: Visual max retry exhausted, must be non-viable
    decision2 = evaluate_recovery_viability("click login", plan, mem)
    assert decision2.viable is False
    assert "exceeded maximum retry limit" in decision2.reason


# ==============================================================================
# Requirement G: Visual Max Replans <= 2
# ==============================================================================

def test_g_visual_max_replans_clamped():
    """G: Visual action plans have max_replans clamped to <= 2 in AIManager."""
    ai = AIManager(max_replans=5)  # user configured 5

    task = Task(id="t1", action="visual_click", target="btn")
    plan = Plan(query="click btn", tasks=[task])

    exec_res = ExecutionResult(success=False, failed_tasks=[task])

    # attempt 0: allowed (< 2)
    assert ai._should_replan(exec_res, attempt=0, plan=plan) is True
    # attempt 1: allowed (< 2)
    assert ai._should_replan(exec_res, attempt=1, plan=plan) is True
    # attempt 2: halted (clamped to 2, so attempt >= 2 returns False)
    assert ai._should_replan(exec_res, attempt=2, plan=plan) is False


# ==============================================================================
# Requirements H, I, J, K, L: Permanent Non-Recoverable Failures Halt Recovery
# ==============================================================================

@pytest.mark.parametrize("error_text,status_code", [
    ("SECURITY_BLOCKED by security policy", "SECURITY_BLOCKED"),
    ("Target is SENSITIVE_PROTECTED and strictly prohibited", "SENSITIVE_PROTECTED"),
    ("DISABLED_CONTROL: button is currently disabled", "DISABLED_CONTROL"),
    ("Action requires explicit confirmation. Token: 'conf_123'", "CONFIRMATION_REQUIRED"),
    ("VERIFICATION_UNCERTAIN: could not confirm screen update", "VERIFICATION_UNCERTAIN"),
])
def test_h_i_j_k_l_permanent_visual_failures_halt_recovery(error_text, status_code):
    """H, I, J, K, L: Permanent failures halt recovery immediately (0 retries)."""
    task = Task(id="t_vis", action="visual_click", target="button")
    plan = Plan(query="click button", tasks=[task])

    mem = ExecutionMemory()
    res = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output=error_text,
    )
    mem = mem.record_execution(res, wave=1)

    decision = evaluate_recovery_viability("click button", plan, mem)
    assert decision.viable is False
    assert status_code in decision.reason
    assert "permanent non-recoverable status" in decision.reason


# ==============================================================================
# Requirement M: VERIFICATION_FAILED Allows Exactly One Fresh Recovery Attempt
# ==============================================================================

def test_m_verification_failed_allows_one_retry():
    """M: VERIFICATION_FAILED allows exactly one fresh recovery attempt."""
    task = Task(id="t1", action="visual_click", target="save")
    plan = Plan(query="click save", tasks=[task])

    mem = ExecutionMemory()
    res1 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="VERIFICATION_FAILED: expected visual delta not detected",
    )
    mem = mem.record_execution(res1, wave=1)

    decision1 = evaluate_recovery_viability("click save", plan, mem)
    assert decision1.viable is True

    # Second failure
    res2 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="VERIFICATION_FAILED: expected visual delta not detected",
    )
    mem = mem.record_execution(res2, wave=2)

    decision2 = evaluate_recovery_viability("click save", plan, mem)
    assert decision2.viable is False


# ==============================================================================
# Requirements N, O: PREFLIGHT TOCTOU Failures Cause Fresh Grounding
# ==============================================================================

@pytest.mark.parametrize("toctou_err", [
    "PREFLIGHT_WINDOW_MISMATCH: HWND changed",
    "PREFLIGHT_STALE_COORDINATES: target moved",
])
def test_n_o_preflight_toctou_allows_single_recovery_and_fresh_grounding(toctou_err):
    """N, O: Window mismatch and stale coordinates permit recovery via fresh grounding."""
    task = Task(id="t_click", action="visual_click", target="ok_btn")
    plan = Plan(query="click ok", tasks=[task])

    mem = ExecutionMemory()
    res = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output=toctou_err,
    )
    mem = mem.record_execution(res, wave=1)

    decision = evaluate_recovery_viability("click ok", plan, mem)
    assert decision.viable is True


# ==============================================================================
# Requirement P: PREFLIGHT_MODAL_CHANGED Allows Semantic Modal-Resolution Recovery
# ==============================================================================

def test_p_modal_changed_recovery_validation():
    """P: Modal failure requires 'visual_dismiss_modal' prior to retrying action."""
    planner = Planner()
    task = Task(id="t_submit", action="visual_click", target="submit")

    mem = ExecutionMemory()
    res = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="PREFLIGHT_MODAL_CHANGED: active modal dialog appeared",
    )
    mem = mem.record_execution(res, wave=1)

    # 1. Proposing a blind duplicate click without modal dismissal must be rejected
    blind_retry = [Task(id="t_submit_retry", action="visual_click", target="submit")]
    is_valid, errors = planner._validate_recovery_tasks(blind_retry, mem)
    assert is_valid is False
    assert any("blocked by modal" in err for err in errors)

    # 2. Proposing visual_dismiss_modal followed by visual_click is valid
    modal_recovery = [
        Task(id="t_dismiss", action="visual_dismiss_modal", target="modal"),
        Task(id="t_submit_retry", action="visual_click", target="submit", dependencies=["t_dismiss"]),
    ]
    is_valid2, errors2 = planner._validate_recovery_tasks(modal_recovery, mem)
    assert is_valid2 is True
    assert len(errors2) == 0


# ==============================================================================
# Requirement Q: Semantic Recovery Task Contains No Raw Coordinates
# ==============================================================================

def test_q_recovery_task_rejects_raw_coordinates():
    """Q: Semantic recovery task cannot contain raw coordinates in target or parameters."""
    planner = Planner()
    mem = ExecutionMemory()

    # Raw coordinates in target
    raw_coord_task = Task(id="t_raw", action="visual_click", target="(350, 420)")
    is_valid, errors = planner._validate_recovery_tasks([raw_coord_task], mem)
    assert is_valid is False
    assert any("raw coordinates" in err for err in errors)

    # Coordinates in parameters
    raw_param_task = Task(id="t_param", action="visual_click", target="button", parameters={"x": 350, "y": 420})
    is_valid, errors = planner._validate_recovery_tasks([raw_param_task], mem)
    assert is_valid is False
    assert any("coordinate parameters" in err for err in errors)


def test_q_stale_grounded_parameters_are_purged():
    """Q: Pre-existing grounded parameters are stripped automatically."""
    planner = Planner()
    task = Task(
        id="t_rec",
        action="visual_click",
        target="login_button",
        parameters={
            "action_target": {"point": (10, 20)},
            "target_point": {"x": 10, "y": 20},
            "bounds": {"left": 0, "top": 0},
            "window_handle": 9999,
            "grounded_at": 12345.6,
            "confirmation_token": "token_abc",
            "semantic_param": "keep_me",
        },
    )
    planner._sanitize_recovery_task(task)
    assert "target_point" not in task.parameters
    assert "action_target" not in task.parameters
    assert "bounds" not in task.parameters
    assert "window_handle" not in task.parameters
    assert "grounded_at" not in task.parameters
    assert "confirmation_token" not in task.parameters
    assert task.parameters["semantic_param"] == "keep_me"


# ==============================================================================
# Requirements R, S: Strict Privacy in Planner Memory & Prompts
# ==============================================================================

def test_r_s_privacy_in_memory_summary_and_prompts():
    """R, S: Input_text, passwords, and secrets never leak in memory summaries or prompts."""
    mem = ExecutionMemory()
    task = Task(
        id="t_type",
        action="visual_type",
        target="password_box",
        parameters={"input_text": "SuperSecretPassword123!", "token": "secret_token_xyz"},
    )
    res = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="PREFLIGHT_STALE_COORDINATES with password: 'SuperSecretPassword123!'",
    )
    mem = mem.record_execution(res, wave=1)

    summary = MemorySummaryBuilder.build_planner_summary(mem)

    # Strict privacy verification
    assert "SuperSecretPassword123!" not in summary
    assert "secret_token_xyz" not in summary
    assert "input_present=True" in summary
    assert "input_redacted=True" in summary
    assert "PREFLIGHT_STALE_COORDINATES" in summary


# ==============================================================================
# Requirement T: Timeout Does Not Blindly Repeat Side Effects
# ==============================================================================

def test_t_timeout_classified_and_bounded():
    """T: Visual timeout failure is classified and bounded to at most 1 recovery attempt."""
    task = Task(id="t_sub", action="visual_click", target="pay_now")
    plan = Plan(query="click pay now", tasks=[task])

    mem = ExecutionMemory()
    res1 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="TIMEOUT: Visual action timed out after 5.0 seconds",
    )
    mem = mem.record_execution(res1, wave=1)

    # 1 attempt is allowed
    decision1 = evaluate_recovery_viability("click pay now", plan, mem)
    assert decision1.viable is True

    # If it fails again, it must halt to prevent repeating side effects
    res2 = ExecutionResult(
        success=False,
        failed_tasks=[task],
        output="TIMEOUT: Visual action timed out after 5.0 seconds",
    )
    mem = mem.record_execution(res2, wave=2)
    decision2 = evaluate_recovery_viability("click pay now", plan, mem)
    assert decision2.viable is False


# ==============================================================================
# Requirement U: Existing Non-Visual Recovery Behavior Remains Unchanged
# ==============================================================================

def test_u_non_visual_recovery_behavior_unchanged():
    """U: Non-visual tasks retain their standard retry budget (default 3 retries)."""
    task = Task(id="t_search", action="web_search", target="weather")
    plan = Plan(query="search weather", tasks=[task])

    mem = ExecutionMemory()
    # Wave 1 failure
    mem = mem.record_execution(ExecutionResult(success=False, failed_tasks=[task], output="Network error"), wave=1)
    assert evaluate_recovery_viability("search weather", plan, mem).viable is True

    # Wave 2 failure
    mem = mem.record_execution(ExecutionResult(success=False, failed_tasks=[task], output="Network error"), wave=2)
    assert evaluate_recovery_viability("search weather", plan, mem).viable is True

    # Wave 3 failure: reaches default max_task_retries (3)
    mem = mem.record_execution(ExecutionResult(success=False, failed_tasks=[task], output="Network error"), wave=3)
    assert evaluate_recovery_viability("search weather", plan, mem).viable is False
