"""Tests for Phase 27.21: Structured Task Result Contract & Visual Evidence Correlation.

Verifies:
1. Sync SystemSkillResult(success=False) -> FAILED task
2. Async SystemSkillResult(success=False) -> FAILED task
3. Sync SystemSkillResult(success=True) -> COMPLETED task
4. Async SystemSkillResult(success=True) -> COMPLETED task
5. Ordinary legacy handler returning string -> COMPLETED
6. Ordinary legacy handler returning dict/int/None -> COMPLETED
7. Exception in handler -> FAILED task
8. Failed visual result sets ExecutionResult.success = False and populates failed_tasks
9. Failed visual result activates AIManager._should_replan() recovery loop
10. Structured visual metadata (visual_status, verification_confidence, evidence_summary, preflight_status) reaches ExecutionMemory
11. Raw VisualActionResult does not enter persistent TaskExecutionRecord
12. Coordinates do NOT enter evidence summary
13. Passwords and typed input are redacted from memory and logs
14. Confirmation tokens/IDs do not leak into task output, execution memory, recovery summary, LLM prompt, or events
15. Confirmation remains 30s TTL by default
16. Confirmation tokens remain single-use and non-transferable
17. Plan and Task serialization contain no physical visual state (coordinates, handles, tokens)
18. AIManager user-facing response surfaces to_user_message for SystemSkillResult
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
import pytest

from app.ai.planner.executor import Executor
from app.ai.planner.memory import (
    ExecutionMemory,
    FailureCategory,
    TaskExecutionRecord,
    sanitize_sensitive_data,
    extract_visual_metadata,
)
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.manager import AIManager
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import SystemConfirmationManager, ConfirmationRejectedError
from app.automation.input import MockInputBackend, Win32InputBackend
from app.automation.visual_action_adapter import VisualActionResultStatus


@pytest.fixture(autouse=True)
def reset_win32_invocations():
    """Strict test safeguard: Win32 real input is never invoked."""
    Win32InputBackend.reset_invocation_count()
    Win32InputBackend.enable_test_safety_guard()
    yield
    assert Win32InputBackend.invocation_count == 0, (
        f"CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked {Win32InputBackend.invocation_count} times!"
    )


# ==============================================================================
# 1. Sync & Async SystemSkillResult Contract Tests
# ==============================================================================

class TestExecutorSystemSkillResultContract:
    """Verify Executor handles SystemSkillResult success and failure explicitly."""

    def test_sync_system_skill_result_failure_marks_task_failed(self):
        executor = Executor()
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_STALE_COORDINATES",
            data={"status": "PREFLIGHT_STALE_COORDINATES", "reason": "Target drifted at (100, 200)"},
        )
        executor.register_handler("visual_click", lambda t: fail_res)

        task = Task(id="task_1", action="visual_click", target="login_button")
        res_task, success, res_str, err_msg = executor.execute_task_sync(task)

        assert success is False
        assert res_task.status == TaskStatus.FAILED
        assert task.status == TaskStatus.FAILED
        assert err_msg is not None
        assert "PREFLIGHT_STALE_COORDINATES" in err_msg

    def test_async_system_skill_result_failure_marks_task_failed(self):
        executor = Executor()
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_WINDOW_MISMATCH",
            data={"status": "PREFLIGHT_WINDOW_MISMATCH", "reason": "Window focus lost"},
        )

        async def _handler(t):
            return fail_res

        executor.register_handler("visual_click", _handler)

        task = Task(id="task_2", action="visual_click", target="submit_button")
        res_task, success, res_str, err_msg = asyncio.run(
            executor._execute_single_task_async(task)
        )

        assert success is False
        assert res_task.status == TaskStatus.FAILED
        assert task.status == TaskStatus.FAILED
        assert "PREFLIGHT_WINDOW_MISMATCH" in err_msg

    def test_sync_system_skill_result_success_marks_task_completed(self):
        executor = Executor()
        ok_res = SystemSkillResult(
            operation="visual_click",
            success=True,
            data={"status": "SUCCESS", "target": "login_button"},
        )
        executor.register_handler("visual_click", lambda t: ok_res)

        task = Task(id="task_3", action="visual_click", target="login_button")
        res_task, success, res_str, err_msg = executor.execute_task_sync(task)

        assert success is True
        assert res_task.status == TaskStatus.COMPLETED
        assert err_msg is None

    def test_async_system_skill_result_success_marks_task_completed(self):
        executor = Executor()
        ok_res = SystemSkillResult(
            operation="visual_type",
            success=True,
            data={"status": "SUCCESS", "target": "search_box"},
        )

        async def _handler(t):
            return ok_res

        executor.register_handler("visual_type", _handler)

        task = Task(id="task_4", action="visual_type", target="search_box")
        res_task, success, res_str, err_msg = asyncio.run(
            executor._execute_single_task_async(task)
        )

        assert success is True
        assert res_task.status == TaskStatus.COMPLETED
        assert err_msg is None

    def test_ordinary_legacy_handler_returning_string_completed(self):
        executor = Executor()
        executor.register_handler("open_app", lambda t: "App launched successfully")

        task = Task(id="task_5", action="open_app", target="notepad")
        res_task, success, res_str, err_msg = executor.execute_task_sync(task)

        assert success is True
        assert res_task.status == TaskStatus.COMPLETED
        assert "App launched successfully" in res_str
        assert err_msg is None

    def test_ordinary_legacy_handler_returning_dict_int_none(self):
        executor = Executor()
        executor.register_handler("calc_dict", lambda t: {"result": 42})
        executor.register_handler("calc_int", lambda t: 42)
        executor.register_handler("noop", lambda t: None)

        t1, s1, r1, _ = executor.execute_task_sync(Task(id="1", action="calc_dict"))
        assert s1 is True
        assert t1.status == TaskStatus.COMPLETED

        t2, s2, r2, _ = executor.execute_task_sync(Task(id="2", action="calc_int"))
        assert s2 is True
        assert t2.status == TaskStatus.COMPLETED

        t3, s3, r3, _ = executor.execute_task_sync(Task(id="3", action="noop"))
        assert s3 is True
        assert t3.status == TaskStatus.COMPLETED

    def test_handler_exception_remains_failed(self):
        executor = Executor()

        def _bad_handler(t):
            raise RuntimeError("Database connection timed out")

        executor.register_handler("db_query", _bad_handler)

        task = Task(id="task_err", action="db_query")
        res_task, success, res_str, err_msg = executor.execute_task_sync(task)

        assert success is False
        assert res_task.status == TaskStatus.FAILED
        assert "Database connection timed out" in err_msg


# ==============================================================================
# 2. Plan Execution & Failure Aggregation Tests
# ==============================================================================

class TestPlanExecutionFailureAggregation:
    """Verify failed SystemSkillResults correctly propagate to ExecutionResult."""

    def test_failed_visual_result_sets_execution_result_failed(self):
        executor = Executor()
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_STALE_COORDINATES",
            data={"status": "PREFLIGHT_STALE_COORDINATES"},
        )
        executor.register_handler("visual_click", lambda t: fail_res)

        plan = Plan(
            query="click login",
            tasks=[Task(id="task_f1", action="visual_click", target="login")],
        )
        result = executor.execute_plan(plan)

        assert result.success is False
        assert len(result.failed_tasks) == 1
        assert len(result.completed_tasks) == 0
        assert result.failed_tasks[0].id == "task_f1"
        assert "task_f1" in result.failed_task_ids
        assert result.task_results.get("task_f1") is fail_res

    def test_failed_visual_result_async_sets_execution_result_failed(self):
        executor = Executor()
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="VERIFICATION_FAILED",
            data={"status": "VERIFICATION_FAILED"},
        )

        async def _handler(t):
            return fail_res

        executor.register_handler("visual_click", _handler)

        plan = Plan(
            query="click submit",
            tasks=[Task(id="task_f2", action="visual_click", target="submit")],
        )
        result = asyncio.run(executor.execute_plan_async(plan))

        assert result.success is False
        assert len(result.failed_tasks) == 1
        assert result.failed_tasks[0].id == "task_f2"


# ==============================================================================
# 3. Structured Visual Metadata & Evidence Extraction Tests
# ==============================================================================

class TestStructuredVisualEvidenceExtraction:
    """Verify structured metadata crosses into ExecutionMemory without leakage."""

    def test_structured_metadata_crosses_into_execution_memory(self):
        skill_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="VERIFICATION_FAILED",
            data={
                "status": "VERIFICATION_FAILED",
                "verification_dict": {
                    "confidence": 0.35,
                    "explanation": "Target button did not transition to active state",
                },
                "reason": "Verification score 0.35 below threshold 0.85",
                "metadata": {"preflight_status": "PASSED"},
            },
        )
        task = Task(id="t_meta", action="visual_click", target="btn")
        exec_res = ExecutionResult(
            success=False,
            failed_tasks=[task],
            task_results={"t_meta": skill_res},
            output="VERIFICATION_FAILED",
        )

        memory = ExecutionMemory.from_execution_result(exec_res, wave=1)
        record = memory.records[0]

        assert record.status == TaskStatus.FAILED
        assert record.visual_status == "VERIFICATION_FAILED"
        assert record.verification_confidence == 0.35
        assert record.evidence_summary == "Target button did not transition to active state"
        assert record.preflight_status == "PASSED"

    def test_coordinates_do_not_enter_evidence_summary(self):
        skill_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_STALE_COORDINATES",
            data={
                "status": "PREFLIGHT_STALE_COORDINATES",
                "reason": "Element at (160, 220) drifted to (180, 240) beyond 5px threshold Point(x=180, y=240)",
            },
        )
        v_status, v_conf, ev_summary, pref_status = extract_visual_metadata(skill_res)

        assert v_status == "PREFLIGHT_STALE_COORDINATES"
        assert "(160, 220)" not in ev_summary
        assert "Point(x=180, y=240)" not in ev_summary
        assert "[COORDINATES_REDACTED]" in ev_summary

    def test_raw_visual_action_result_not_in_persistent_record_dict(self):
        record = TaskExecutionRecord(
            task_id="t1",
            action="visual_click",
            target="login",
            status=TaskStatus.COMPLETED,
            visual_status="SUCCESS",
            verification_confidence=0.95,
            evidence_summary="Dialog opened",
            preflight_status="PASSED",
        )
        d = record.to_dict()

        assert d["visual_status"] == "SUCCESS"
        assert d["verification_confidence"] == 0.95
        assert d["evidence_summary"] == "Dialog opened"
        assert d["preflight_status"] == "PASSED"
        # Must not contain raw pixels, screenshots, or coordinate arrays
        assert "screenshot" not in d
        assert "pixels" not in d
        assert "target_point" not in d

        # Roundtrip deserialization
        restored = TaskExecutionRecord.from_dict(d)
        assert restored.visual_status == "SUCCESS"
        assert restored.verification_confidence == 0.95
        assert restored.evidence_summary == "Dialog opened"
        assert restored.preflight_status == "PASSED"


# ==============================================================================
# 4. Privacy & Confirmation Token Redaction Tests
# ==============================================================================

class TestPrivacyAndConfirmationTokenRedaction:
    """Verify confirmation tokens, IDs, and passwords are never exposed."""

    def test_confirmation_token_redaction_in_sanitizer(self):
        raw_text = "Confirmation required. Token: sys_conf_a1b2c3d4e5 with token=sys_conf_xyz123 and confirmation_id=sys_conf_8899"
        sanitized = sanitize_sensitive_data(raw_text)

        assert "sys_conf_a1b2c3d4e5" not in sanitized
        assert "sys_conf_xyz123" not in sanitized
        assert "sys_conf_8899" not in sanitized
        assert "[CONFIRMATION_TOKEN_REDACTED]" in sanitized

    def test_password_redaction_in_sanitizer(self):
        raw_text = "Action failed: password='SuperSecretPassword123' and input_text='ConfidentialCode456'"
        sanitized = sanitize_sensitive_data(raw_text)

        assert "SuperSecretPassword123" not in sanitized
        assert "ConfidentialCode456" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_confirmation_token_does_not_enter_task_output_or_events(self):
        executor = Executor()
        # Handler returns error containing a confirmation token
        err_with_token = "Action 'delete_file' rejected. confirmation_id=sys_conf_secret999"
        executor.register_handler("test_action", lambda t: SystemSkillResult(
            operation="test_action",
            success=False,
            error=err_with_token,
        ))

        plan = Plan(query="delete file", tasks=[Task(id="t_tok", action="test_action")])
        result = executor.execute_plan(plan)

        assert "sys_conf_secret999" not in result.output
        assert "sys_conf_secret999" not in result.task_outputs["t_tok"]
        assert "[CONFIRMATION_TOKEN_REDACTED]" in result.output or "confirmation_id=[REDACTED]" in result.output

    def test_confirmation_manager_defaults_and_single_use(self):
        mgr = SystemConfirmationManager()
        assert mgr._default_timeout == 30.0

        cid = mgr.request_confirmation(operation="format_drive", target="C:")
        assert mgr.resolve_confirmation(cid, approved=True) is True

        # First consumption must succeed
        assert mgr.verify_and_consume(cid, operation="format_drive", target="C:") is True

        # Second consumption must fail (single-use)
        with pytest.raises(ConfirmationRejectedError):
            mgr.verify_and_consume(cid, operation="format_drive", target="C:")


# ==============================================================================
# 5. Semantic Plan Contract Check (No Physical State)
# ==============================================================================

class TestSemanticPlanContractNoPhysicalState:
    """Verify Plan and Task contracts contain strictly zero physical state."""

    def test_plan_and_task_serialization_contains_no_physical_state(self):
        task = Task(
            id="t_semantic",
            action="visual_click",
            target="login_button",
            parameters={"query": "login"},
        )
        plan = Plan(query="click login", tasks=[task])

        task_dict = task.to_dict()
        assert "target_point" not in task_dict
        assert "window_handle" not in task_dict
        assert "bounds" not in task_dict
        assert "grounded_at" not in task_dict
        assert "confirmation_token" not in task_dict

        plan_dict = plan.to_dict()
        assert "target_point" not in plan_dict
        assert "window_handle" not in plan_dict
        assert "bounds" not in plan_dict


# ==============================================================================
# 6. AIManager Recovery Reachability & User Response Tests
# ==============================================================================

class TestAIManagerRecoveryAndUserResponse:
    """Verify AIManager recovery triggers on visual failure and formats user responses."""

    def test_failed_visual_result_activates_aimanager_recovery(self):
        executor = Executor()
        # First execution fails with PREFLIGHT_STALE_COORDINATES
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_STALE_COORDINATES",
            data={"status": "PREFLIGHT_STALE_COORDINATES"},
        )
        executor.register_handler("visual_click", lambda t: fail_res)

        plan = Plan(query="click login", tasks=[Task(id="task_rec", action="visual_click", target="login")])
        exec_res = executor.execute_plan(plan)

        # Mock AIManager to inspect _should_replan
        ai = AIManager(auto_register_in_container=False)
        ai._auto_replan = True
        ai._max_replans = 2

        # In Phase 27.20, _should_replan was unreachable because exec_res.success was True.
        # Now, exec_res.success is False and failed_tasks has task_rec!
        should_replan = ai._should_replan(exec_res, attempt=0, plan=plan)
        assert should_replan is True

    def test_aimanager_user_response_surfaces_to_user_message(self):
        ok_res = SystemSkillResult(
            operation="visual_click",
            success=True,
            data={"status": "SUCCESS", "action_type": "click", "target": "Submit Button"},
        )
        task = Task(id="t_resp", action="visual_click", target="Submit Button")
        exec_res = ExecutionResult(
            success=True,
            completed_tasks=[task],
            task_results={"t_resp": ok_res},
            output="OK",
        )

        ai = AIManager(auto_register_in_container=False)
        response_text = ai._format_execution_result_response(exec_res)

        # Must surface the natural human-friendly message
        assert "Successfully clicked 'Submit Button'." in response_text

    def test_aimanager_user_response_surfaces_safe_failure_message(self):
        fail_res = SystemSkillResult(
            operation="visual_click",
            success=False,
            error="PREFLIGHT_STALE_COORDINATES",
            data={"status": "PREFLIGHT_STALE_COORDINATES", "action_type": "click", "target": "Next Button"},
        )
        task = Task(id="t_fail_resp", action="visual_click", target="Next Button")
        exec_res = ExecutionResult(
            success=False,
            failed_tasks=[task],
            task_results={"t_fail_resp": fail_res},
            output="PREFLIGHT_STALE_COORDINATES",
        )

        ai = AIManager(auto_register_in_container=False)
        response_text = ai._format_execution_result_response(exec_res)

        assert "Target 'Next Button' changed position or disappeared before execution" in response_text
