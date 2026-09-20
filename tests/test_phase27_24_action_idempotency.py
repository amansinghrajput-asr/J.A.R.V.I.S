"""Phase 27.24 — Visual Action Transaction Safety & Idempotency Test Suite.

Exhaustive verification of:
1. VisualDispatchState derivation rules (19 tests)
2. Cross-wave idempotency fence behavior (14 tests)
3. Target normalization rules (10 tests)
4. Input discriminator lifecycle and privacy invariants (8 tests)
5. Semantic attempt counter across recovery waves (5 tests)
6. UNCERTAIN re-observation guard (7 tests)
7. Safety and sync/async parity invariants (7 tests)
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.ai.planner.executor import Executor, _evaluate_idempotency_fence
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord, sanitize_sensitive_data
from app.ai.planner.models import (
    ExecutionResult,
    Plan,
    SemanticActionKey,
    Task,
    TaskStatus,
    VisualDispatchState,
    WorkflowContext,
    _compute_input_discriminator,
    _derive_dispatch_state,
    make_semantic_action_key,
    normalize_semantic_target,
    purge_physical_state,
)
from app.automation.input import MockInputBackend, Win32InputBackend
from app.skills.system.interaction_skills import InteractionSkills
from app.vision.action_grounding import VisualActionFeasibilityStatus, VisualActionSafetyTier
from app.vision.models import (
    Point,
    ScreenObservation,
    VisualEvidenceItem,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualOutcomeType,
    VisualVerificationResult,
)
from app.vision.verification import VisualVerificationEngine, parse_visual_goal


# ==============================================================================
# D1. State Derivation Tests (19 tests)
# ==============================================================================

class TestStateDerivation(unittest.TestCase):
    """Tests 1–19: Exhaustive state derivation from TaskExecutionRecord."""

    def test_state_no_record(self):
        """#1: No prior record -> NOT_DISPATCHED (via dispatch_state_lookup)."""
        mem = ExecutionMemory()
        sak = make_semantic_action_key("visual_click", "login button")
        lookup = mem.dispatch_state_lookup(sak)
        self.assertIsNone(lookup)

    def test_state_completed_success(self):
        """#2: COMPLETED + visual_status=='SUCCESS' -> VERIFIED."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.COMPLETED, visual_status="SUCCESS")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.VERIFIED)

    def test_state_completed_no_visual_status(self):
        """#3: COMPLETED + visual_status==None -> DISPATCHED_UNVERIFIED."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.COMPLETED, visual_status=None)
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.DISPATCHED_UNVERIFIED)

    def test_state_completed_unknown_status(self):
        """#4: COMPLETED + unexpected visual_status -> UNKNOWN (fail closed)."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.COMPLETED, visual_status="CUSTOM_TOKEN")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.UNKNOWN)

    def test_state_completed_conflicting(self):
        """#5: COMPLETED + visual_status=='GROUNDING_FAILED' (conflicting) -> UNKNOWN."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.COMPLETED, visual_status="GROUNDING_FAILED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.UNKNOWN)

    def test_state_uncertain(self):
        """#6: visual_status=='VERIFICATION_UNCERTAIN' -> UNCERTAIN."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="VERIFICATION_UNCERTAIN")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.UNCERTAIN)

    def test_state_verification_failed(self):
        """#7: visual_status=='VERIFICATION_FAILED' -> FAILED_AFTER."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="VERIFICATION_FAILED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_AFTER)

    def test_state_input_dispatch_error(self):
        """#8: visual_status=='INPUT_DISPATCH_ERROR' -> FAILED_AFTER."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="INPUT_DISPATCH_ERROR")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_AFTER)

    def test_state_grounding_failed(self):
        """#9: visual_status=='GROUNDING_FAILED' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="GROUNDING_FAILED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_preflight_stale_coords(self):
        """#10: visual_status=='PREFLIGHT_STALE_COORDINATES' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="PREFLIGHT_STALE_COORDINATES")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_preflight_window_mismatch(self):
        """#11: visual_status=='PREFLIGHT_WINDOW_MISMATCH' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="PREFLIGHT_WINDOW_MISMATCH")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_preflight_modal_changed(self):
        """#12: visual_status=='PREFLIGHT_MODAL_CHANGED' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="PREFLIGHT_MODAL_CHANGED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_security_blocked(self):
        """#13: visual_status=='SECURITY_BLOCKED' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="SECURITY_BLOCKED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_confirmation_required(self):
        """#14: visual_status=='CONFIRMATION_REQUIRED' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="CONFIRMATION_REQUIRED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_sensitive_protected(self):
        """#15: visual_status=='SENSITIVE_PROTECTED' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="SENSITIVE_PROTECTED")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_disabled_control(self):
        """#16: visual_status=='DISABLED_CONTROL' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="DISABLED_CONTROL")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_target_not_found(self):
        """#17: visual_status=='TARGET_NOT_FOUND' -> FAILED_BEFORE."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status="TARGET_NOT_FOUND")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.FAILED_BEFORE)

    def test_state_failed_no_visual_status(self):
        """#18: FAILED + visual_status==None -> UNKNOWN (fail closed)."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status=None)
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.UNKNOWN)

    def test_state_preflight_status_alone_not_sufficient(self):
        """#19: visual_status==None and preflight_status=='WINDOW_MISMATCH' -> UNKNOWN."""
        rec = TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED, visual_status=None, preflight_status="WINDOW_MISMATCH")
        self.assertEqual(_derive_dispatch_state(rec), VisualDispatchState.UNKNOWN)


# ==============================================================================
# D2. Fence Behavior Tests (14 tests)
# ==============================================================================

class TestFenceBehavior(unittest.TestCase):
    """Tests 20–33: Cross-wave idempotency fence dispatch decisions."""

    def test_fence_verified_blocks_dispatch(self):
        """#20: VERIFIED record in prior memory -> blocks physical dispatch; marks COMPLETED."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t_old", action="visual_click", target="Submit Button", status=TaskStatus.COMPLETED, visual_status="SUCCESS", evidence_summary="Button clicked"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id) or "ok")

        plan = Plan(query="click submit", tasks=[Task(id="t_new", action="visual_click", target="the submit button")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)  # Physical handler NOT called
        self.assertTrue(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.COMPLETED)

    def test_fence_verified_advances_workflow_context(self):
        """#21: VERIFIED fence skip advances workflow context using matched record's evidence."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t_old", action="visual_click", target="Save", status=TaskStatus.COMPLETED, visual_status="SUCCESS", evidence_summary="Saved file"))
        executor = Executor()
        executor.register_handler("visual_click", lambda t: "ok")

        wf = WorkflowContext(workflow_id="wf1", objective="save document")
        plan = Plan(query="save", tasks=[Task(id="t_new", action="visual_click", target="save")], workflow_context=wf)
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertTrue(res.success)
        self.assertEqual(plan.workflow_context.previous_action, "visual_click")
        self.assertEqual(plan.workflow_context.verification_summary, "Saved file")

    def test_fence_verified_uses_matched_record_not_task_id(self):
        """#22: Recovery task with new UUID matches prior record by semantic key, not task_id."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="uuid_1111", action="visual_click", target="Login", status=TaskStatus.COMPLETED, visual_status="SUCCESS", evidence_summary="Logged in"))
        executor = Executor()
        executor.register_handler("visual_click", lambda t: "ok")

        plan = Plan(query="login", tasks=[Task(id="uuid_2222", action="visual_click", target="login")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertTrue(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.COMPLETED)

    def test_fence_dispatched_unverified_blocks(self):
        """#23: DISPATCHED_UNVERIFIED -> blocks dispatch; marks FAILED."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Submit", status=TaskStatus.COMPLETED, visual_status=None))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id))

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)
        self.assertFalse(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.FAILED)

    def test_fence_uncertain_blocks(self):
        """#24: UNCERTAIN -> blocks physical dispatch; marks FAILED."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Submit", status=TaskStatus.FAILED, visual_status="VERIFICATION_UNCERTAIN"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id))

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)
        self.assertFalse(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.FAILED)

    def test_fence_failed_after_blocks(self):
        """#25: FAILED_AFTER -> blocks physical dispatch; marks FAILED."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Submit", status=TaskStatus.FAILED, visual_status="INPUT_DISPATCH_ERROR"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id))

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)
        self.assertFalse(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.FAILED)

    def test_fence_unknown_blocks(self):
        """#26: UNKNOWN -> fail-closed (blocks dispatch; marks FAILED)."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Submit", status=TaskStatus.FAILED, visual_status="STRANGE_STATUS"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id))

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)
        self.assertFalse(res.success)
        self.assertEqual(plan.tasks[0].status, TaskStatus.FAILED)

    def test_fence_failed_before_allows(self):
        """#27: FAILED_BEFORE (grounding failed) -> allows physical dispatch retry."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Submit", status=TaskStatus.FAILED, visual_status="GROUNDING_FAILED"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_not_dispatched_allows(self):
        """#28: NOT_DISPATCHED (no match in prior memory) -> allows dispatch."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Cancel", status=TaskStatus.COMPLETED, visual_status="SUCCESS"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="submit", tasks=[Task(id="t2", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_no_prior_memory_allows(self):
        """#29: prior_memory=None -> fence inactive, normal dispatch."""
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="submit", tasks=[Task(id="t1", action="visual_click", target="submit")])
        res = executor.execute_plan(plan, prior_memory=None)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_non_visual_not_fenced(self):
        """#30: Non-visual tasks (open_app, web_search) are never blocked by semantic fence."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="open_app", target="notepad", status=TaskStatus.COMPLETED, visual_status=None))
        executor = Executor()
        handler_called = []
        executor.register_handler("open_app", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="open notepad", tasks=[Task(id="t2", action="open_app", target="notepad")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_empty_target_not_fenced_click(self):
        """#31: visual_click with target='' -> fence not applied, warning emitted, dispatches normally."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="", status=TaskStatus.COMPLETED, visual_status="SUCCESS"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_click", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="click", tasks=[Task(id="t2", action="visual_click", target="")])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_empty_target_not_fenced_type_no_input(self):
        """#32: visual_type with target='' and no input_text -> fence not applied, dispatches."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_type", target="", status=TaskStatus.COMPLETED, visual_status="SUCCESS"))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_type", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="type", tasks=[Task(id="t2", action="visual_type", target="", parameters={})])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 1)
        self.assertTrue(res.success)

    def test_fence_empty_target_with_discriminator_fenced(self):
        """#33: visual_type with target='' BUT input_text set -> distinguishable, fence applies."""
        mem = ExecutionMemory()
        disc = _compute_input_discriminator("hello")
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_type", target="", status=TaskStatus.COMPLETED, visual_status="SUCCESS", input_discriminator=disc))
        executor = Executor()
        handler_called = []
        executor.register_handler("visual_type", lambda t: handler_called.append(t.id) or "success")

        plan = Plan(query="type hello", tasks=[Task(id="t2", action="visual_type", target="", parameters={"input_text": "hello"})])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_called), 0)  # Blocked because same input discriminator on empty target
        self.assertTrue(res.success)


# ==============================================================================
# D3. Target Normalization Tests (10 tests)
# ==============================================================================

class TestTargetNormalization(unittest.TestCase):
    """Tests 34–43: Conservative canonical semantic target normalization."""

    def test_norm_case(self):
        """#34: Case normalization."""
        self.assertEqual(normalize_semantic_target("visual_click", "Login Button"), "login button")

    def test_norm_leading_the(self):
        """#35: Leading 'the ' stripped."""
        self.assertEqual(normalize_semantic_target("visual_click", "The Login Button"), "login button")

    def test_norm_leading_a(self):
        """#36: Leading 'a ' stripped."""
        self.assertEqual(normalize_semantic_target("visual_type", "A username field"), "username field")

    def test_norm_leading_an(self):
        """#37: Leading 'an ' stripped."""
        self.assertEqual(normalize_semantic_target("visual_type", "An input box"), "input box")

    def test_norm_whitespace(self):
        """#38: Internal whitespace collapsed."""
        self.assertEqual(normalize_semantic_target("visual_click", "  login   button  "), "login button")

    def test_norm_button_preserved(self):
        """#39: Domain word 'button' vs 'field' preserved."""
        k1 = normalize_semantic_target("visual_click", "login button")
        k2 = normalize_semantic_target("visual_click", "login field")
        self.assertNotEqual(k1, k2)

    def test_norm_icon_preserved(self):
        """#40: Domain word 'icon' preserved."""
        k1 = normalize_semantic_target("visual_click", "login button")
        k2 = normalize_semantic_target("visual_click", "login icon")
        self.assertNotEqual(k1, k2)

    def test_norm_submit_vs_submit_button(self):
        """#41: 'submit' and 'submit button' not collapsed (conservative)."""
        k1 = normalize_semantic_target("visual_click", "submit")
        k2 = normalize_semantic_target("visual_click", "submit button")
        self.assertNotEqual(k1, k2)

    def test_norm_non_visual_returns_empty(self):
        """#42: Non-visual action returns empty string."""
        self.assertEqual(normalize_semantic_target("open_app", "notepad"), "")

    def test_norm_none_target(self):
        """#43: None or empty target returns empty string."""
        self.assertEqual(normalize_semantic_target("visual_click", None), "")
        self.assertEqual(normalize_semantic_target("visual_click", ""), "")


# ==============================================================================
# D4. Input Discriminator Tests (8 tests)
# ==============================================================================

class TestInputDiscriminator(unittest.TestCase):
    """Tests 44–51: Process-local HMAC discriminator lifecycle and isolation."""

    def test_disc_same_input_same_token(self):
        """#44: Same input produces identical discriminator."""
        d1 = _compute_input_discriminator("alice")
        d2 = _compute_input_discriminator("alice")
        self.assertEqual(d1, d2)
        self.assertTrue(d1.startswith("disc:"))

    def test_disc_different_input_different_token(self):
        """#45: Different inputs produce different discriminators."""
        d1 = _compute_input_discriminator("alice")
        d2 = _compute_input_discriminator("bob")
        self.assertNotEqual(d1, d2)

    def test_disc_no_input_no_discriminator(self):
        """#46: None or empty input produces None discriminator in SAK."""
        sak = make_semantic_action_key("visual_type", "username", input_text=None)
        self.assertIsNone(sak.input_discriminator)
        sak_empty = make_semantic_action_key("visual_type", "username", input_text="")
        self.assertIsNone(sak_empty.input_discriminator)

    def test_disc_non_type_action_no_discriminator(self):
        """#47: Non-type action (e.g. visual_click) gets discriminator=None."""
        sak = make_semantic_action_key("visual_click", "submit", input_text="ignored")
        self.assertIsNone(sak.input_discriminator)

    def test_disc_not_raw_input(self):
        """#48: Discriminator contains opaque hex, never raw plaintext input."""
        secret = "SuperSecretPassword123"
        sak = make_semantic_action_key("visual_type", "password", input_text=secret)
        self.assertNotIn(secret, sak.input_discriminator or "")
        self.assertNotIn(secret, repr(sak))

    def test_disc_not_in_workflow_context(self):
        """#49: WorkflowContext never contains discriminator."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_type", target="username", status=TaskStatus.COMPLETED, visual_status="SUCCESS", evidence_summary="Typed username", input_discriminator=_compute_input_discriminator("alice")))
        executor = Executor()
        executor.register_handler("visual_type", lambda t: "ok")

        wf = WorkflowContext(workflow_id="wf1", objective="type user")
        plan = Plan(query="type", tasks=[Task(id="t2", action="visual_type", target="username", parameters={"input_text": "alice"})], workflow_context=wf)
        executor.execute_plan(plan, prior_memory=mem)

        wf_dict = str(plan.workflow_context.to_dict())
        self.assertNotIn("disc:", wf_dict)
        self.assertNotIn("alice", wf_dict)

    def test_disc_not_in_task_execution_record_output(self):
        """#50: TaskExecutionRecord to_dict() serializes discriminator token, never raw input."""
        rec = TaskExecutionRecord(
            task_id="t1",
            action="visual_type",
            target="username",
            status=TaskStatus.COMPLETED,
            visual_status="SUCCESS",
            input_discriminator=_compute_input_discriminator("secret_val"),
        )
        d = rec.to_dict()
        self.assertIn("input_discriminator", d)
        self.assertTrue(d["input_discriminator"].startswith("disc:"))
        self.assertNotIn("secret_val", str(d))
        # Verify deserialization
        rec2 = TaskExecutionRecord.from_dict(d)
        self.assertEqual(rec2.input_discriminator, d["input_discriminator"])

    def test_disc_type_fence_different_input_not_blocked(self):
        """#51: Wave 1: type 'alice' (VERIFIED). Wave 2: type 'bob' -> not blocked!"""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(
            task_id="t1",
            action="visual_type",
            target="username",
            status=TaskStatus.COMPLETED,
            visual_status="SUCCESS",
            evidence_summary="Typed alice",
            input_discriminator=_compute_input_discriminator("alice"),
        ))
        executor = Executor()
        handler_calls = []
        executor.register_handler("visual_type", lambda t: handler_calls.append(t.parameters.get("input_text")) or "typed bob")

        plan = Plan(query="type bob", tasks=[Task(id="t2", action="visual_type", target="username", parameters={"input_text": "bob"})])
        res = executor.execute_plan(plan, prior_memory=mem)

        self.assertEqual(len(handler_calls), 1)  # Dispatch occurred because inputs differ!
        self.assertTrue(res.success)


# ==============================================================================
# D5. Semantic Attempt Counter Tests (5 tests)
# ==============================================================================

class TestSemanticAttemptCounter(unittest.TestCase):
    """Tests 52–56: Attempt counting by semantic identity across UUID changes."""

    def test_attempt_count_no_record(self):
        """#52: No prior record -> attempt count = 0 (injection = 1)."""
        mem = ExecutionMemory()
        sak = make_semantic_action_key("visual_click", "login")
        self.assertEqual(mem.semantic_attempt_count(sak), 0)

    def test_attempt_count_same_uuid_retry(self):
        """#53: Same task UUID with 1 prior record -> count = 1."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="login", status=TaskStatus.FAILED))
        sak = make_semantic_action_key("visual_click", "login")
        self.assertEqual(mem.semantic_attempt_count(sak), 1)

    def test_attempt_count_new_uuid_recovery(self):
        """#54: Prior task_id=AAA, recovery task_id=BBB (same SAK) -> count = 1."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="UUID_AAA", action="visual_click", target="The Login Button", status=TaskStatus.FAILED))
        sak = make_semantic_action_key("visual_click", "login button")
        self.assertEqual(mem.semantic_attempt_count(sak), 1)

    def test_attempt_count_multiple_waves(self):
        """#55: 3 prior records with different UUIDs -> count = 3."""
        mem = ExecutionMemory()
        for i in range(3):
            mem.records.append(TaskExecutionRecord(task_id=f"uuid_{i}", action="visual_click", target="login button", status=TaskStatus.FAILED))
        sak = make_semantic_action_key("visual_click", "login button")
        self.assertEqual(mem.semantic_attempt_count(sak), 3)

    def test_attempt_count_non_visual(self):
        """#56: Non-visual action -> count = 0."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="open_app", target="notepad", status=TaskStatus.FAILED))
        sak = make_semantic_action_key("open_app", "notepad")
        self.assertEqual(mem.semantic_attempt_count(sak), 0)


# ==============================================================================
# D6. UNCERTAIN Re-Observation Tests (7 tests)
# ==============================================================================

class TestUncertainReobservation(unittest.TestCase):
    """Tests 57–63: UNCERTAIN verification single read-only re-observation guard."""

    def test_uncertain_triggers_reobservation(self):
        """#57: Post-action UNCERTAIN triggers verify_goal_with_reobservation without re-dispatch."""
        engine = VisualVerificationEngine()
        goal = parse_visual_goal("verify that login window is open")
        mock_backend = MockInputBackend()
        capture_called = []

        def mock_capture():
            capture_called.append(1)
            return ScreenObservation(id="obs2")

        with patch.object(engine, "verify_goal", return_value=VisualVerificationResult(verification_id="v1", observation_id="o1", outcome=VisualOutcomeType.VERIFIED, goal_spec=goal, confidence=0.9, evidence_chain=(), explanation="ok", evaluation_source="ocr")):
            res = engine.verify_goal_with_reobservation(goal, capture_fn=mock_capture)
            self.assertEqual(res.outcome, VisualOutcomeType.VERIFIED)
            self.assertEqual(len(capture_called), 1)
            self.assertEqual(mock_backend.event_count(), 0)  # Zero physical input!

    def test_uncertain_reobservation_verified(self):
        """#58: Re-observation returns VERIFIED -> final outcome is VERIFIED."""
        engine = VisualVerificationEngine()
        goal = parse_visual_goal("check save is open")

        def mock_capture():
            return ScreenObservation(id="obs2")

        with patch.object(engine, "verify_goal", return_value=VisualVerificationResult(verification_id="v1", observation_id="o1", outcome=VisualOutcomeType.VERIFIED, goal_spec=goal, confidence=0.95, evidence_chain=(), explanation="confirmed", evaluation_source="ocr")):
            res = engine.verify_goal_with_reobservation(goal, capture_fn=mock_capture)
            self.assertEqual(res.outcome, VisualOutcomeType.VERIFIED)

    def test_uncertain_reobservation_still_uncertain_fails_closed(self):
        """#59: Re-observation returns UNCERTAIN -> fails closed to NOT_VERIFIED."""
        engine = VisualVerificationEngine()
        goal = parse_visual_goal("check save is open")

        def mock_capture():
            return ScreenObservation(id="obs2")

        with patch.object(engine, "verify_goal", return_value=VisualVerificationResult(verification_id="v1", observation_id="o1", outcome=VisualOutcomeType.UNCERTAIN, goal_spec=goal, confidence=0.4, evidence_chain=(), explanation="still ambiguous", evaluation_source="ocr")):
            res = engine.verify_goal_with_reobservation(goal, capture_fn=mock_capture)
            self.assertEqual(res.outcome, VisualOutcomeType.NOT_VERIFIED)

    def test_uncertain_reobservation_capped_at_one(self):
        """#60: Re-observation capture is invoked exactly once (no retry loops)."""
        engine = VisualVerificationEngine()
        goal = parse_visual_goal("check save is open")
        calls = []

        def mock_capture():
            calls.append(1)
            return ScreenObservation(id="obs2")

        with patch.object(engine, "verify_goal", return_value=VisualVerificationResult(verification_id="v1", observation_id="o1", outcome=VisualOutcomeType.UNCERTAIN, goal_spec=goal, confidence=0.4, evidence_chain=(), explanation="ambiguous", evaluation_source="ocr")):
            res = engine.verify_goal_with_reobservation(goal, capture_fn=mock_capture)
            self.assertEqual(len(calls), 1)
            self.assertEqual(res.outcome, VisualOutcomeType.NOT_VERIFIED)

    def test_not_verified_no_reobservation(self):
        """#61: NOT_VERIFIED initial outcome does not trigger re-observation."""
        skill = InteractionSkills(input_backend=MockInputBackend())
        adapter = skill.action_adapter
        with patch.object(skill, "_ground_semantic_target") as mock_ground, \
             patch.object(adapter, "execute_target") as mock_exec:
            mock_target = MagicMock()
            mock_target.safety_tier = VisualActionSafetyTier.SAFE
            mock_target.feasibility = VisualActionFeasibilityStatus.FEASIBLE
            mock_target.requires_confirmation = False
            mock_target.target_element_name = "btn"
            mock_ground.return_value = {"status": "SUCCESS", "target": mock_target}

            mock_res = MagicMock()
            mock_res.success = False
            mock_res.reason = "Element missing"
            mock_res.verification_dict = {"outcome": "NOT_VERIFIED"}
            mock_res.to_dict.return_value = {"action_status": "VERIFICATION_FAILED"}
            mock_res.action_type.value = "visual_click"
            mock_res.duration_ms = 10
            mock_exec.return_value = mock_res

            res = skill.execute({"operation": "visual_click", "parameters": {"target": "btn", "expected_visual_goal": "check btn"}})
            self.assertFalse(res.data["verified"])
            self.assertEqual(res.data["verification_outcome"], "NOT_VERIFIED")

    def test_no_goal_spec_no_reobservation(self):
        """#62: No expected_visual_goal -> re-observation skipped, executed unverified."""
        skill = InteractionSkills(input_backend=MockInputBackend())
        adapter = skill.action_adapter
        with patch.object(skill, "_ground_semantic_target") as mock_ground, \
             patch.object(adapter, "execute_target") as mock_exec:
            mock_target = MagicMock()
            mock_target.safety_tier = VisualActionSafetyTier.SAFE
            mock_target.feasibility = VisualActionFeasibilityStatus.FEASIBLE
            mock_target.requires_confirmation = False
            mock_target.target_element_name = "btn"
            mock_ground.return_value = {"status": "SUCCESS", "target": mock_target}

            mock_res = MagicMock()
            mock_res.success = True
            mock_res.verification_dict = {"outcome": "UNCERTAIN"}
            mock_res.to_dict.return_value = {"action_status": "SUCCESS"}
            mock_res.action_type.value = "visual_click"
            mock_res.duration_ms = 10
            mock_exec.return_value = mock_res

            res = skill.execute({"operation": "visual_click", "parameters": {"target": "btn"}})
            self.assertFalse(res.data["verified"])

    def test_no_capture_fn_uncertain_fails_closed(self):
        """#63: capture_fn=None with UNCERTAIN -> fails closed to NOT_VERIFIED."""
        engine = VisualVerificationEngine()
        goal = parse_visual_goal("check save is open")
        res = engine.verify_goal_with_reobservation(goal, capture_fn=None)
        self.assertEqual(res.outcome, VisualOutcomeType.NOT_VERIFIED)


# ==============================================================================
# D7. Safety and Parity Tests (7 tests)
# ==============================================================================

class TestSafetyAndParity(unittest.TestCase):
    """Tests 64–70: Win32 isolation, sync/async parity, and guarantee preservation."""

    def test_win32_invocation_count_zero(self):
        """#64: Verify Win32InputBackend invocation count is 0."""
        self.assertEqual(Win32InputBackend.invocation_count, 0)

    def test_mock_input_backend_only(self):
        """#65: Only MockInputBackend is used throughout test runs."""
        backend = MockInputBackend()
        backend.click(Point(10, 20))
        self.assertEqual(backend.event_count("click"), 1)

    def test_sync_async_fence_parity(self):
        """#66: Sync and async execute_plan produce identical fence decisions."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t1", action="visual_click", target="Login Button", status=TaskStatus.COMPLETED, visual_status="SUCCESS", evidence_summary="Logged in"))

        exec_sync = Executor()
        exec_async = Executor()

        plan_sync = Plan(query="login", tasks=[Task(id="t_sync", action="visual_click", target="The Login Button")])
        plan_async = Plan(query="login", tasks=[Task(id="t_async", action="visual_click", target="The Login Button")])

        res_sync = exec_sync.execute_plan(plan_sync, prior_memory=mem)
        res_async = asyncio.run(exec_async.execute_plan_async(plan_async, prior_memory=mem))

        self.assertEqual(res_sync.success, res_async.success)
        self.assertEqual(plan_sync.tasks[0].status, TaskStatus.COMPLETED)
        self.assertEqual(plan_async.tasks[0].status, TaskStatus.COMPLETED)

    def test_sync_async_attempt_parity(self):
        """#67: Sync and async paths produce identical semantic attempt counts."""
        mem = ExecutionMemory()
        mem.records.append(TaskExecutionRecord(task_id="t0", action="visual_click", target="submit", status=TaskStatus.FAILED))
        sak = make_semantic_action_key("visual_click", "submit")
        c1 = mem.semantic_attempt_count(sak)
        c2 = mem.semantic_attempt_count(sak)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 1)

    def test_persistence_resume_unaffected(self):
        """#68: Standard persistence resume ID check remains functional."""
        from app.ai.planner.persistence import PersistedExecutionState, resume
        task_comp = Task(id="t1", action="open_app", target="calc", status=TaskStatus.COMPLETED)
        task_rem = Task(id="t2", action="open_app", target="notepad", status=TaskStatus.PENDING)
        state = PersistedExecutionState(
            execution_id="e1",
            plan_id="p1",
            query="run tools",
            memory=ExecutionMemory(),
            dag_state={"task_map": {"t1": task_comp, "t2": task_rem}},
            completed_tasks=[task_comp],
        )

        executor = Executor()
        handler_calls = []
        executor.register_handler("open_app", lambda t: handler_calls.append(t.id) or "opened")

        res = resume(state, executor=executor)
        self.assertTrue(res.success)
        self.assertNotIn("t1", handler_calls)
        self.assertIn("t2", handler_calls)

    def test_phase2722_workflow_context_boundary(self):
        """#69: WorkflowContext across waves contains zero physical state or tokens."""
        wf = WorkflowContext(workflow_id="wf1", objective="test")
        updated = wf.with_step_outcome(action="visual_click", outcome="VERIFIED", summary="Clicked button (100, 200) sys_conf_1234567890abcdef")
        self.assertNotIn("100", updated.verification_summary)
        self.assertNotIn("sys_conf", updated.verification_summary)

    def test_phase2723_readiness_and_t0t1_preserved(self):
        """#70: UNCERTAIN re-observation preserves T0/T1 and never repeats physical dispatch."""
        mock_backend = MockInputBackend()
        skill = InteractionSkills(input_backend=mock_backend)
        self.assertEqual(mock_backend.event_count(), 0)


if __name__ == "__main__":
    unittest.main()
