"""Unit tests for Phase 20 Sprint 20.4 Causal Reflection Engine & Macro-Skill Compiler.

Covers:
- Causal failure diagnosis (MISSING_TOOL, EXECUTION_TIMEOUT, CONSENSUS_FAILURE, POLICY_REJECTION, PRECONDITION_VIOLATION, UNKNOWN)
- Invariant constraint discovery and deduplication
- Prescriptions generation and report export
- Macro-skill subplan compilation into standalone Python code
- Sandbox verification and automatic promotion rules
- Direct invocation of compiled macro-skills
- Quarantine and unquarantine mechanics
- Skill removal, deterministic listing, and metadata queries
- Concurrent compilations, invocations, and thread safety
"""

import concurrent.futures
import unittest
from typing import Any, Dict

from app.ai.planner.events import (
    CausalDiagnosisGenerated,
    PlannerEvent,
    PlannerEventBus,
    SkillDistillationCompleted,
)
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.models import (
    CausalDiagnosis,
    DistilledSkillMetadata,
)
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.models import Plan, Task


class TestCausalReflectionEngine(unittest.TestCase):
    """Test suite for CausalReflectionEngine root cause analysis and invariant discovery."""

    def setUp(self) -> None:
        self.event_bus = PlannerEventBus()
        self.events_received: list[PlannerEvent] = []
        self.event_bus.subscribe(PlannerEvent, lambda e: self.events_received.append(e))
        self.engine = CausalReflectionEngine(event_bus=self.event_bus)

    def test_diagnose_missing_tool(self) -> None:
        """Verify diagnosing a missing tool handler failure."""
        traj = {
            "trajectory_id": "traj_miss",
            "failed_task_id": "task_calc",
            "error": "Task execution failed: No handler registered for action 'calc_tax'",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "MISSING_TOOL")
        self.assertIn("Always verify tool availability", diag.invariant_constraint)
        self.assertGreaterEqual(diag.confidence, 0.90)

        # Verify event published
        event_types = [type(e) for e in self.events_received]
        self.assertIn(CausalDiagnosisGenerated, event_types)

    def test_diagnose_execution_timeout(self) -> None:
        """Verify diagnosing an execution deadline timeout."""
        traj = {
            "trajectory_id": "traj_time",
            "failed_task_id": "task_heavy",
            "error": "Task execution failed: Deadline timeout expired after 5.0s",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "EXECUTION_TIMEOUT")
        self.assertIn("exponential backoff", diag.invariant_constraint)

    def test_diagnose_consensus_failure(self) -> None:
        """Verify diagnosing a multi-agent consensus quorum failure."""
        traj = {
            "trajectory_id": "traj_cons",
            "failed_task_id": "task_vote",
            "error": "Consensus error: Ballot rejected, tie broken without quorum",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "CONSENSUS_FAILURE")
        self.assertIn("consensus", diag.invariant_constraint.lower())

    def test_diagnose_policy_rejection(self) -> None:
        """Verify diagnosing a security policy guardrail rejection."""
        traj = {
            "trajectory_id": "traj_sec",
            "failed_task_id": "task_rm",
            "error": "Policy violation: rm -rf /root is strictly forbidden",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "POLICY_REJECTION")
        self.assertIn("safety policy", diag.invariant_constraint)

    def test_diagnose_precondition_violation(self) -> None:
        """Verify diagnosing an invalid input schema or missing prerequisite."""
        traj = {
            "trajectory_id": "traj_pre",
            "failed_task_id": "task_db",
            "error": "Precondition failed: missing parameter 'db_token' required by schema",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "PRECONDITION_VIOLATION")
        self.assertIn("prerequisites", diag.invariant_constraint)

    def test_diagnose_unknown_failure(self) -> None:
        """Verify diagnosing an unspecified runtime failure."""
        traj = {
            "trajectory_id": "traj_un",
            "failed_task_id": "task_crash",
            "error": "Unexpected NullReferenceException in module X",
            "success": False,
        }
        diag = self.engine.diagnose_failure(traj)
        self.assertEqual(diag.root_cause_type, "UNKNOWN")
        self.assertEqual(diag.confidence, 0.50)

    def test_discover_invariants_and_deduplication(self) -> None:
        """Verify invariant discovery across multiple trajectories removes duplicates."""
        trajs = [
            {"trajectory_id": "t1", "error": "missing_tool handler not found", "success": False},
            {"trajectory_id": "t2", "error": "another missing_tool error", "success": False},
            {"trajectory_id": "t3", "error": "timeout expired", "success": False},
            {"trajectory_id": "t4", "error": "all good", "success": True},
        ]
        invariants = self.engine.discover_invariants(trajs)
        # Should contain missing tool and timeout invariants, deduplicated and sorted
        self.assertEqual(len(invariants), 2)
        self.assertTrue(any("verify tool availability" in inv for inv in invariants))
        self.assertTrue(any("exponential backoff" in inv for inv in invariants))

    def test_generate_prescriptions(self) -> None:
        """Verify generating actionable prescriptions based on goal intent."""
        # Seed engine with diagnoses
        self.engine.diagnose_failure({"trajectory_id": "t1", "error": "missing_tool in calculate", "success": False})
        self.engine.diagnose_failure({"trajectory_id": "t2", "error": "timeout in large download", "success": False})

        prescriptions = self.engine.generate_prescriptions("calculate and download large dataset")
        self.assertGreaterEqual(len(prescriptions), 1)

    def test_export_report_and_clear(self) -> None:
        """Verify exporting diagnostic report and clearing engine state."""
        self.engine.diagnose_failure({"trajectory_id": "t1", "error": "timeout", "success": False})
        report = self.engine.export_report()
        self.assertEqual(report["total_diagnoses"], 1)
        self.assertIn("EXECUTION_TIMEOUT", report["by_root_cause"])

        self.engine.clear()
        self.assertEqual(len(self.engine.export_report()["diagnoses"]), 0)


class TestMacroSkillCompiler(unittest.TestCase):
    """Test suite for MacroSkillCompiler compilation, promotion, and execution."""

    def setUp(self) -> None:
        self.sandbox = SandboxedExecutionHarness(timeout_seconds=2.0)
        self.event_bus = PlannerEventBus()
        self.events_received: list[PlannerEvent] = []
        self.event_bus.subscribe(PlannerEvent, lambda e: self.events_received.append(e))
        self.compiler = MacroSkillCompiler(sandbox=self.sandbox, event_bus=self.event_bus)

    def test_compile_and_invoke_macro_skill(self) -> None:
        """Verify compiling a multi-task subplan into an executable macro-skill."""
        plan = Plan(
            query="data pipeline",
            tasks=[
                Task(action="fetch_data", target="users_table"),
                Task(action="transform_data", target="clean_records"),
            ],
        )

        meta = self.compiler.compile_subplan_to_skill(
            subplan=plan,
            skill_name="user_data_pipeline",
            description="Extract and transform user records",
            historical_avg_latency_ms=500.0,
        )

        self.assertTrue(meta.verified)
        self.assertEqual(meta.skill_name, "user_data_pipeline")
        self.assertGreaterEqual(meta.speedup_multiplier, 1.0)
        self.assertIn("user_data_pipeline", self.compiler.list_skills())

        # Invoke the compiled macro-skill
        output = self.compiler.invoke_skill("user_data_pipeline")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["step_count"], 2)

        # Verify event emitted
        event_types = [type(e) for e in self.events_received]
        self.assertIn(SkillDistillationCompleted, event_types)

    def test_quarantine_prevents_invocation(self) -> None:
        """Verify quarantining a skill prevents its execution."""
        plan = Plan(query="step", tasks=[Task(action="op", target="t1")])
        self.compiler.compile_subplan_to_skill(plan, "quarantine_test")

        # Quarantined list excludes skill by default
        self.compiler.quarantine_skill("quarantine_test", reason="Unstable behavior detected")
        self.assertNotIn("quarantine_test", self.compiler.list_skills(include_quarantined=False))
        self.assertIn("quarantine_test", self.compiler.list_skills(include_quarantined=True))

        # Invoking quarantined skill raises RuntimeError
        with self.assertRaises(RuntimeError):
            self.compiler.invoke_skill("quarantine_test")

        # Unquarantine restores execution
        self.compiler.unquarantine_skill("quarantine_test")
        res = self.compiler.invoke_skill("quarantine_test")
        self.assertEqual(res["status"], "success")

    def test_remove_skill(self) -> None:
        """Verify removing a skill deletes it and its metadata."""
        plan = Plan(query="step", tasks=[Task(action="op", target="t1")])
        self.compiler.compile_subplan_to_skill(plan, "temp_skill")
        self.assertIn("temp_skill", self.compiler.list_skills())

        self.assertTrue(self.compiler.remove_skill("temp_skill"))
        self.assertNotIn("temp_skill", self.compiler.list_skills())
        self.assertIsNone(self.compiler.get_metadata("temp_skill"))

    def test_automatic_promotion_rule_requires_verification(self) -> None:
        """Verify unverified skills cannot be registered into compiler registry."""
        unverified_meta = DistilledSkillMetadata(
            skill_name="unverified",
            source_subplan_signature="tasks_0",
            compiled_code="",
            verified=False,
        )

        with self.assertRaises(ValueError):
            self.compiler.register_skill("unverified", lambda: 1, unverified_meta)

    def test_invoke_missing_skill_raises_key_error(self) -> None:
        """Verify invoking a non-existent skill raises KeyError."""
        with self.assertRaises(KeyError):
            self.compiler.invoke_skill("non_existent_skill")

    def test_concurrent_compilation_and_invocation(self) -> None:
        """Verify thread-safety during concurrent skill compilations and invocations."""
        def _compile_and_run(idx: int) -> Any:
            p = Plan(query=f"plan_{idx}", tasks=[Task(action="step", target=f"t_{idx}")])
            name = f"skill_{idx}"
            self.compiler.compile_subplan_to_skill(p, name)
            return self.compiler.invoke_skill(name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_compile_and_run, i) for i in range(15)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 15)
        for r in results:
            self.assertEqual(r["status"], "success")

        self.assertEqual(len(self.compiler.list_skills()), 15)

    def test_deterministic_metadata_and_clear(self) -> None:
        """Verify get_metadata and clear operations."""
        plan = Plan(query="test", tasks=[Task(action="a1")])
        self.compiler.compile_subplan_to_skill(plan, "alpha_skill")

        meta = self.compiler.get_metadata("alpha_skill")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.skill_name, "alpha_skill")

        self.compiler.clear()
        self.assertEqual(len(self.compiler.list_skills(include_quarantined=True)), 0)

    def test_sandbox_failure_blocks_promotion(self) -> None:
        """Verify sandbox failure marks skill unverified and blocks automatic promotion."""
        class FailingSandbox(SandboxedExecutionHarness):
            def execute_in_sandbox(self, code: str, test_code: str) -> Any:
                from app.ai.planner.metacognition.models import SandboxVerificationResult
                return SandboxVerificationResult(success=False, exit_code=1, error="Assertion failed in test")

        compiler = MacroSkillCompiler(sandbox=FailingSandbox())
        plan = Plan(query="test", tasks=[Task(action="a1")])
        meta = compiler.compile_subplan_to_skill(plan, "failing_skill")

        self.assertFalse(meta.verified)
        self.assertNotIn("failing_skill", compiler.list_skills(include_quarantined=True))

    def test_ast_rejection_blocks_promotion(self) -> None:
        """Verify AST security violation marks skill unverified and blocks promotion."""
        class RejectionSandbox(SandboxedExecutionHarness):
            def scan_ast_security(self, python_code: str) -> tuple[bool, str | None]:
                return False, "Security violation: Forbidden call"

        compiler = MacroSkillCompiler(sandbox=RejectionSandbox())
        plan = Plan(query="test", tasks=[Task(action="a1")])
        meta = compiler.compile_subplan_to_skill(plan, "rejected_skill")

        self.assertFalse(meta.verified)
        self.assertNotIn("rejected_skill", compiler.list_skills(include_quarantined=True))


if __name__ == "__main__":
    unittest.main()
