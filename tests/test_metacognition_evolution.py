"""Unit tests for Phase 21: Autonomous Learning & Skill Evolution.

Covers all 18 required scenarios:
1. Skill registration
2. Invocation tracking
3. Success tracking
4. Failure tracking
5. Success-rate calculation
6. Confidence scoring
7. Degradation
8. Quarantine
9. Retirement
10. Version creation
11. Stable-version selection
12. Rollback
13. Improvement workflow
14. Reflection integration
15. Concurrent updates
16. Controller integration
17. Backward compatibility
18. Deterministic state transitions
"""

import concurrent.futures
import threading
import time
from typing import Tuple
import unittest
from unittest.mock import MagicMock

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SkillDegraded,
    SkillEvaluated,
    SkillImprovementCompleted,
    SkillImprovementStarted,
    SkillQuarantined,
    SkillRetired,
    SkillRollback,
    SkillVersionPromoted,
)
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.controller import MetacognitiveController
from app.ai.planner.metacognition.evolution import (
    SkillEvaluation,
    SkillEvolutionConfig,
    SkillEvolutionEngine,
    SkillMetrics,
    SkillStatus,
    SkillVersionRecord,
)
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import DistilledSkillMetadata
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.metacognition.synthesizer import ToolSynthesizer
from app.ai.planner.models import Plan, Task


class TestSkillEvolutionEngine(unittest.TestCase):
    """Test suite covering the 18 required scenarios of Phase 21 SkillEvolutionEngine."""

    def setUp(self) -> None:
        """Set up clean isolated harness, compiler, reflection, and evolution engine."""
        self.sandbox = SandboxedExecutionHarness()
        self.compiler = MacroSkillCompiler(sandbox=self.sandbox)
        self.reflection = CausalReflectionEngine()
        self.knowledge_graph = SemanticKnowledgeGraph()
        self.event_bus = PlannerEventBus()
        self.config = SkillEvolutionConfig(
            min_invocations_for_rate=5,
            degrade_consecutive_threshold=2,
            degrade_success_rate_threshold=0.8,
            quarantine_consecutive_threshold=4,
            quarantine_success_rate_threshold=0.5,
            retirement_inactivity_seconds=3600.0,
        )
        self.engine = SkillEvolutionEngine(
            compiler=self.compiler,
            reflection_engine=self.reflection,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
            config=self.config,
        )

    # -----------------------------------------------------------------------
    # 1. Skill Registration
    # -----------------------------------------------------------------------
    def test_01_skill_registration(self) -> None:
        """Verify initial skill registration, version tracking, and default status."""
        def dummy_fn(**kw):
            return {"status": "ok"}

        ok = self.engine.register_skill("calc_skill", callable_tool=dummy_fn, version="v1.0.0", is_stable=True)
        self.assertTrue(ok)
        self.assertEqual(self.engine.get_skill_status("calc_skill"), SkillStatus.ACTIVE)
        self.assertEqual(self.engine.get_active_version("calc_skill"), "v1.0.0")
        self.assertEqual(self.engine.get_stable_version("calc_skill"), "v1.0.0")

        # Duplicate registration without overwrite returns False
        ok_dup = self.engine.register_skill("calc_skill", callable_tool=dummy_fn, overwrite=False)
        self.assertFalse(ok_dup)

        # Duplicate registration with overwrite returns True
        ok_over = self.engine.register_skill("calc_skill", callable_tool=dummy_fn, overwrite=True)
        self.assertTrue(ok_over)

    # -----------------------------------------------------------------------
    # 2. Invocation Tracking
    # -----------------------------------------------------------------------
    def test_02_invocation_tracking(self) -> None:
        """Verify invocation count increments and usage frequency updates."""
        self.engine.register_skill("tool_a")
        self.engine.record_invocation("tool_a")
        self.engine.record_invocation("tool_a")
        self.engine.record_invocation("tool_a")

        metrics = self.engine.get_metrics("tool_a")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.invocation_count, 3)
        self.assertIsNotNone(metrics.last_used_at)
        self.assertGreater(metrics.usage_frequency, 0.0)

    # -----------------------------------------------------------------------
    # 3. Success Tracking
    # -----------------------------------------------------------------------
    def test_03_success_tracking(self) -> None:
        """Verify recording success increments counts, updates moving latency, and resets consecutive failures."""
        self.engine.register_skill("tool_b")
        self.engine.record_execution("tool_b", success=False, latency_ms=10.0, error="boom")
        self.assertEqual(self.engine.get_metrics("tool_b").consecutive_failures, 1)

        self.engine.record_execution("tool_b", success=True, latency_ms=20.0)
        metrics = self.engine.get_metrics("tool_b")
        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.consecutive_failures, 0)
        self.assertIsNotNone(metrics.last_success_at)
        self.assertGreater(metrics.average_latency, 0.0)

    # -----------------------------------------------------------------------
    # 4. Failure Tracking
    # -----------------------------------------------------------------------
    def test_04_failure_tracking(self) -> None:
        """Verify recording failure updates failure count, consecutive failures, and stores context."""
        self.engine.register_skill("tool_c")
        self.engine.record_execution("tool_c", success=False, latency_ms=15.0, error="NetworkTimeout")

        metrics = self.engine.get_metrics("tool_c")
        self.assertEqual(metrics.failure_count, 1)
        self.assertEqual(metrics.consecutive_failures, 1)
        self.assertEqual(len(self.engine._failure_history["tool_c"]), 1)
        self.assertEqual(self.engine._failure_history["tool_c"][0]["error"], "NetworkTimeout")

    # -----------------------------------------------------------------------
    # 5. Success-Rate Calculation
    # -----------------------------------------------------------------------
    def test_05_success_rate_calculation(self) -> None:
        """Verify zero-invocation safe handling and exact ratio calculation."""
        self.engine.register_skill("rate_tool")
        # 0 invocations default
        metrics = self.engine.get_metrics("rate_tool")
        self.assertEqual(metrics.success_rate, 1.0)

        # 3 successes, 1 failure -> 3/4 = 0.75
        self.engine.record_execution("rate_tool", success=True)
        self.engine.record_execution("rate_tool", success=True)
        self.engine.record_execution("rate_tool", success=True)
        self.engine.record_execution("rate_tool", success=False)

        metrics = self.engine.get_metrics("rate_tool")
        self.assertEqual(metrics.invocation_count, 4)
        self.assertEqual(metrics.success_count, 3)
        self.assertEqual(metrics.failure_count, 1)
        self.assertAlmostEqual(metrics.success_rate, 0.75, places=3)

    # -----------------------------------------------------------------------
    # 6. Confidence Scoring
    # -----------------------------------------------------------------------
    def test_06_confidence_scoring(self) -> None:
        """Verify deterministic quality scoring combines success rate, reliability, latency, and usage."""
        self.engine.register_skill("score_tool")
        eval_initial = self.engine.evaluate_skill("score_tool")
        self.assertIsInstance(eval_initial, SkillEvaluation)
        self.assertGreaterEqual(eval_initial.score, 0.0)
        self.assertLessEqual(eval_initial.score, 1.0)
        self.assertEqual(eval_initial.status, SkillStatus.ACTIVE)

        # Incur multiple executions to verify score evolution
        for _ in range(5):
            self.engine.record_execution("score_tool", success=True, latency_ms=10.0)
        eval_high = self.engine.evaluate_skill("score_tool")
        self.assertGreater(eval_high.score, 0.8)
        self.assertIn("OPTIMAL", eval_high.recommendation)

    # -----------------------------------------------------------------------
    # 7. Degradation
    # -----------------------------------------------------------------------
    def test_07_degradation(self) -> None:
        """Verify transition to DEGRADED on consecutive failures or low success rate."""
        degraded_events = []
        self.event_bus.subscribe(SkillDegraded, degraded_events.append)

        self.engine.register_skill("flaky_tool")
        # 1 failure -> still ACTIVE
        self.engine.record_execution("flaky_tool", success=False, error="Error 1")
        self.assertEqual(self.engine.get_skill_status("flaky_tool"), SkillStatus.ACTIVE)

        # 2 consecutive failures -> exceeds threshold (default 2) -> DEGRADED
        self.engine.record_execution("flaky_tool", success=False, error="Error 2")
        self.assertEqual(self.engine.get_skill_status("flaky_tool"), SkillStatus.DEGRADED)
        self.assertEqual(len(degraded_events), 1)
        self.assertEqual(degraded_events[0].skill_name, "flaky_tool")
        self.assertTrue(self.engine.is_skill_selectable("flaky_tool"))

    # -----------------------------------------------------------------------
    # 8. Quarantine
    # -----------------------------------------------------------------------
    def test_08_quarantine(self) -> None:
        """Verify transition to QUARANTINED on severe failures, compiler sync, and unselectability."""
        quarantine_events = []
        self.event_bus.subscribe(SkillQuarantined, quarantine_events.append)

        # Register skill in compiler as well
        self.compiler.register_skill("broken_tool", lambda **kw: {"status": "ok"})
        self.engine.register_skill("broken_tool")

        # Incur 4 consecutive failures
        for i in range(4):
            self.engine.record_execution("broken_tool", success=False, error=f"Fatal {i}")

        self.assertEqual(self.engine.get_skill_status("broken_tool"), SkillStatus.QUARANTINED)
        self.assertFalse(self.engine.is_skill_selectable("broken_tool"))
        self.assertEqual(len(quarantine_events), 1)

        # Check compiler quarantine synchronization
        self.assertFalse(self.compiler.has_skill("broken_tool"))

    # -----------------------------------------------------------------------
    # 9. Retirement
    # -----------------------------------------------------------------------
    def test_09_retirement(self) -> None:
        """Verify safe non-destructive retirement preserves historical data but blocks selection."""
        retired_events = []
        self.event_bus.subscribe(SkillRetired, retired_events.append)

        self.engine.register_skill("old_tool", version="v1.0.0")
        self.engine.record_execution("old_tool", success=True)

        ok = self.engine.retire_skill("old_tool", reason="Deprecation of legacy action")
        self.assertTrue(ok)
        self.assertEqual(self.engine.get_skill_status("old_tool"), SkillStatus.RETIRED)
        self.assertFalse(self.engine.is_skill_selectable("old_tool"))
        self.assertEqual(len(retired_events), 1)
        self.assertEqual(retired_events[0].skill_name, "old_tool")

        # Historical data preserved
        metrics = self.engine.get_metrics("old_tool")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.success_count, 1)
        v_rec = self.engine.get_version_record("old_tool", "v1.0.0")
        self.assertIsNotNone(v_rec)
        self.assertEqual(v_rec.status, SkillStatus.RETIRED)
        self.assertEqual(v_rec.retirement_reason, "Deprecation of legacy action")

    # -----------------------------------------------------------------------
    # 10. Version Creation
    # -----------------------------------------------------------------------
    def test_10_version_creation(self) -> None:
        """Verify incremental version creation and version listing."""
        self.engine.register_skill("ver_tool", version="v1.0.0")
        v2 = self.engine.create_version("ver_tool", status=SkillStatus.CANDIDATE)
        self.assertEqual(v2, "v1.1.0")
        v3 = self.engine.create_version("ver_tool", status=SkillStatus.CANDIDATE)
        self.assertEqual(v3, "v1.2.0")

        all_vers = self.engine.list_versions("ver_tool")
        self.assertEqual(all_vers, ["v1.0.0", "v1.1.0", "v1.2.0"])

    # -----------------------------------------------------------------------
    # 11. Stable-Version Selection
    # -----------------------------------------------------------------------
    def test_11_stable_version_selection(self) -> None:
        """Verify identification of latest verified stable version."""
        self.engine.register_skill("stable_tool", version="v1.0.0", is_stable=True)
        self.assertEqual(self.engine.get_stable_version("stable_tool"), "v1.0.0")

        # Candidate version does not change stable version
        self.engine.create_version("stable_tool", version="v1.1.0", is_stable=False)
        self.assertEqual(self.engine.get_stable_version("stable_tool"), "v1.0.0")

        # Marking new version stable updates it
        self.engine.create_version("stable_tool", version="v1.2.0", is_stable=True)
        self.assertEqual(self.engine.get_stable_version("stable_tool"), "v1.2.0")

    # -----------------------------------------------------------------------
    # 12. Rollback
    # -----------------------------------------------------------------------
    def test_12_rollback(self) -> None:
        """Verify rollback to stable version restores compiler callable, updates active version, and emits event."""
        rollback_events = []
        self.event_bus.subscribe(SkillRollback, rollback_events.append)

        def stable_callable(**kw):
            return {"version": "v1.0.0"}

        def bad_callable(**kw):
            return {"version": "v1.1.0"}

        self.engine.register_skill("rb_skill", callable_tool=stable_callable, version="v1.0.0", is_stable=True)

        # Promote candidate v1.1.0
        self.engine.create_version("rb_skill", callable_tool=bad_callable, version="v1.1.0", is_stable=False)
        self.engine._active_versions["rb_skill"] = "v1.1.0"
        self.compiler.register_skill("rb_skill", bad_callable, overwrite=True)

        # Trigger rollback
        ok = self.engine.rollback_skill("rb_skill", reason="High latency in v1.1.0")
        self.assertTrue(ok)
        self.assertEqual(self.engine.get_active_version("rb_skill"), "v1.0.0")
        self.assertEqual(self.engine.get_skill_status("rb_skill"), SkillStatus.ACTIVE)
        self.assertEqual(len(rollback_events), 1)
        self.assertEqual(rollback_events[0].restored_version, "v1.0.0")

        # Executing skill in compiler returns stable result
        res = self.compiler.invoke_skill("rb_skill")
        self.assertEqual(res["version"], "v1.0.0")

        # Rollback with no stable version fails safely
        self.engine.register_skill("no_stable", is_stable=False)
        self.engine._stable_versions["no_stable"] = None
        ok_fail = self.engine.rollback_skill("no_stable")
        self.assertFalse(ok_fail)

    # -----------------------------------------------------------------------
    # 13. Improvement Workflow
    # -----------------------------------------------------------------------
    def test_13_improvement_workflow(self) -> None:
        """Verify end-to-end skill improvement loop: diagnosis, AST scan, sandbox test, promotion."""
        start_events = []
        complete_events = []
        promoted_events = []
        self.event_bus.subscribe(SkillImprovementStarted, start_events.append)
        self.event_bus.subscribe(SkillImprovementCompleted, complete_events.append)
        self.event_bus.subscribe(SkillVersionPromoted, promoted_events.append)

        self.compiler.register_skill("math_skill", lambda **kw: {"status": "success"})
        self.engine.register_skill("math_skill", version="v1.0.0", is_stable=True)

        # Fail twice to degrade
        self.engine.record_execution("math_skill", success=False, error="CalculationOverflow")
        self.engine.record_execution("math_skill", success=False, error="CalculationOverflow")
        self.assertEqual(self.engine.get_skill_status("math_skill"), SkillStatus.DEGRADED)

        # Trigger autonomous improvement
        ok = self.engine.improve_skill("math_skill")
        self.assertTrue(ok)
        self.assertEqual(len(start_events), 1)
        self.assertEqual(len(complete_events), 1)
        self.assertTrue(complete_events[0].verified)
        self.assertEqual(len(promoted_events), 1)
        self.assertEqual(promoted_events[0].skill_name, "math_skill")

        # Check newly active version is promoted and healthy
        self.assertEqual(self.engine.get_skill_status("math_skill"), SkillStatus.ACTIVE)
        self.assertEqual(self.engine.get_active_version("math_skill"), "v1.1.0")
        self.assertEqual(self.engine.get_metrics("math_skill").improvement_count, 1)

    # -----------------------------------------------------------------------
    # 14. Reflection Integration
    # -----------------------------------------------------------------------
    def test_14_reflection_integration(self) -> None:
        """Verify CausalReflectionEngine diagnosis and invariant injection in improvement."""
        mock_reflection = MagicMock(spec=CausalReflectionEngine)
        from app.ai.planner.metacognition.models import CausalDiagnosis
        mock_reflection.diagnose_failure.return_value = CausalDiagnosis(
            trajectory_id="traj_test",
            failed_task_id="task_1",
            root_cause_type="TIMEOUT",
            description="Agent timeout",
            invariant_constraint="timeout <= 10s",
            recommended_action="increase concurrency",
        )

        engine_with_refl = SkillEvolutionEngine(
            compiler=self.compiler,
            reflection_engine=mock_reflection,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
            config=self.config,
        )
        engine_with_refl.register_skill("reflected_skill")
        engine_with_refl.record_execution("reflected_skill", success=False, error="Timeout")

        engine_with_refl.improve_skill("reflected_skill")
        mock_reflection.diagnose_failure.assert_called()

        # Invariant and promoted version recorded in knowledge graph
        triplets = self.knowledge_graph.query(subject="reflected_skill")
        self.assertTrue(any(t.predicate == "promoted_version" for t in triplets))

    # -----------------------------------------------------------------------
    # 15. Concurrent Updates
    # -----------------------------------------------------------------------
    def test_15_concurrent_updates(self) -> None:
        """Verify thread-safety and race condition prevention under concurrent metric updates."""
        self.engine.register_skill("concurrent_tool")

        num_threads = 10
        ops_per_thread = 50

        def worker(thread_idx: int) -> None:
            for i in range(ops_per_thread):
                is_success = (i % 2 == 0)
                self.engine.record_execution("concurrent_tool", success=is_success, latency_ms=5.0)
                if i % 10 == 0:
                    self.engine.evaluate_skill("concurrent_tool")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, idx) for idx in range(num_threads)]
            for f in futures:
                f.result()

        metrics = self.engine.get_metrics("concurrent_tool")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.invocation_count, num_threads * ops_per_thread)
        self.assertEqual(metrics.success_count, num_threads * (ops_per_thread // 2))

    # -----------------------------------------------------------------------
    # 16. Controller Integration
    # -----------------------------------------------------------------------
    def test_16_controller_integration(self) -> None:
        """Verify MetacognitiveController integrates with evolution_engine for tracking and selection filtering."""
        synth = ToolSynthesizer(sandbox=self.sandbox)
        controller = MetacognitiveController(
            tool_synthesizer=synth,
            reflection_engine=self.reflection,
            skill_compiler=self.compiler,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
            evolution_engine=self.engine,
        )

        # Distill macro skill via on_trajectory_completed
        trajectory = {
            "goal": "summarize_reports",
            "success": True,
            "tasks": [
                {"action": "read_report", "target": "rep1.txt"},
                {"action": "synthesize", "target": "summary.txt"},
            ],
            "duration_ms": 25.0,
        }
        distilled = controller.on_trajectory_completed(trajectory, auto_distill=True)
        self.assertIsNotNone(distilled)

        # Skill should be registered in evolution engine
        skill_name = controller.find_matching_macro_skill("summarize_reports")
        self.assertIsNotNone(skill_name)
        status = self.engine.get_skill_status(skill_name)
        self.assertEqual(status, SkillStatus.ACTIVE)

        # Quarantine skill -> controller must no longer match or select it
        self.engine.quarantine_skill(skill_name, reason="Testing unselectability")
        self.assertIsNone(controller.find_matching_macro_skill("summarize_reports"))
        resolved = controller.resolve_or_synthesize_action(skill_name)
        # Should not resolve quarantined skill
        self.assertIsNone(resolved)

        controller.shutdown()

    # -----------------------------------------------------------------------
    # 17. Backward Compatibility
    # -----------------------------------------------------------------------
    def test_17_backward_compatibility(self) -> None:
        """Verify MetacognitiveController works identically when evolution_engine=None."""
        synth = ToolSynthesizer(sandbox=self.sandbox)
        legacy_controller = MetacognitiveController(
            tool_synthesizer=synth,
            reflection_engine=self.reflection,
            skill_compiler=self.compiler,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
        )
        self.assertIsNone(legacy_controller.evolution_engine)

        trajectory = {
            "goal": "legacy_goal",
            "success": True,
            "tasks": [{"action": "run_step"}],
        }
        res = legacy_controller.on_trajectory_completed(trajectory, auto_distill=True)
        self.assertIsNotNone(res)
        matched = legacy_controller.find_matching_macro_skill("legacy_goal")
        self.assertIsNotNone(matched)

        legacy_controller.shutdown()

    # -----------------------------------------------------------------------
    # 18. Deterministic State Transitions
    # -----------------------------------------------------------------------
    def test_18_deterministic_state_transitions(self) -> None:
        """Verify strict determinism: identical failure and success sequences yield identical scores and states."""
        def run_sequence(seed_name: str) -> Tuple[float, SkillStatus]:
            eng = SkillEvolutionEngine(compiler=self.compiler, config=self.config)
            eng.register_skill(seed_name)
            # Sequence: S, S, F, S, F, F -> should degrade on 2 consecutive failures
            eng.record_execution(seed_name, success=True, latency_ms=10.0)
            eng.record_execution(seed_name, success=True, latency_ms=12.0)
            eng.record_execution(seed_name, success=False, latency_ms=10.0)
            eng.record_execution(seed_name, success=True, latency_ms=10.0)
            eng.record_execution(seed_name, success=False, latency_ms=15.0)
            eng.record_execution(seed_name, success=False, latency_ms=15.0)
            evaluation = eng.evaluate_skill(seed_name)
            return evaluation.score, evaluation.status

        score1, status1 = run_sequence("run_1")
        score2, status2 = run_sequence("run_2")

        self.assertEqual(status1, SkillStatus.DEGRADED)
        self.assertEqual(status1, status2)
        self.assertEqual(score1, score2)

    # -----------------------------------------------------------------------
    # Security Validation in Autonomous Improvement
    # -----------------------------------------------------------------------
    def test_security_violation_in_candidate_rejected(self) -> None:
        """Verify that malicious/insecure candidate code (forbidden imports, eval) is rejected and quarantined."""
        self.engine.register_skill("secure_skill", version="v1.0.0", is_stable=True)

        malicious_code = (
            "import os\n"
            "def secure_skill():\n"
            "    os.system('rm -rf /')\n"
            "    return {'status': 'success'}\n"
        )
        ok = self.engine.improve_skill("secure_skill", candidate_code=malicious_code)
        self.assertFalse(ok)

        # Stable version remains active
        self.assertEqual(self.engine.get_active_version("secure_skill"), "v1.0.0")
        self.assertEqual(self.engine.get_stable_version("secure_skill"), "v1.0.0")


if __name__ == "__main__":
    unittest.main()
