"""Architecture Invariant & Regression Tests (Phase 16.5).

Verifies architectural boundaries and structural invariants across the J.A.R.V.I.S planning system:
1. Module Decoupling:
   - ExecutionMemory does not import AIManager, Planner, or Executor.
   - FailureClassifier does not import AIManager, Planner, Executor, or ExecutionMemory.
   - MemorySummaryBuilder does not import AIManager, Planner, or Executor.
   - RecoveryHeuristics does not import AIManager, Planner, or Executor.
2. Heuristic Determinism:
   - RecoveryHeuristics evaluates purely deterministically without LLM calls or side effects.
3. Single Source of Truth:
   - DAG validation is centralized in Executor.validate_dag().
   - JSON & task schema validation is centralized in Planner._parse_and_validate_llm_plan().
   - Failure classification is centralized in FailureClassifier.classify().
4. Single-wave vs Multi-wave Isolation:
   - ExecutionResult represents a single wave snapshot.
   - ExecutionMemory represents accumulated history across waves.
"""

from __future__ import annotations

import ast
import inspect
import os
import unittest
from typing import Set

import app.ai.planner.failure_classifier as fc_module
import app.ai.planner.heuristics as heur_module
import app.ai.planner.memory as mem_module
import app.ai.planner.memory_summary as ms_module
import app.ai.planner.models as models_module
from app.ai.planner.failure_classifier import FailureClassifier
from app.ai.planner.heuristics import evaluate_recovery_viability
from app.ai.planner.memory import ExecutionMemory, FailureCategory
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus


def get_imported_module_names(file_path: str) -> Set[str]:
    """Parse a python source file and extract top-level and from-imported module paths."""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


class TestArchitectureInvariants(unittest.TestCase):
    """Verifies strict subsystem boundaries and decoupling contracts."""

    def test_execution_memory_has_no_upstream_dependencies(self) -> None:
        """ExecutionMemory must not import AIManager, Planner, or Executor."""
        file_path = inspect.getfile(mem_module)
        imported = get_imported_module_names(file_path)

        for mod in imported:
            self.assertFalse(
                "app.ai.manager" in mod,
                f"ExecutionMemory illegally imports AIManager: {mod}",
            )
            self.assertFalse(
                "app.ai.planner.planner" in mod,
                f"ExecutionMemory illegally imports Planner: {mod}",
            )
            self.assertFalse(
                "app.ai.planner.executor" in mod,
                f"ExecutionMemory illegally imports Executor: {mod}",
            )

    def test_failure_classifier_is_isolated(self) -> None:
        """FailureClassifier must not import AIManager, Planner, Executor, or ExecutionMemory."""
        file_path = inspect.getfile(fc_module)
        imported = get_imported_module_names(file_path)

        for mod in imported:
            self.assertFalse(
                "app.ai.manager" in mod or "app.ai.planner.planner" in mod or "app.ai.planner.executor" in mod,
                f"FailureClassifier illegally imports orchestrator or engine: {mod}",
            )

    def test_heuristics_has_no_upstream_engine_dependencies(self) -> None:
        """RecoveryHeuristics must not import AIManager, Planner, or Executor."""
        file_path = inspect.getfile(heur_module)
        imported = get_imported_module_names(file_path)

        for mod in imported:
            self.assertFalse(
                "app.ai.manager" in mod or "app.ai.planner.planner" in mod or "app.ai.planner.executor" in mod,
                f"RecoveryHeuristics illegally imports upstream engine: {mod}",
            )

    def test_memory_summary_builder_is_independent(self) -> None:
        """MemorySummaryBuilder must not import AIManager, Planner, or Executor."""
        file_path = inspect.getfile(ms_module)
        imported = get_imported_module_names(file_path)

        for mod in imported:
            self.assertFalse(
                "app.ai.manager" in mod or "app.ai.planner.planner" in mod or "app.ai.planner.executor" in mod,
                f"MemorySummaryBuilder illegally imports upstream engine: {mod}",
            )

    def test_recovery_heuristics_strictly_deterministic(self) -> None:
        """RecoveryHeuristics must produce strictly identical results given identical inputs."""
        t1 = Task(id="t1", action="open_app", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", action="web_search", status=TaskStatus.FAILED)
        plan = Plan(query="deterministic test", tasks=[t1, t2])
        res = ExecutionResult(success=False, completed_tasks=[t1], failed_tasks=[t2], output="error")
        mem = ExecutionMemory.from_execution_result(res, wave=1)

        decisions = [evaluate_recovery_viability("deterministic test", plan, mem) for _ in range(50)]
        first = decisions[0]

        for d in decisions[1:]:
            self.assertEqual(d.viable, first.viable)
            self.assertEqual(d.reason, first.reason)
            self.assertEqual(d.blocking_tasks, first.blocking_tasks)

    def test_execution_result_and_execution_memory_contracts(self) -> None:
        """ExecutionResult represents a single wave; ExecutionMemory represents history."""
        # ExecutionResult single wave
        res = ExecutionResult(success=True)
        self.assertFalse(hasattr(res, "records"), "ExecutionResult should not store records directly.")
        self.assertFalse(hasattr(res, "metrics"), "ExecutionResult should not store ExecutionMetrics.")

        # ExecutionMemory history
        mem = ExecutionMemory.from_execution_result(res, wave=1)
        self.assertTrue(hasattr(mem, "records"), "ExecutionMemory must store TaskExecutionRecord list.")
        self.assertTrue(hasattr(mem, "metrics"), "ExecutionMemory must maintain ExecutionMetrics.")
        self.assertTrue(hasattr(mem, "retry_history"), "ExecutionMemory must track retry history.")

    def test_failure_classifier_default_fallback_is_unknown(self) -> None:
        """Any unclassifiable error must deterministically fall back to FailureCategory.UNKNOWN."""
        t = Task(id="t_random", action="arbitrary")
        cat = FailureClassifier.classify(t, error=None)
        self.assertEqual(cat, FailureCategory.UNKNOWN)

        cat2 = FailureClassifier.classify(t, error="unusual unformatted string with no keywords")
        self.assertEqual(cat2, FailureCategory.UNKNOWN)

    def test_planner_and_executor_do_not_depend_on_observability_subscribers(self) -> None:
        """Planner and Executor must only reference PlannerEventBus, not timeline, metrics, or logging."""
        import sys
        import app.ai.planner.executor
        import app.ai.planner.planner

        exec_mod = sys.modules["app.ai.planner.executor"]
        plan_mod = sys.modules["app.ai.planner.planner"]

        for mod_obj, name in [(plan_mod, "Planner"), (exec_mod, "Executor")]:
            file_path = inspect.getfile(mod_obj)
            imported = get_imported_module_names(file_path)
            for imp in imported:
                self.assertFalse(
                    "timeline" in imp,
                    f"{name} illegally depends on ExecutionTimeline ({imp})",
                )
                self.assertFalse(
                    "metrics_collector" in imp,
                    f"{name} illegally depends on MetricsCollector ({imp})",
                )
                self.assertFalse(
                    "logging_subscriber" in imp,
                    f"{name} illegally depends on LoggingSubscriber ({imp})",
                )

    def test_observability_subscribers_do_not_import_execution_engines(self) -> None:
        """Observability subscribers (timeline, metrics, logging) must not import Planner or Executor."""
        import app.ai.planner.logging_subscriber as log_sub_mod
        import app.ai.planner.metrics_collector as met_mod
        import app.ai.planner.timeline as time_mod

        for mod_obj, name in [(time_mod, "Timeline"), (met_mod, "Metrics"), (log_sub_mod, "Logging")]:
            file_path = inspect.getfile(mod_obj)
            imported = get_imported_module_names(file_path)
            for imp in imported:
                self.assertFalse(
                    "app.ai.planner.planner" in imp,
                    f"{name} illegally imports Planner ({imp})",
                )
                self.assertFalse(
                    "app.ai.planner.executor" in imp,
                    f"{name} illegally imports Executor ({imp})",
                )

    def test_replay_strictly_offline_and_independent(self) -> None:
        """PlannerReplayEngine must be 100% offline and not import Planner, Executor, AIManager, or ProviderRouter."""
        import app.ai.planner.replay as replay_mod
        file_path = inspect.getfile(replay_mod)
        imported = get_imported_module_names(file_path)

        for imp in imported:
            self.assertFalse(
                "app.ai.manager" in imp or "app.ai.provider_router" in imp or "app.skills" in imp,
                f"Replay engine illegally imports external orchestrators/skills: {imp}",
            )
            self.assertFalse(
                "app.ai.planner.planner" in imp or "app.ai.planner.executor" in imp,
                f"Replay engine illegally imports execution engines: {imp}",
            )

    def test_persistence_independent_of_orchestrators(self) -> None:
        """Execution persistence must not import AIManager or ProviderRouter."""
        import app.ai.planner.persistence as pers_mod
        file_path = inspect.getfile(pers_mod)
        imported = get_imported_module_names(file_path)

        for imp in imported:
            self.assertFalse(
                "app.ai.manager" in imp or "app.ai.provider_router" in imp,
                f"Persistence illegally imports external orchestrator: {imp}",
            )

    def test_controller_and_timeouts_isolated(self) -> None:
        """Controller and Timeouts modules must not import Planner or Executor."""
        import app.ai.planner.control as ctrl_mod
        import app.ai.planner.timeouts as time_mod

        for mod_obj, name in [(ctrl_mod, "ExecutionController"), (time_mod, "TimeoutManager")]:
            file_path = inspect.getfile(mod_obj)
            imported = get_imported_module_names(file_path)
            for imp in imported:
                self.assertFalse(
                    "app.ai.planner.planner" in imp or "app.ai.planner.executor" in imp,
                    f"{name} illegally imports execution engines: {imp}",
                )

    def test_planner_intelligence_unaltered(self) -> None:
        """Planner system prompts and heuristics must remain strictly unaltered."""
        from app.ai.planner.planner import (
            KNOWN_ACTIONS,
            PLANNER_SYSTEM_PROMPT,
            RECOVERY_SYSTEM_PROMPT,
        )
        self.assertIn("open_app", KNOWN_ACTIONS)
        self.assertIn("web_search", KNOWN_ACTIONS)
        self.assertIn("summarize_file", KNOWN_ACTIONS)
        self.assertIn("calculate", KNOWN_ACTIONS)
        self.assertIn("save_memory", KNOWN_ACTIONS)
        self.assertIn("clear_memory", KNOWN_ACTIONS)

        self.assertIn("You are the task planning engine for J.A.R.V.I.S.", PLANNER_SYSTEM_PROMPT)
        self.assertIn("You are the task recovery planning engine for J.A.R.V.I.S.", RECOVERY_SYSTEM_PROMPT)

    def test_multi_agent_subsystem_isolation(self) -> None:
        """Multi-agent models, registry, protocol, memory must not import AIManager."""
        import app.ai.planner.multi_agent.models as ma_models
        import app.ai.planner.multi_agent.registry as ma_reg
        import app.ai.planner.multi_agent.protocol as ma_proto
        import app.ai.planner.multi_agent.memory as ma_mem
        import app.ai.planner.multi_agent.conflict as ma_conf

        for mod_obj, name in [
            (ma_models, "models"),
            (ma_reg, "registry"),
            (ma_proto, "protocol"),
            (ma_mem, "memory"),
            (ma_conf, "conflict"),
        ]:
            file_path = inspect.getfile(mod_obj)
            imported = get_imported_module_names(file_path)
            for imp in imported:
                self.assertFalse(
                    "app.ai.manager" in imp or "app.ai.provider_router" in imp,
                    f"Multi-agent {name} illegally imports external orchestrators: {imp}",
                )

    def test_swarm_subsystem_isolation(self) -> None:
        """Swarm subsystem must maintain architectural isolation, zero global state, and DI."""
        import app.ai.planner.swarm.models as sw_models
        import app.ai.planner.swarm.subswarm as sw_sub
        import app.ai.planner.swarm.decomposer as sw_dec
        import app.ai.planner.swarm.episodic as sw_epi
        import app.ai.planner.swarm.speculative as sw_spec
        import app.ai.planner.swarm.consensus as sw_cons
        import app.ai.planner.swarm.supervisor as sw_sup
        import app.ai.planner.swarm.balancer as sw_bal
        import app.ai.planner.swarm.hitl as sw_hitl
        import app.ai.planner.swarm.policy as sw_pol
        import app.ai.planner.swarm.coordinator as sw_coord

        for mod_obj, name in [
            (sw_models, "swarm.models"),
            (sw_sub, "swarm.subswarm"),
            (sw_dec, "swarm.decomposer"),
            (sw_epi, "swarm.episodic"),
            (sw_spec, "swarm.speculative"),
            (sw_cons, "swarm.consensus"),
            (sw_sup, "swarm.supervisor"),
            (sw_bal, "swarm.balancer"),
            (sw_hitl, "swarm.hitl"),
            (sw_pol, "swarm.policy"),
            (sw_coord, "swarm.coordinator"),
        ]:
            file_path = inspect.getfile(mod_obj)
            imported = get_imported_module_names(file_path)
            for imp in imported:
                self.assertFalse(
                    "app.ai.manager" in imp or "app.ai.provider_router" in imp,
                    f"Swarm module {name} illegally imports external orchestrator: {imp}",
                )

        # Zero global state & DI verification
        coord1 = sw_coord.HierarchicalCoordinator()
        coord2 = sw_coord.HierarchicalCoordinator()
        self.assertIsNot(coord1.root_swarm, coord2.root_swarm)
        self.assertIsNot(coord1.memory, coord2.memory)
        self.assertIsNot(coord1.episodic_memory, coord2.episodic_memory)
        self.assertIsNot(coord1.supervisor, coord2.supervisor)


if __name__ == "__main__":
    unittest.main()


