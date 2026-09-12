"""Unit Tests for Phase 20.5 Master Metacognitive Controller Integration.

Validates complete dependency injection, synthesized tool resolution, macro-skill fast path,
hierarchical swarm fallback, knowledge graph updates, reflection execution, background execution,
event publication, backward compatibility, and concurrent execution.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import unittest
from typing import Any, Dict, List

from app.ai.planner.events import (
    CausalDiagnosisGenerated,
    KnowledgeGraphUpdated,
    PlannerEvent,
    PlannerEventBus,
    SkillDistillationCompleted,
)
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.controller import MetacognitiveController
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import DistilledSkillMetadata
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.registry import DynamicToolRegistry
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.metacognition.synthesizer import SyntheticTestGenerator, ToolSynthesizer
from app.ai.planner.multi_agent.base import WorkerAgent
from app.ai.planner.multi_agent.models import AgentCapability, AgentManifest, AgentRole
from app.ai.planner.swarm.coordinator import HierarchicalCoordinator
from app.ai.planner.swarm.episodic import EpisodicMemoryStore, TrajectoryRecord


class TestMetacognitiveController(unittest.TestCase):
    """Test suite for MetacognitiveController orchestration layer."""

    def setUp(self) -> None:
        self.event_bus = PlannerEventBus()
        self.events: List[Any] = []
        self.event_bus.subscribe(PlannerEvent, lambda e: self.events.append(e))

        self.sandbox = SandboxedExecutionHarness(timeout_seconds=5.0)
        self.tool_registry = DynamicToolRegistry()
        self.synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            registry=self.tool_registry,
            event_bus=self.event_bus,
        )
        self.reflection_engine = CausalReflectionEngine(event_bus=self.event_bus)
        self.skill_compiler = MacroSkillCompiler(sandbox=self.sandbox, event_bus=self.event_bus)
        self.knowledge_graph = SemanticKnowledgeGraph()
        self.episodic_memory = EpisodicMemoryStore()

        self.controller = MetacognitiveController(
            tool_synthesizer=self.synthesizer,
            reflection_engine=self.reflection_engine,
            skill_compiler=self.skill_compiler,
            knowledge_graph=self.knowledge_graph,
            event_bus=self.event_bus,
            episodic_memory=self.episodic_memory,
            executor_handlers={"existing_action": lambda **kw: "handled"},
        )

    def tearDown(self) -> None:
        self.controller.shutdown()

    def test_dependency_injection_isolation(self) -> None:
        """Verify complete isolation between multiple MetacognitiveController instances."""
        ctrl2 = MetacognitiveController(
            tool_synthesizer=ToolSynthesizer(sandbox=SandboxedExecutionHarness()),
            reflection_engine=CausalReflectionEngine(),
            skill_compiler=MacroSkillCompiler(sandbox=SandboxedExecutionHarness()),
            knowledge_graph=SemanticKnowledgeGraph(),
        )
        self.assertIsNot(self.controller.knowledge_graph, ctrl2.knowledge_graph)
        self.assertIsNot(self.controller.tool_synthesizer, ctrl2.tool_synthesizer)
        self.assertIsNot(self.controller.reflection_engine, ctrl2.reflection_engine)
        self.assertIsNot(self.controller.skill_compiler, ctrl2.skill_compiler)
        self.assertIsNot(self.controller._lock, ctrl2._lock)
        ctrl2.shutdown()

    def test_synthesized_tool_resolution_executor_hit(self) -> None:
        """Pre-registered executor handlers take precedence before synthesis."""
        handler = self.controller.resolve_or_synthesize_action("existing_action")
        self.assertIsNotNone(handler)
        self.assertEqual(handler(), "handled")

    def test_synthesized_tool_resolution_dynamic_registry_hit(self) -> None:
        """Actions already present in dynamic tool registry are returned directly."""
        self.tool_registry.register("cached_calc", lambda x: x * 2)
        handler = self.controller.resolve_or_synthesize_action("cached_calc")
        self.assertIsNotNone(handler)
        self.assertEqual(handler(5), 10)

    def test_synthesized_tool_resolution_compiler_hit(self) -> None:
        """Actions already present in macro skill compiler are returned directly."""
        meta = DistilledSkillMetadata(
            skill_name="compiled_step",
            source_subplan_signature="sig",
            compiled_code="",
            historical_avg_latency_ms=10.0,
            compiled_latency_ms=1.0,
            speedup_multiplier=10.0,
            verified=True,
        )
        self.skill_compiler.register_skill("compiled_step", lambda **kw: {"res": "compiled"}, metadata=meta)
        handler = self.controller.resolve_or_synthesize_action("compiled_step")
        self.assertIsNotNone(handler)
        self.assertEqual(handler()["res"], "compiled")

    def test_synthesized_tool_resolution_on_demand(self) -> None:
        """Missing tools are synthesized, verified, and hot-bound into the registry."""
        handler = self.controller.resolve_or_synthesize_action(
            action_name="add_numbers",
            context={
                "description": "Compute arithmetic sum of two numbers",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
        )
        self.assertIsNotNone(handler)
        self.assertEqual(handler(3, 7), 10)
        self.assertTrue(self.knowledge_graph.contains("system", "has_synthesized_tool", "add_numbers"))

    def test_synthesized_tool_resolution_graceful_failure(self) -> None:
        """Dangerous or failing synthesis returns None gracefully and never raises exceptions."""
        bad_synth = ToolSynthesizer(
            sandbox=self.sandbox,
            code_generator=lambda action, ctx: f"def {action}(x):\n    return eval(x)\n",
            registry=self.tool_registry,
        )
        fail_ctrl = MetacognitiveController(
            tool_synthesizer=bad_synth,
            reflection_engine=self.reflection_engine,
            skill_compiler=self.skill_compiler,
            knowledge_graph=self.knowledge_graph,
        )
        handler = fail_ctrl.resolve_or_synthesize_action(
            action_name="danger_action",
            context={
                "description": "Malicious action using eval()",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
        )
        self.assertIsNone(handler)
        fail_ctrl.shutdown()

    def test_macro_skill_fast_path(self) -> None:
        """HierarchicalCoordinator routes to compiled macro skill fast path, bypassing swarm decomposition."""
        # Compile and promote a macro skill
        subplan = {
            "tasks": [
                {"action": "lookup_data", "code": "def run(): pass"},
                {"action": "format_data", "code": "def run(): pass"},
            ]
        }
        self.skill_compiler.compile_subplan_to_skill(
            subplan=subplan,
            skill_name="macro_summarize_sales",
            description="Macro skill for summarize sales",
        )
        self.knowledge_graph.add_relation("summarize sales", "distilled_as_macro_skill", "macro_summarize_sales")

        coord = HierarchicalCoordinator(
            event_bus=self.event_bus,
            metacognitive_controller=self.controller,
        )

        t0 = time.perf_counter()
        result = coord.execute_goal("summarize sales")
        elapsed = time.perf_counter() - t0

        self.assertTrue(result.success)
        self.assertEqual(result.metrics.get("tier"), "macro_skill_fast_path")
        self.assertIn("macro_skill", result.outputs)
        self.assertLess(elapsed, 0.05)  # Fast path must be under 50ms
        coord.shutdown()

    def test_swarm_fallback_when_no_macro_skill(self) -> None:
        """If no matching macro skill exists, HierarchicalCoordinator executes through swarm hierarchy."""
        coord = HierarchicalCoordinator(
            event_bus=self.event_bus,
            metacognitive_controller=self.controller,
        )
        # Register workers
        w1 = WorkerAgent(
            manifest=AgentManifest(
                agent_id="w_res",
                role=AgentRole.WORKER,
                capabilities=[AgentCapability(name="cap_res", supported_actions={"research"})],
            ),
            handlers={"research": lambda t: "Researched"},
        )
        w2 = WorkerAgent(
            manifest=AgentManifest(
                agent_id="w_syn",
                role=AgentRole.WORKER,
                capabilities=[AgentCapability(name="cap_syn", supported_actions={"synthesize"})],
            ),
            handlers={"synthesize": lambda t: "Synthesized"},
        )
        coord.registry.register(w1)
        coord.registry.register(w2)

        res = coord.execute_goal("Analyze blockchain scalability")
        self.assertTrue(res.success)
        self.assertNotEqual(res.metrics.get("tier"), "macro_skill_fast_path")
        coord.shutdown()

    def test_knowledge_graph_updates_on_success(self) -> None:
        """Successful trajectory inserts goal, tasks, artifacts into knowledge graph and emits event."""
        trajectory = {
            "goal": "Build dashboard",
            "success": True,
            "tasks": [{"action": "query_db"}, {"action": "render_chart"}],
            "outputs": {"chart_url": "http://charts/1"},
        }
        self.controller.on_trajectory_completed(trajectory, auto_distill=False)

        self.assertTrue(self.knowledge_graph.contains("system", "completed_goal", "Build dashboard"))
        self.assertTrue(self.knowledge_graph.contains("Build dashboard", "contains_task", "query_db"))
        self.assertTrue(self.knowledge_graph.contains("Build dashboard", "contains_task", "render_chart"))
        self.assertTrue(self.knowledge_graph.contains("Build dashboard", "produced_artifact", "chart_url:http://charts/1"))

        kg_events = [e for e in self.events if isinstance(e, KnowledgeGraphUpdated)]
        self.assertGreaterEqual(len(kg_events), 1)

    def test_knowledge_graph_updates_and_reflection_on_failure(self) -> None:
        """Failed trajectory triggers causal reflection, records invariants, and emits events."""
        trajectory = {
            "goal": "Fetch weather",
            "success": False,
            "error": "TimeoutError: Weather API call exceeded 30s deadline",
        }
        self.controller.on_trajectory_completed(trajectory, auto_distill=False)

        # Graph should reflect failure
        triplets = self.knowledge_graph.query(subject="Fetch weather", predicate="failed_due_to")
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0].object_, "EXECUTION_TIMEOUT")

        inv_triplets = self.knowledge_graph.query(subject="Fetch weather", predicate="violates_invariant")
        self.assertEqual(len(inv_triplets), 1)

        # Diagnostic event published
        diag_events = [e for e in self.events if isinstance(e, CausalDiagnosisGenerated)]
        self.assertEqual(len(diag_events), 1)
        self.assertEqual(diag_events[0].root_cause, "EXECUTION_TIMEOUT")

    def test_prescriptive_guidance_generation(self) -> None:
        """Guidance combines reflection patterns with historical knowledge graph constraints."""
        # Populate history
        self.controller.on_trajectory_completed(
            {"goal": "Process payment", "success": False, "error": "PolicyRejection: Forbidden transaction"},
            auto_distill=False,
        )

        guidance = self.controller.get_prescriptive_guidance("Process payment")
        self.assertIsInstance(guidance, list)
        self.assertGreater(len(guidance), 0)
        # Verify deterministic sorted list
        self.assertEqual(guidance, sorted(guidance))

    def test_auto_distillation_on_success(self) -> None:
        """Auto distillation compiles repeated subplans into verified macro skills."""
        trajectory = {
            "goal": "Process incoming records",
            "success": True,
            "tasks": [
                {"action": "parse_record", "code": "def run(): pass"},
                {"action": "validate_record", "code": "def run(): pass"},
            ],
            "outputs": {"records_processed": 50},
        }
        meta = self.controller.on_trajectory_completed(trajectory, auto_distill=True)
        self.assertIsNotNone(meta)
        self.assertTrue(meta.verified)

        # Skill compiler should have registered it
        skills = self.skill_compiler.list_skills()
        self.assertIn(meta.skill_name, skills)

        # Knowledge graph updated with distillation link
        self.assertTrue(self.knowledge_graph.contains("Process incoming records", "distilled_as_macro_skill", meta.skill_name))

        # SkillDistillationCompleted event emitted
        dist_events = [e for e in self.events if isinstance(e, SkillDistillationCompleted)]
        self.assertEqual(len(dist_events), 1)

    def test_background_reflection_execution(self) -> None:
        """Async trajectory completion executes in background thread pool without blocking caller."""
        trajectory = {
            "goal": "Background heavy task",
            "success": True,
            "tasks": [{"action": "bg_step"}],
            "outputs": {"status": "ok"},
        }
        future = self.controller.on_trajectory_completed_async(trajectory, auto_distill=False)
        self.assertIsInstance(future, concurrent.futures.Future)
        future.result(timeout=2.0)  # Wait for background completion

        self.assertTrue(self.knowledge_graph.contains("system", "completed_goal", "Background heavy task"))

    def test_backward_compatibility_without_controller(self) -> None:
        """HierarchicalCoordinator functions 100% normally when metacognitive_controller is omitted."""
        coord = HierarchicalCoordinator(event_bus=self.event_bus)
        self.assertIsNone(coord.metacognitive_controller)

        # Execute goal using single agent fallback
        res = coord.execute_goal("open browser")
        self.assertTrue(res.success)
        coord.shutdown()

    def test_concurrent_execution(self) -> None:
        """Thread safety verified under concurrent trajectory updates and tool queries."""
        def worker(worker_id: int) -> int:
            for i in range(10):
                self.controller.on_trajectory_completed(
                    {
                        "goal": f"Goal_{worker_id}_{i}",
                        "success": i % 2 == 0,
                        "error": "TimeoutError" if i % 2 != 0 else None,
                        "tasks": [{"action": f"task_{worker_id}_{i}"}],
                        "outputs": {"val": i},
                    },
                    auto_distill=False,
                )
                self.controller.resolve_or_synthesize_action("existing_action")
                self.controller.get_prescriptive_guidance(f"Goal_{worker_id}_{i}")
            return worker_id

        num_threads = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [f.result(timeout=5.0) for f in futures]
            self.assertEqual(len(results), num_threads)

    def test_episodic_memory_integration(self) -> None:
        """Completed trajectories are linked into EpisodicMemoryStore without data loss."""
        rec = TrajectoryRecord(
            goal="Deploy microservice",
            plan_signature="deploy_sig",
            success=True,
            total_duration_ms=45.0,
        )
        self.controller.on_trajectory_completed(rec, auto_distill=False)

        similar = self.episodic_memory.query_similar_goals("Deploy microservice", top_k=1)
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].goal, "Deploy microservice")


if __name__ == "__main__":
    unittest.main()
