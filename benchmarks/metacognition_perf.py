"""Phase 20.5 Master Metacognitive Integration Performance Benchmarks.

Measures latency and throughput metrics against Phase 20 architectural performance budgets:
- Controller Overhead (Target: < 1.0 ms)
- Causal Reflection Latency (Target: < 2.0 ms)
- Knowledge Graph Update Latency (Target: < 100.0 µs)
- Macro Skill Dispatch Latency (Target: < 5.0 ms)
- End-to-End Execution & Macro Skill Speedup
"""

import logging
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.abspath("."))

# Silence debug/info logs during benchmark
logging.getLogger("app.ai.planner").setLevel(logging.WARNING)

from app.ai.planner.events import PlannerEventBus
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.controller import MetacognitiveController
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import DistilledSkillMetadata
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.registry import DynamicToolRegistry
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.metacognition.synthesizer import ToolSynthesizer
from app.ai.planner.swarm.coordinator import HierarchicalCoordinator


def setup_bench_controller() -> MetacognitiveController:
    """Build an isolated MetacognitiveController for microbenchmarking."""
    event_bus = PlannerEventBus()
    sandbox = SandboxedExecutionHarness(timeout_seconds=5.0)
    registry = DynamicToolRegistry()
    synthesizer = ToolSynthesizer(sandbox=sandbox, registry=registry, event_bus=event_bus)
    reflection = CausalReflectionEngine(event_bus=event_bus)
    compiler = MacroSkillCompiler(sandbox=sandbox, event_bus=event_bus)
    kg = SemanticKnowledgeGraph()

    # Pre-register a test skill
    meta = DistilledSkillMetadata(
        skill_name="macro_bench_skill",
        source_subplan_signature="bench_sig",
        compiled_code="",
        historical_avg_latency_ms=25.0,
        compiled_latency_ms=0.5,
        speedup_multiplier=50.0,
        verified=True,
    )
    compiler.register_skill("macro_bench_skill", lambda **kw: {"status": "ok", "val": 42}, metadata=meta)
    kg.add_relation("bench goal", "distilled_as_macro_skill", "macro_bench_skill")

    return MetacognitiveController(
        tool_synthesizer=synthesizer,
        reflection_engine=reflection,
        skill_compiler=compiler,
        knowledge_graph=kg,
        event_bus=event_bus,
        executor_handlers={"cached_handler": lambda **kw: "ok"},
    )


def benchmark_controller_overhead(ctrl: MetacognitiveController, iterations: int = 2000) -> float:
    """Measure resolution and dispatch overhead of controller in milliseconds (< 1 ms budget)."""
    # Warmup
    for _ in range(50):
        ctrl.resolve_or_synthesize_action("cached_handler")

    t0 = time.perf_counter()
    for _ in range(iterations):
        handler = ctrl.resolve_or_synthesize_action("cached_handler")
        assert handler is not None
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_reflection_latency(ctrl: MetacognitiveController, iterations: int = 500) -> float:
    """Measure causal reflection diagnosis latency in milliseconds (< 2 ms budget)."""
    sample_traj = {
        "trajectory_id": "bench_traj",
        "error": "TimeoutError: Swarm worker timed out",
        "events": ["start", "timeout"],
    }
    # Warmup
    for _ in range(20):
        ctrl.reflection_engine.diagnose_failure(sample_traj)

    t0 = time.perf_counter()
    for _ in range(iterations):
        diag = ctrl.reflection_engine.diagnose_failure(sample_traj)
        assert diag.root_cause_type == "EXECUTION_TIMEOUT"
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_knowledge_graph_update(ctrl: MetacognitiveController, iterations: int = 5000) -> float:
    """Measure knowledge graph update latency in microseconds (< 100 µs budget)."""
    # Warmup
    for i in range(50):
        ctrl.knowledge_graph.add_relation(f"s_{i}", "rel", f"o_{i}")

    t0 = time.perf_counter()
    for i in range(iterations):
        ctrl.knowledge_graph.add_relation(f"subj_{i}", "executed", f"obj_{i}", confidence=0.95)
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    return avg_us


def benchmark_macro_skill_dispatch(ctrl: MetacognitiveController, iterations: int = 2000) -> float:
    """Measure compiled macro skill lookup and dispatch latency in milliseconds (< 5 ms budget)."""
    # Warmup
    for _ in range(50):
        ctrl.invoke_macro_skill("macro_bench_skill")

    t0 = time.perf_counter()
    for _ in range(iterations):
        matching = ctrl.find_matching_macro_skill("bench goal")
        assert matching == "macro_bench_skill"
        res = ctrl.invoke_macro_skill(matching)
        assert res["val"] == 42
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_end_to_end_speedup(ctrl: MetacognitiveController) -> Dict[str, float]:
    """Compare hierarchical swarm tier execution vs macro-skill fast path execution."""
    coord = HierarchicalCoordinator(metacognitive_controller=ctrl)

    # 1. Macro skill fast path execution
    t0 = time.perf_counter()
    fast_res = coord.execute_goal("bench goal")
    fast_duration_ms = (time.perf_counter() - t0) * 1000.0
    assert fast_res.success
    assert fast_res.metrics.get("tier") == "macro_skill_fast_path"

    # 2. Standard fallback execution
    t0 = time.perf_counter()
    std_res = coord.execute_goal("open browser")
    std_duration_ms = (time.perf_counter() - t0) * 1000.0
    assert std_res.success

    speedup = max(1.0, std_duration_ms / max(0.001, fast_duration_ms))
    coord.shutdown()

    return {
        "fast_path_ms": fast_duration_ms,
        "standard_path_ms": std_duration_ms,
        "speedup_multiplier": speedup,
    }


def main() -> None:
    print("=" * 68)
    print("      J.A.R.V.I.S. Phase 20.5 Master Metacognition Benchmark        ")
    print("=" * 68)
    print()

    ctrl = setup_bench_controller()

    # 1. Controller Overhead
    print("[1/5] Benchmarking Controller Overhead (2,000 runs)...")
    ctrl_ms = benchmark_controller_overhead(ctrl, iterations=2000)
    ctrl_budget = 1.0
    ctrl_pass = ctrl_ms < ctrl_budget
    ctrl_status = "PASS" if ctrl_pass else "FAIL"
    print(f"  Result: {ctrl_ms:.4f} ms (Budget: < {ctrl_budget} ms) -> {ctrl_status}\n")

    # 2. Reflection Latency
    print("[2/5] Benchmarking Causal Reflection Latency (500 runs)...")
    refl_ms = benchmark_reflection_latency(ctrl, iterations=500)
    refl_budget = 2.0
    refl_pass = refl_ms < refl_budget
    refl_status = "PASS" if refl_pass else "FAIL"
    print(f"  Result: {refl_ms:.4f} ms (Budget: < {refl_budget} ms) -> {refl_status}\n")

    # 3. Knowledge Graph Update Latency
    print("[3/5] Benchmarking Knowledge Graph Update Latency (5,000 updates)...")
    kg_us = benchmark_knowledge_graph_update(ctrl, iterations=5000)
    kg_budget = 100.0
    kg_pass = kg_us < kg_budget
    kg_status = "PASS" if kg_pass else "FAIL"
    print(f"  Result: {kg_us:.4f} µs (Budget: < {kg_budget} µs) -> {kg_status}\n")

    # 4. Macro Skill Dispatch Latency
    print("[4/5] Benchmarking Macro Skill Dispatch Latency (2,000 runs)...")
    skill_ms = benchmark_macro_skill_dispatch(ctrl, iterations=2000)
    skill_budget = 5.0
    skill_pass = skill_ms < skill_budget
    skill_status = "PASS" if skill_pass else "FAIL"
    print(f"  Result: {skill_ms:.4f} ms (Budget: < {skill_budget} ms) -> {skill_status}\n")

    # 5. End-to-End Speedup
    print("[5/5] Benchmarking End-to-End Execution & Macro Skill Speedup...")
    e2e = benchmark_end_to_end_speedup(ctrl)
    print(f"  Macro Skill Fast Path:  {e2e['fast_path_ms']:.4f} ms")
    print(f"  Standard Fallback Tier: {e2e['standard_path_ms']:.4f} ms")
    print(f"  Observed Speedup:       {e2e['speedup_multiplier']:.1f}x\n")

    ctrl.shutdown()

    all_passed = ctrl_pass and refl_pass and kg_pass and skill_pass

    print("=" * 68)
    print("                     BENCHMARK SUMMARY                           ")
    print("=" * 68)
    print(f"Controller Overhead:     {ctrl_ms:.4f} ms       (Budget: < {ctrl_budget} ms)       [{ctrl_status}]")
    print(f"Causal Reflection:       {refl_ms:.4f} ms       (Budget: < {refl_budget} ms)       [{refl_status}]")
    print(f"Knowledge Graph Update:  {kg_us:.4f} µs       (Budget: < {kg_budget} µs)     [{kg_status}]")
    print(f"Macro Skill Dispatch:    {skill_ms:.4f} ms       (Budget: < {skill_budget} ms)       [{skill_status}]")
    print(f"Macro Skill Speedup:     {e2e['speedup_multiplier']:.1f}x speedup")
    print("=" * 68)
    if all_passed:
        print("OVERALL STATUS: ALL PERFORMANCE BUDGETS MET [PASS]")
    else:
        print("OVERALL STATUS: AT LEAST ONE BUDGET FAILED [FAIL]")
    print("=" * 68)


if __name__ == "__main__":
    main()
