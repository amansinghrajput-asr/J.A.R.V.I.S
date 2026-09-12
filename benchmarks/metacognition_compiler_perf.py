"""Performance Benchmarking Suite for Phase 20.4 Causal Reflection Engine & Macro-Skill Compiler.

Measures latency and throughput metrics against Phase 20 architectural performance budgets:
- Reflection Diagnosis Latency (Target: < 2.0 ms)
- Macro Skill Compilation Latency (Target: < 500.0 ms)
- Skill Lookup Latency (Target: < 10.0 µs)
- Skill Invocation Latency (Target: < 5.0 ms)
- Concurrent Skill Registry Throughput (Target: > 50,000 ops/sec)
"""

import concurrent.futures
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

# Silence debug/info logs during benchmark
logging.getLogger("app.ai.planner.metacognition").setLevel(logging.WARNING)

from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.models import DistilledSkillMetadata
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness


def benchmark_reflection_diagnosis(iterations: int = 500) -> float:
    """Measure single trajectory diagnosis latency in milliseconds."""
    engine = CausalReflectionEngine()
    sample_trajectory = {
        "trajectory_id": "traj_bench",
        "error": "TimeoutError: Swarm agent execution timed out after 30s",
        "events": ["plan_start", "agent_dispatch", "timeout_signal"],
    }

    # Warmup
    for _ in range(20):
        engine.diagnose_failure(sample_trajectory)

    t0 = time.perf_counter()
    for _ in range(iterations):
        diag = engine.diagnose_failure(sample_trajectory)
        assert diag.root_cause_type == "EXECUTION_TIMEOUT"
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_skill_compilation(iterations: int = 5) -> float:
    """Measure end-to-end subplan compilation + sandbox verification latency in milliseconds."""
    sandbox = SandboxedExecutionHarness(timeout_seconds=5.0)
    compiler = MacroSkillCompiler(sandbox=sandbox)

    subplan = {
        "tasks": [
            {"action": "compute_a", "code": "def run(x):\n    return x * 2\nresult = run(5)"},
            {"action": "compute_b", "code": "def run(x):\n    return x + 10\nresult = run(10)"},
        ]
    }

    t0 = time.perf_counter()
    for i in range(iterations):
        compiler.compile_subplan_to_skill(
            subplan,
            skill_name=f"bench_skill_{i}",
            description="Benchmark macro skill compilation",
        )
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_skill_lookup(queries: int = 20_000) -> float:
    """Measure registry lookup latency in microseconds (< 10 µs budget)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)

    # Pre-register 10 skills
    for i in range(10):
        meta = DistilledSkillMetadata(
            skill_name=f"skill_{i}",
            source_subplan_signature=f"sig_{i}",
            compiled_code="def run(): pass",
            historical_avg_latency_ms=10.0,
            compiled_latency_ms=1.0,
            speedup_multiplier=10.0,
            verified=True,
        )
        compiler.register_skill(f"skill_{i}", lambda **kw: {"status": "ok"}, metadata=meta)

    # Warmup
    for _ in range(100):
        compiler.get_metadata("skill_5")

    t0 = time.perf_counter()
    for i in range(queries):
        name = f"skill_{i % 10}"
        m = compiler.get_metadata(name)
        assert m is not None
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / queries) * 1_000_000.0
    return avg_us


def benchmark_skill_invocation(iterations: int = 5_000) -> float:
    """Measure distilled skill invocation latency in milliseconds (< 5 ms budget)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)

    def sample_macro(x: int = 10, y: int = 20) -> dict:
        return {"result": x * y, "status": "computed"}

    meta = DistilledSkillMetadata(
        skill_name="fast_calc",
        source_subplan_signature="calc_sig",
        compiled_code="def run(): pass",
        historical_avg_latency_ms=15.0,
        compiled_latency_ms=0.5,
        speedup_multiplier=30.0,
        verified=True,
    )
    compiler.register_skill("fast_calc", sample_macro, metadata=meta)

    # Warmup
    for _ in range(50):
        compiler.invoke_skill("fast_calc", x=5, y=6)

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = compiler.invoke_skill("fast_calc", x=12, y=34)
        assert res["result"] == 408
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    return avg_ms


def benchmark_concurrent_throughput(num_threads: int = 4, ops_per_thread: int = 10_000) -> float:
    """Measure concurrent registry lookup & invocation throughput in ops/sec."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)

    for i in range(5):
        meta = DistilledSkillMetadata(
            skill_name=f"conc_skill_{i}",
            source_subplan_signature=f"sig_{i}",
            compiled_code="",
            historical_avg_latency_ms=10.0,
            compiled_latency_ms=1.0,
            speedup_multiplier=10.0,
            verified=True,
        )
        compiler.register_skill(f"conc_skill_{i}", lambda idx=i, **kw: {"val": idx}, metadata=meta)

    def worker(worker_id: int) -> int:
        count = 0
        for k in range(ops_per_thread):
            s_name = f"conc_skill_{k % 5}"
            meta = compiler.get_metadata(s_name)
            res = compiler.invoke_skill(s_name)
            if meta is not None and res is not None:
                count += 1
        return count

    total_ops = num_threads * ops_per_thread
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    t1 = time.perf_counter()

    duration = t1 - t0
    ops_per_sec = total_ops / max(0.0001, duration)
    return ops_per_sec


def main() -> None:
    print("=" * 68)
    print("   J.A.R.V.I.S. Phase 20.4 Metacognition & Macro-Skill Benchmark    ")
    print("=" * 68)
    print()

    # 1. Reflection Diagnosis Latency
    print("[1/5] Benchmarking Reflection Diagnosis Latency (500 runs)...")
    refl_ms = benchmark_reflection_diagnosis(iterations=500)
    refl_budget = 2.0
    refl_pass = refl_ms < refl_budget
    refl_status = "PASS" if refl_pass else "FAIL"
    print(f"  Result: {refl_ms:.4f} ms (Budget: < {refl_budget} ms) -> {refl_status}\n")

    # 2. Skill Compilation Latency
    print("[2/5] Benchmarking Skill Compilation Latency (5 compilations)...")
    comp_ms = benchmark_skill_compilation(iterations=5)
    comp_budget = 500.0
    comp_pass = comp_ms < comp_budget
    comp_status = "PASS" if comp_pass else "FAIL"
    print(f"  Result: {comp_ms:.2f} ms (Budget: < {comp_budget} ms) -> {comp_status}\n")

    # 3. Skill Lookup Latency
    print("[3/5] Benchmarking Skill Lookup Latency (20,000 queries)...")
    lookup_us = benchmark_skill_lookup(queries=20_000)
    lookup_budget = 10.0
    lookup_pass = lookup_us < lookup_budget
    lookup_status = "PASS" if lookup_pass else "FAIL"
    print(f"  Result: {lookup_us:.4f} µs (Budget: < {lookup_budget} µs) -> {lookup_status}\n")

    # 4. Skill Invocation Latency
    print("[4/5] Benchmarking Skill Invocation Latency (5,000 runs)...")
    inv_ms = benchmark_skill_invocation(iterations=5_000)
    inv_budget = 5.0
    inv_pass = inv_ms < inv_budget
    inv_status = "PASS" if inv_pass else "FAIL"
    print(f"  Result: {inv_ms:.4f} ms (Budget: < {inv_budget} ms) -> {inv_status}\n")

    # 5. Concurrent Throughput
    print("[5/5] Benchmarking Concurrent Skill Registry Throughput (4 threads)...")
    thru_ops = benchmark_concurrent_throughput(num_threads=4, ops_per_thread=10_000)
    thru_budget = 50_000.0
    thru_pass = thru_ops > thru_budget
    thru_status = "PASS" if thru_pass else "FAIL"
    print(f"  Result: {thru_ops:,.0f} ops/sec (Budget: > {thru_budget:,.0f} ops/sec) -> {thru_status}\n")

    all_passed = refl_pass and comp_pass and lookup_pass and inv_pass and thru_pass

    print("=" * 68)
    print("                     BENCHMARK SUMMARY                           ")
    print("=" * 68)
    print(f"Reflection Diagnosis:    {refl_ms:.4f} ms       (Budget: < {refl_budget} ms)       [{refl_status}]")
    print(f"Skill Compilation:       {comp_ms:.2f} ms     (Budget: < {comp_budget} ms)     [{comp_status}]")
    print(f"Skill Lookup:            {lookup_us:.4f} µs     (Budget: < {lookup_budget} µs)      [{lookup_status}]")
    print(f"Skill Invocation:        {inv_ms:.4f} ms       (Budget: < {inv_budget} ms)       [{inv_status}]")
    print(f"Concurrent Throughput:   {thru_ops:,.0f} ops/sec (Budget: > {thru_budget:,.0f} ops/sec) [{thru_status}]")
    print("=" * 68)
    if all_passed:
        print("OVERALL STATUS: ALL PERFORMANCE BUDGETS MET [PASS]")
    else:
        print("OVERALL STATUS: AT LEAST ONE BUDGET FAILED [FAIL]")
    print("=" * 68)


if __name__ == "__main__":
    main()
