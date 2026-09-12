"""Performance Benchmarking Suite for Phase 21 Skill Evolution Engine.

Measures latency and throughput metrics against Phase 21 architectural performance budgets:
1. Metric update latency (Target: < 50.0 µs)
2. Skill evaluation latency (Target: < 50.0 µs)
3. Registry lookup latency (Target: < 10.0 µs)
4. Version lookup latency (Target: < 10.0 µs)
5. Concurrent metric-update throughput (Target: > 20,000 ops/sec)
6. Evolution decision latency (Target: < 100.0 µs)
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
from app.ai.planner.metacognition.evolution import (
    SkillEvolutionConfig,
    SkillEvolutionEngine,
    SkillStatus,
)
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness


def benchmark_metric_update_latency(iterations: int = 10_000) -> float:
    """Measure single metric update latency in microseconds (< 50.0 µs target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    engine = SkillEvolutionEngine(compiler=compiler)
    engine.register_skill("bench_tool_update")

    # Warmup
    for _ in range(100):
        engine.record_invocation("bench_tool_update")

    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.record_invocation("bench_tool_update")
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    return avg_us


def benchmark_skill_evaluation_latency(iterations: int = 5_000) -> float:
    """Measure deterministic skill evaluation scoring latency in microseconds (< 50.0 µs target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    engine = SkillEvolutionEngine(compiler=compiler)
    engine.register_skill("bench_tool_eval")
    for _ in range(10):
        engine.record_execution("bench_tool_eval", success=True, latency_ms=12.0)

    # Warmup
    for _ in range(100):
        engine.evaluate_skill("bench_tool_eval")

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = engine.evaluate_skill("bench_tool_eval")
        assert res.score > 0.0
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    return avg_us


def benchmark_registry_lookup_latency(queries: int = 20_000) -> float:
    """Measure skill status/metrics lookup latency in microseconds (< 10.0 µs target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    engine = SkillEvolutionEngine(compiler=compiler)

    for i in range(10):
        engine.register_skill(f"tool_{i}")

    # Warmup
    for _ in range(100):
        engine.get_skill_status("tool_5")

    t0 = time.perf_counter()
    for i in range(queries):
        name = f"tool_{i % 10}"
        status = engine.get_skill_status(name)
        assert status == SkillStatus.ACTIVE
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / queries) * 1_000_000.0
    return avg_us


def benchmark_version_lookup_latency(queries: int = 20_000) -> float:
    """Measure active/stable version resolution latency in microseconds (< 10.0 µs target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    engine = SkillEvolutionEngine(compiler=compiler)

    for i in range(10):
        engine.register_skill(f"vtool_{i}", version="v1.0.0", is_stable=True)

    # Warmup
    for _ in range(100):
        engine.get_active_version("vtool_3")

    t0 = time.perf_counter()
    for i in range(queries):
        name = f"vtool_{i % 10}"
        ver = engine.get_active_version(name)
        assert ver == "v1.0.0"
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / queries) * 1_000_000.0
    return avg_us


def benchmark_concurrent_metric_throughput(num_threads: int = 8, ops_per_thread: int = 5_000) -> float:
    """Measure throughput of concurrent multi-threaded metric updates in ops/sec (> 20,000 ops/sec target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    engine = SkillEvolutionEngine(compiler=compiler)

    for i in range(5):
        engine.register_skill(f"thread_tool_{i}")

    def worker(tid: int) -> int:
        for j in range(ops_per_thread):
            target = f"thread_tool_{(tid + j) % 5}"
            engine.record_invocation(target)
        return ops_per_thread

    total_ops = num_threads * ops_per_thread

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(num_threads)]
        for f in futures:
            f.result()
    t1 = time.perf_counter()

    duration = max(0.0001, t1 - t0)
    throughput = total_ops / duration
    return throughput


def benchmark_evolution_decision_latency(iterations: int = 5_000) -> float:
    """Measure lifecycle state transition evaluation decision latency in microseconds (< 100.0 µs target)."""
    sandbox = SandboxedExecutionHarness()
    compiler = MacroSkillCompiler(sandbox=sandbox)
    config = SkillEvolutionConfig(degrade_consecutive_threshold=2)
    engine = SkillEvolutionEngine(compiler=compiler, config=config)
    engine.register_skill("decision_tool")

    # Warmup
    for _ in range(100):
        engine.is_skill_selectable("decision_tool")

    t0 = time.perf_counter()
    for i in range(iterations):
        engine.is_skill_selectable("decision_tool")
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    return avg_us


def run_all_benchmarks() -> None:
    """Execute all Phase 21 performance benchmarks and report results."""
    print("=" * 72)
    print("PHASE 21 SKILL EVOLUTION ENGINE PERFORMANCE BENCHMARK SUITE")
    print("=" * 72)

    results = []

    # 1. Metric Update Latency
    print("1. Benchmarking Metric Update Latency (10,000 iterations)...")
    lat_update = benchmark_metric_update_latency(iterations=10_000)
    ok_update = lat_update < 50.0
    results.append(("Metric Update Latency", f"{lat_update:.3f} µs", "< 50.0 µs", ok_update))

    # 2. Skill Evaluation Latency
    print("2. Benchmarking Skill Evaluation Latency (5,000 iterations)...")
    lat_eval = benchmark_skill_evaluation_latency(iterations=5_000)
    ok_eval = lat_eval < 50.0
    results.append(("Skill Evaluation Latency", f"{lat_eval:.3f} µs", "< 50.0 µs", ok_eval))

    # 3. Registry Lookup Latency
    print("3. Benchmarking Registry Lookup Latency (20,000 queries)...")
    lat_lookup = benchmark_registry_lookup_latency(queries=20_000)
    ok_lookup = lat_lookup < 10.0
    results.append(("Registry Lookup Latency", f"{lat_lookup:.3f} µs", "< 10.0 µs", ok_lookup))

    # 4. Version Lookup Latency
    print("4. Benchmarking Version Lookup Latency (20,000 queries)...")
    lat_ver = benchmark_version_lookup_latency(queries=20_000)
    ok_ver = lat_ver < 10.0
    results.append(("Version Lookup Latency", f"{lat_ver:.3f} µs", "< 10.0 µs", ok_ver))

    # 5. Concurrent Metric-Update Throughput
    print("5. Benchmarking Concurrent Metric-Update Throughput (8 threads, 40,000 ops)...")
    tp_concurrent = benchmark_concurrent_metric_throughput(num_threads=8, ops_per_thread=5_000)
    ok_tp = tp_concurrent > 20_000.0
    results.append(("Concurrent Update Throughput", f"{tp_concurrent:,.0f} ops/s", "> 20,000 ops/s", ok_tp))

    # 6. Evolution Decision Latency
    print("6. Benchmarking Evolution Decision Latency (5,000 iterations)...")
    lat_decision = benchmark_evolution_decision_latency(iterations=5_000)
    ok_decision = lat_decision < 100.0
    results.append(("Evolution Decision Latency", f"{lat_decision:.3f} µs", "< 100.0 µs", ok_decision))

    print("\n" + "=" * 72)
    print(f"{'Benchmark Metric':<32} | {'Measured':<14} | {'Target Budget':<14} | {'Status'}")
    print("-" * 72)
    all_passed = True
    for metric, measured, target, status in results:
        status_str = "PASS" if status else "FAIL"
        if not status:
            all_passed = False
        print(f"{metric:<32} | {measured:<14} | {target:<14} | {status_str}")
    print("=" * 72)

    if all_passed:
        print("\nOVERALL BENCHMARK RESULT: ALL ARCHITECTURAL BUDGETS PASSED!")
    else:
        print("\nOVERALL BENCHMARK RESULT: ONE OR MORE BUDGETS EXCEEDED!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_benchmarks()
