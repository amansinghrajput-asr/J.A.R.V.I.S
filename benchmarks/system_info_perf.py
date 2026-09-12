"""Performance benchmark suite for Phase 22.4 System Information Skills.

Measures operational execution latencies for:
- get_cpu_info dispatch
- get_memory_info dispatch
- get_disk_info dispatch
- get_battery_info dispatch
- get_gpu_info dispatch
- get_network_info dispatch
- get_system_summary dispatch
- skill lookup & can_handle routing
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, List, Tuple

sys.path.insert(0, os.path.abspath("."))

# Silence logging during benchmarks
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)

from app.skills.system.security import SystemSecurityPolicy
from app.skills.system.system_info_skills import SystemInfoSkills


def benchmark_op(name: str, fn: Callable[[], None], iterations: int = 50) -> Tuple[str, float, float, float]:
    """Execute a function repeatedly and compute latency statistics in milliseconds."""
    # Warmup
    for _ in range(5):
        try:
            fn()
        except Exception:
            pass

    durations: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(durations)
    p50_ms = statistics.median(durations)
    p95_ms = sorted(durations)[int(len(durations) * 0.95)]
    return name, mean_ms, p50_ms, p95_ms


def run_all_benchmarks() -> None:
    """Execute all system information performance benchmarks."""
    policy = SystemSecurityPolicy()
    skill = SystemInfoSkills(security_policy=policy)

    benchmarks = [
        ("can_handle & routing", lambda: skill.can_handle("What is my CPU usage?"), 200, 0.1),
        ("get_cpu_info", lambda: skill.execute({"operation": "get_cpu_info"}), 50, 5.0),
        ("get_memory_info", lambda: skill.execute({"operation": "get_memory_info"}), 50, 5.0),
        ("get_disk_info", lambda: skill.execute({"operation": "get_disk_info"}), 50, 5.0),
        ("get_battery_info", lambda: skill.execute({"operation": "get_battery_info"}), 50, 5.0),
        ("get_gpu_info", lambda: skill.execute({"operation": "get_gpu_info"}), 20, 100.0),
        ("get_network_info", lambda: skill.execute({"operation": "get_network_info"}), 50, 50.0),
        ("get_system_summary", lambda: skill.execute({"operation": "get_system_summary"}), 20, 150.0),
    ]

    print("\n================================================================================")
    print("PHASE 22.4 SYSTEM INFORMATION SKILLS PERFORMANCE BENCHMARK")
    print("================================================================================")
    print(f"{'Operation':<26} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Budget (ms)':<12} | {'Status'}")
    print("-" * 86)

    all_passed = True
    for name, fn, iters, budget in benchmarks:
        op_name, mean_ms, p50_ms, p95_ms = benchmark_op(name, fn, iterations=iters)
        passed = mean_ms <= budget
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{op_name:<26} | {mean_ms:<10.3f} | {p50_ms:<10.3f} | {p95_ms:<10.3f} | < {budget:<10.1f} | {status}")

    print("================================================================================")
    if all_passed:
        print("ALL SYSTEM INFORMATION BENCHMARKS COMPLETED WITHIN BUDGET.")
    else:
        print("SOME BENCHMARKS EXCEEDED ARCHITECTURAL BUDGET.")
    print("================================================================================\n")


if __name__ == "__main__":
    run_all_benchmarks()
