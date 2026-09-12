"""Performance benchmark suite for Phase 22.5 System Control Skills.

Measures operational latencies for:
- can_handle & routing
- Security policy evaluation
- get_volume dispatch
- set_volume dispatch
- get_brightness dispatch
- lock_workstation dispatch (mocked API)
- Confirmation token issuance & verification
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, List, Tuple
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("."))

# Silence logging during benchmarks
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)
logging.getLogger("screen_brightness_control").setLevel(logging.ERROR)

from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy
from app.skills.system.system_control_skills import SystemControlSkills


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
    """Execute all system control performance benchmarks."""
    policy = SystemSecurityPolicy()
    conf_mgr = SystemConfirmationManager()
    skill = SystemControlSkills(security_policy=policy, confirmation_manager=conf_mgr)

    # Mock destructive OS hooks
    skill._call_lock_api = MagicMock(return_value=True)

    def benchmark_conf_cycle():
        token = conf_mgr.request_confirmation(operation="set_volume", parameters={"level": 100})
        conf_mgr.resolve_confirmation(token, approved=True)
        conf_mgr.verify_and_consume(token, operation="set_volume", parameters={"level": 100})

    benchmarks = [
        ("can_handle & routing", lambda: skill.can_handle("What is the volume?"), 200, 0.1),
        ("policy evaluation", lambda: policy.validate_operation("get_volume"), 200, 0.5),
        ("get_volume dispatch", lambda: skill.execute({"operation": "get_volume"}), 50, 5.0),
        ("set_volume dispatch", lambda: skill.execute({"operation": "set_volume", "parameters": {"level": 50}}), 50, 5.0),
        ("get_brightness dispatch", lambda: skill.execute({"operation": "get_brightness"}), 20, 100.0),
        ("lock_workstation dispatch", lambda: skill.execute({"operation": "lock_workstation"}), 50, 1.0),
        ("confirmation workflow", benchmark_conf_cycle, 100, 1.0),
    ]

    print("\n================================================================================")
    print("PHASE 22.5 SYSTEM CONTROL SKILLS PERFORMANCE BENCHMARK")
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
        print("ALL SYSTEM CONTROL BENCHMARKS COMPLETED WITHIN BUDGET.")
    else:
        print("SOME BENCHMARKS EXCEEDED ARCHITECTURAL BUDGET.")
    print("================================================================================\n")


if __name__ == "__main__":
    run_all_benchmarks()
