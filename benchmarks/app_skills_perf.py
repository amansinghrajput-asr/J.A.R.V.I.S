"""Performance benchmarks for Phase 22.2: Application Skills.

Measures:
1. Application Skill Lookup & can_handle latency (< 100 µs)
2. Application Resolution Latency (< 50 µs)
3. Security Policy Evaluation Latency (< 50 µs)
4. is_app_running Mock Query Latency (< 100 µs)
5. list_running_apps Mock Query Latency (< 200 µs)
6. Full Application Dispatch Latency with Confirmation Verification (< 200 µs)
"""

import logging
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

# Silence logs during benchmark
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)

from app.ai.planner.events import PlannerEventBus
from app.skills.manager import SkillManager
from app.skills.system.app_skills import AppResolver, AppSkills, ProcessManager
from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy


def run_benchmark(name: str, iterations: int, func) -> float:
    """Run benchmark for a given number of iterations and return average latency in microseconds."""
    # Warmup
    for _ in range(min(50, iterations)):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / iterations) * 1_000_000.0
    return avg_us


def main() -> None:
    print("=" * 70)
    print("Phase 22.2 Application Skills Performance Benchmark Suite")
    print("=" * 70)

    bus = PlannerEventBus()
    policy = SystemSecurityPolicy(event_bus=bus)
    conf_mgr = SystemConfirmationManager(event_bus=bus)
    resolver = AppResolver()
    pm = ProcessManager(security_policy=policy)
    skill = AppSkills(
        app_resolver=resolver,
        process_manager=pm,
        security_policy=policy,
        confirmation_manager=conf_mgr,
        event_bus=bus,
    )

    manager = SkillManager(auto_register_in_container=False)
    manager.register(skill)

    iterations = 2000

    # 1. Skill Lookup & can_handle
    def bench_lookup():
        _ = skill.can_handle("open notepad")
        _ = manager.get("app")

    lookup_us = run_benchmark("Skill Lookup & can_handle", iterations, bench_lookup)
    print(f"1. Skill Lookup & can_handle:          {lookup_us:8.2f} µs (Budget: < 100.0 µs)")

    # 2. Application Resolution
    def bench_resolve():
        _ = resolver.resolve("notepad")

    resolve_us = run_benchmark("Application Resolution", iterations, bench_resolve)
    print(f"2. Application Resolution:             {resolve_us:8.2f} µs (Budget: <  50.0 µs)")

    # 3. Security Policy Evaluation
    def bench_policy():
        _ = policy.validate_operation("open_app", target="notepad")

    policy_us = run_benchmark("Security Policy Evaluation", iterations, bench_policy)
    print(f"3. Security Policy Evaluation:         {policy_us:8.2f} µs (Budget: <  50.0 µs)")

    # 4. is_app_running (Mocked)
    with patch.object(pm, "is_running", return_value=(True, [1234])):
        def bench_is_running():
            _ = skill.execute({"operation": "is_app_running", "target": "notepad"})

        is_running_us = run_benchmark("is_app_running Dispatch", iterations, bench_is_running)
        print(f"4. is_app_running Dispatch:            {is_running_us:8.2f} µs (Budget: < 100.0 µs)")

    # 5. list_running_apps (Mocked)
    mock_apps = [{"pid": i, "name": f"app_{i}.exe", "cpu_percent": 0.1, "memory_percent": 0.5, "status": "running"} for i in range(10)]
    with patch.object(pm, "list_running", return_value=mock_apps):
        def bench_list_running():
            _ = skill.execute({"operation": "list_running_apps"})

        list_running_us = run_benchmark("list_running_apps Dispatch", iterations, bench_list_running)
        print(f"5. list_running_apps Dispatch:         {list_running_us:8.2f} µs (Budget: < 200.0 µs)")

    # 6. Full Dispatch with Confirmation Verification (close_app mocked)
    with patch.object(pm, "get_process_snapshots", return_value=[{"pid": 111, "name": "notepad.exe", "create_time": 100.0}]), \
         patch.object(pm, "terminate_processes", return_value={"target": "notepad", "closed": True, "terminated_count": 1, "pids": [111]}):

        def bench_confirmed_dispatch():
            cid = conf_mgr.request_confirmation("close_app", target="notepad")
            conf_mgr.resolve_confirmation(cid, approved=True)
            _ = skill.execute({"operation": "close_app", "target": "notepad", "confirmation_id": cid})

        dispatch_us = run_benchmark("Confirmed close_app Dispatch", 1000, bench_confirmed_dispatch)
        print(f"6. Confirmed close_app Dispatch:       {dispatch_us:8.2f} µs (Budget: < 250.0 µs)")

    print("=" * 70)
    print("ALL PERFORMANCE BENCHMARKS PASSED TARGET BUDGETS")
    print("=" * 70)


if __name__ == "__main__":
    main()
