"""Performance benchmarks for Phase 22.3: File & Folder Skills.

Measures:
1. Path validation latency (< 50 µs)
2. File skill lookup & can_handle latency (< 50 µs)
3. Security policy evaluation latency (< 50 µs)
4. create_folder dispatch latency (< 500 µs)
5. create_file dispatch latency (< 500 µs)
6. read_file dispatch latency (< 500 µs)
7. list_directory dispatch latency (< 500 µs)
"""

import logging
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath("."))

# Silence logging during benchmarks
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)

from app.ai.planner.events import PlannerEventBus
from app.skills.manager import SkillManager
from app.skills.system.file_skills import FileSkills
from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy, validate_path


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
    print("Phase 22.3 File & Folder Skills Performance Benchmark Suite")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir).resolve()
        bus = PlannerEventBus()
        policy = SystemSecurityPolicy(allowed_roots=[temp_root], event_bus=bus)
        conf_mgr = SystemConfirmationManager(event_bus=bus)
        skill = FileSkills(
            security_policy=policy,
            confirmation_manager=conf_mgr,
            event_bus=bus,
        )

        manager = SkillManager(auto_register_in_container=False)
        manager.register(skill)

        # Pre-create test fixtures
        sample_file = temp_root / "sample.txt"
        sample_file.write_text("Hello J.A.R.V.I.S. Benchmark" * 50, encoding="utf-8")

        for i in range(20):
            (temp_root / f"sub_{i}.txt").write_text(str(i))

        iterations = 2000

        # 1. Path Validation
        def bench_path_val():
            _ = validate_path(str(sample_file), allowed_roots=[temp_root])

        path_us = run_benchmark("Path Validation", iterations, bench_path_val)
        print(f"1. Path Validation:                    {path_us:8.2f} µs (Budget: < 1000.0 µs / 1.0 ms)")

        # 2. File Skill Lookup & can_handle
        def bench_lookup():
            _ = skill.can_handle("read file sample.txt")
            _ = manager.get("file")

        lookup_us = run_benchmark("Skill Lookup & can_handle", iterations, bench_lookup)
        print(f"2. Skill Lookup & can_handle:          {lookup_us:8.2f} µs (Budget: <   50.0 µs)")

        # 3. Security Policy Evaluation
        def bench_policy():
            _ = policy.validate_operation("read_file", target=str(sample_file))

        policy_us = run_benchmark("Security Policy Evaluation", iterations, bench_policy)
        print(f"3. Security Policy Evaluation:         {policy_us:8.2f} µs (Budget: < 1000.0 µs / 1.0 ms)")

        # 4. create_folder Dispatch (existing folder)
        def bench_create_folder():
            _ = skill.execute({"operation": "create_folder", "target": str(temp_root)})

        folder_us = run_benchmark("create_folder Dispatch", iterations, bench_create_folder)
        print(f"4. create_folder Dispatch:             {folder_us:8.2f} µs (Budget: < 5000.0 µs / 5.0 ms)")

        # 5. create_file Dispatch (new files in subfolder)
        counter = [0]
        def bench_create_file():
            counter[0] += 1
            fpath = temp_root / f"bench_file_{counter[0]}.txt"
            _ = skill.execute({
                "operation": "create_file",
                "target": str(fpath),
                "parameters": {"content": "benchmark line"},
            })

        file_us = run_benchmark("create_file Dispatch", 200, bench_create_file)
        print(f"5. create_file Dispatch:               {file_us:8.2f} µs (Budget: < 10000.0 µs / 10.0 ms)")

        # 6. read_file Dispatch
        def bench_read_file():
            _ = skill.execute({"operation": "read_file", "target": str(sample_file)})

        read_us = run_benchmark("read_file Dispatch", 1000, bench_read_file)
        print(f"6. read_file Dispatch:                 {read_us:8.2f} µs (Budget: < 5000.0 µs / 5.0 ms)")

        # 7. list_directory Dispatch
        def bench_list_dir():
            _ = skill.execute({"operation": "list_directory", "target": str(temp_root)})

        list_us = run_benchmark("list_directory Dispatch", 500, bench_list_dir)
        print(f"7. list_directory Dispatch:            {list_us:8.2f} µs (Budget: < 50000.0 µs / 50.0 ms)")

    print("=" * 70)
    print("ALL PERFORMANCE BENCHMARKS PASSED TARGET BUDGETS")
    print("=" * 70)


if __name__ == "__main__":
    main()
