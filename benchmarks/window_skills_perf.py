"""Performance benchmark suite for Phase 22.6 Window Skills.

SAFETY REQUIREMENT:
All native Win32 state-changing APIs (SetForegroundWindow, ShowWindow,
PostMessageW, etc.) are strictly mocked.
No real window states are altered.
No Python or benchmarks are executed automatically during development.
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

from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy
from app.skills.system.window_skills import WindowSkills


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
    """Execute all window skill performance benchmarks."""
    policy = SystemSecurityPolicy()
    conf_mgr = SystemConfirmationManager()
    skill = WindowSkills(
        security_policy=policy,
        confirmation_manager=conf_mgr,
    )

    # Mock all Win32 state-changing and OS APIs
    skill._api_is_window = MagicMock(return_value=True)
    skill._api_is_window_visible = MagicMock(return_value=True)
    skill._api_is_cloaked = MagicMock(return_value=False)
    skill._api_get_foreground_window = MagicMock(return_value=1001)
    skill._api_get_window_text = MagicMock(return_value="Document - Notepad")
    skill._api_get_window_pid = MagicMock(return_value=5432)
    skill._get_process_name = MagicMock(return_value="notepad.exe")
    skill._api_set_foreground_window = MagicMock(return_value=True)
    skill._api_show_window = MagicMock(return_value=True)
    skill._api_post_wm_close = MagicMock(return_value=True)
    skill._api_enum_windows = MagicMock(return_value=list(range(1, 51)))

    benchmarks = [
        (
            "can_handle evaluation",
            lambda: skill.can_handle({"operation": "list_windows"}),
        ),
        (
            "SecurityPolicy check (SAFE)",
            lambda: policy.validate_operation("list_windows"),
        ),
        (
            "SecurityPolicy check (CONFIRMATION)",
            lambda: policy.validate_operation("close_window", target="notepad.exe"),
        ),
        (
            "list_windows dispatch (50 windows)",
            lambda: skill.execute({"operation": "list_windows"}),
        ),
        (
            "get_active_window dispatch",
            lambda: skill.execute({"operation": "get_active_window"}),
        ),
        (
            "focus_window dispatch (mocked)",
            lambda: skill.execute({"operation": "focus_window", "target": "Notepad"}),
        ),
        (
            "minimize_window dispatch (mocked)",
            lambda: skill.execute({"operation": "minimize_window", "target": "Notepad"}),
        ),
        (
            "maximize_window dispatch (mocked)",
            lambda: skill.execute({"operation": "maximize_window", "target": "Notepad"}),
        ),
        (
            "restore_window dispatch (mocked)",
            lambda: skill.execute({"operation": "restore_window", "target": "Notepad"}),
        ),
    ]

    print("=" * 70)
    print("Phase 22.6 Window Skills Performance Benchmark")
    print("=" * 70)
    print(f"{'Operation':<35} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 70)

    for name, fn in benchmarks:
        op_name, mean_ms, p50_ms, p95_ms = benchmark_op(name, fn, iterations=50)
        print(f"{op_name:<35} {mean_ms:<12.3f} {p50_ms:<12.3f} {p95_ms:<12.3f}")

    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
