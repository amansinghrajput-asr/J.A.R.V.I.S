"""Unified Performance Benchmark Suite for Phase 22 System Skills.

Consolidates performance evaluation across all six Phase 22 system skill modules:
1. AppSkills (application lookup, process query, running apps enumeration)
2. FileSkills (directory/file creation, reading, listing, path containment in TemporaryDirectory)
3. SystemInfoSkills (CPU, memory, disk, battery, system summary diagnostics)
4. SystemControlSkills (volume, brightness, workstation lock - native APIs mocked)
5. WindowSkills (window enumeration, active window, focus/state control - user32 mocked)
6. BrowserSkills (URL validation, domain routing, search dispatch - webbrowser.open mocked)

Safety Guarantees:
- Strictly zero destructive operations (no shutdown/restart/sleep/real process termination).
- Filesystem operations execute exclusively within tempfile.TemporaryDirectory.
- All native OS hooks and external launch APIs are mocked.
- Outputs structured JSON to benchmarks/results/system_skills_benchmark_results.json.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

# Silence logging during benchmarks
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)
logging.getLogger("screen_brightness_control").setLevel(logging.ERROR)

from app.skills.system.app_skills import AppResolver, AppSkills, ProcessManager
from app.skills.system.browser_skills import BrowserSkills, is_url_or_domain, validate_url
from app.skills.system.file_skills import FileSkills
from app.skills.system.security import (
    SystemConfirmationManager,
    SystemSecurityPolicy,
    validate_path,
)
from app.skills.system.system_control_skills import SystemControlSkills
from app.skills.system.system_info_skills import SystemInfoSkills
from app.skills.system.window_skills import WindowSkills


def benchmark_op(
    category: str,
    operation: str,
    fn: Callable[[], Any],
    iterations: int = 500,
    warmup: int = 25,
    budget_us: float = 1000.0,
) -> Dict[str, Any]:
    """Execute a function repeatedly and compute microsecond latency statistics."""
    # Warmup
    for _ in range(min(warmup, iterations)):
        try:
            fn()
        except Exception:
            pass

    latencies_us: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000.0)

    mean_us = statistics.mean(latencies_us)
    min_us = min(latencies_us)
    max_us = max(latencies_us)
    stdev_us = statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0.0

    return {
        "category": category,
        "operation": operation,
        "iterations": iterations,
        "mean_us": round(mean_us, 2),
        "min_us": round(min_us, 2),
        "max_us": round(max_us, 2),
        "stdev_us": round(stdev_us, 2),
        "budget_us": budget_us,
        "status": "PASS" if mean_us <= budget_us else "WARN",
    }


def run_unified_benchmarks() -> Dict[str, Any]:
    print("=" * 88)
    print("J.A.R.V.I.S Phase 22 Unified System Skills Performance Benchmark Suite")
    print("=" * 88)

    policy = SystemSecurityPolicy()
    conf_mgr = SystemConfirmationManager()
    all_results: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. AppSkills Benchmarks (Mocked Subprocess / Process Query)
    # =========================================================================
    print("\n[Category 1/6] AppSkills Benchmarks (Process / Application Operations)")
    print("-" * 88)

    mock_pm = MagicMock(spec=ProcessManager)
    mock_pm.is_running = MagicMock(return_value=(True, [1001]))
    mock_pm.list_running_applications = MagicMock(return_value=[
        {"pid": 1001, "name": "notepad.exe", "title": "Untitled - Notepad"},
        {"pid": 1002, "name": "calc.exe", "title": "Calculator"},
    ])
    app_skills = AppSkills(
        process_manager=mock_pm,
        security_policy=policy,
        confirmation_manager=conf_mgr,
    )

    app_ops = [
        ("can_handle 'open notepad'", lambda: app_skills.can_handle("open notepad"), 2000, 100.0),
        ("parse_command 'open notepad'", lambda: app_skills.parse_command("open notepad"), 2000, 100.0),
        ("is_app_running dispatch (mocked)", lambda: app_skills.execute({"action": "is_app_running", "target": "notepad"}), 1000, 500.0),
        ("list_running_apps dispatch (mocked)", lambda: app_skills.execute({"action": "list_running_apps"}), 1000, 500.0),
    ]

    for op_name, fn, iters, budg in app_ops:
        res = benchmark_op("AppSkills", op_name, fn, iterations=iters, budget_us=budg)
        all_results.append(res)
        print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # 2. FileSkills Benchmarks (Isolated in TemporaryDirectory)
    # =========================================================================
    print("\n[Category 2/6] FileSkills Benchmarks (Isolated Temporary Filesystem)")
    print("-" * 88)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_skills = FileSkills(security_policy=policy, confirmation_manager=conf_mgr)

        sample_file = temp_path / "bench_read.txt"
        sample_file.write_text("JARVIS test content for benchmark read operation." * 20, encoding="utf-8")

        def test_path_val():
            validate_path(str(temp_path / "sub" / "file.txt"))

        def test_create_folder():
            file_skills.create_folder(str(temp_path / "bench_folder"), {})

        def test_create_file():
            file_skills.create_file(str(temp_path / "bench_create.txt"), {"content": "data", "overwrite": True})

        def test_read_file():
            file_skills.read_file(str(sample_file), {})

        def test_list_dir():
            file_skills.list_directory(str(temp_path), {})

        file_ops = [
            ("Path containment validation", test_path_val, 2000, 1000.0),
            ("can_handle 'create file'", lambda: file_skills.can_handle("create file test.txt"), 2000, 100.0),
            ("create_folder (filesystem I/O)", test_create_folder, 1000, 5000.0),
            ("create_file (filesystem I/O)", test_create_file, 1000, 5000.0),
            ("read_file (filesystem I/O)", test_read_file, 1000, 5000.0),
            ("list_directory (filesystem I/O)", test_list_dir, 1000, 5000.0),
        ]

        for op_name, fn, iters, budg in file_ops:
            res = benchmark_op("FileSkills", op_name, fn, iterations=iters, budget_us=budg)
            all_results.append(res)
            print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # 3. SystemInfoSkills Benchmarks (Safe Read-Only Telemetry)
    # =========================================================================
    print("\n[Category 3/6] SystemInfoSkills Benchmarks (Read-Only Diagnostics)")
    print("-" * 88)

    info_skills = SystemInfoSkills(security_policy=policy, confirmation_manager=conf_mgr)
    info_skills.get_gpu_info = MagicMock(return_value={"available": True, "name": "NVIDIA GeForce RTX", "memory_total_bytes": 8589934592, "memory_used_bytes": 1073741824, "memory_free_bytes": 7516192768})
    info_skills.get_network_info = MagicMock(return_value={"hostname": "JARVIS-HOST", "ip_address": "192.168.1.50", "connected": True, "active_interfaces": ["Ethernet"], "bytes_sent": 1024000, "bytes_received": 2048000})

    info_ops = [
        ("can_handle 'what is cpu usage'", lambda: info_skills.can_handle("what is cpu usage"), 2000, 100.0),
        ("get_cpu_info (read-only)", lambda: info_skills.get_cpu_info(), 500, 5000.0),
        ("get_memory_info (read-only)", lambda: info_skills.get_memory_info(), 500, 2000.0),
        ("get_disk_info (read-only)", lambda: info_skills.get_disk_info(), 500, 3000.0),
        ("get_battery_info (read-only)", lambda: info_skills.get_battery_info(), 500, 2000.0),
        ("get_system_summary (aggregated)", lambda: info_skills.get_system_summary(), 500, 15000.0),
    ]

    for op_name, fn, iters, budg in info_ops:
        res = benchmark_op("SystemInfoSkills", op_name, fn, iterations=iters, budget_us=budg)
        all_results.append(res)
        print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # 4. SystemControlSkills Benchmarks (Mocked Native Control APIs)
    # =========================================================================
    print("\n[Category 4/6] SystemControlSkills Benchmarks (Mocked Native Audio & Display)")
    print("-" * 88)

    ctrl_skills = SystemControlSkills(security_policy=policy, confirmation_manager=conf_mgr)

    mock_endpoint = MagicMock()
    mock_endpoint.GetMasterVolumeLevelScalar.return_value = 0.5
    mock_endpoint.GetMute.return_value = False
    mock_sbc = MagicMock()
    mock_sbc.get_brightness.return_value = [75]
    mock_sbc.set_brightness.return_value = [75]

    ctrl_skills._get_audio_endpoint = MagicMock(return_value=mock_endpoint)
    ctrl_skills._get_sbc_module = MagicMock(return_value=mock_sbc)
    ctrl_skills._call_lock_api = MagicMock(return_value=True)

    ctrl_ops = [
        ("can_handle 'set volume 50'", lambda: ctrl_skills.can_handle("set volume 50"), 2000, 100.0),
        ("get_volume dispatch", lambda: ctrl_skills.get_volume(), 1000, 500.0),
        ("set_volume dispatch (mocked)", lambda: ctrl_skills.set_volume(50), 1000, 500.0),
        ("get_brightness dispatch (mocked)", lambda: ctrl_skills.get_brightness(), 1000, 500.0),
        ("set_brightness dispatch (mocked)", lambda: ctrl_skills.set_brightness(75), 1000, 500.0),
        ("lock_workstation dispatch (mocked)", lambda: ctrl_skills.lock_workstation(), 1000, 500.0),
    ]

    for op_name, fn, iters, budg in ctrl_ops:
        res = benchmark_op("SystemControlSkills", op_name, fn, iterations=iters, budget_us=budg)
        all_results.append(res)
        print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # 5. WindowSkills Benchmarks (Mocked Win32 user32 APIs)
    # =========================================================================
    print("\n[Category 5/6] WindowSkills Benchmarks (Mocked Win32 user32 Desktop APIs)")
    print("-" * 88)

    win_skills = WindowSkills(security_policy=policy, confirmation_manager=conf_mgr)

    win_skills._api_is_window = MagicMock(return_value=True)
    win_skills._api_is_window_visible = MagicMock(return_value=True)
    win_skills._api_is_cloaked = MagicMock(return_value=False)
    win_skills._api_get_foreground_window = MagicMock(return_value=1001)
    win_skills._api_get_window_text = MagicMock(return_value="Document - Notepad")
    win_skills._api_get_window_pid = MagicMock(return_value=5432)
    win_skills._get_process_name = MagicMock(return_value="notepad.exe")
    win_skills._api_set_foreground_window = MagicMock(return_value=True)
    win_skills._api_show_window = MagicMock(return_value=True)
    win_skills._api_post_wm_close = MagicMock(return_value=True)
    win_skills._api_enum_windows = MagicMock(return_value=list(range(1, 21)))

    win_ops = [
        ("can_handle 'list windows'", lambda: win_skills.can_handle("list windows"), 2000, 100.0),
        ("list_windows dispatch (mocked)", lambda: win_skills._execute_operation("list_windows", None, {}), 1000, 3000.0),
        ("get_active_window (mocked)", lambda: win_skills._execute_operation("get_active_window", None, {}), 1000, 500.0),
        ("focus_window (mocked)", lambda: win_skills._execute_operation("focus_window", "Document - Notepad", {}), 1000, 3000.0),
        ("minimize_window (mocked)", lambda: win_skills._execute_operation("minimize_window", "Document - Notepad", {}), 1000, 3000.0),
        ("maximize_window (mocked)", lambda: win_skills._execute_operation("maximize_window", "Document - Notepad", {}), 1000, 3000.0),
        ("restore_window (mocked)", lambda: win_skills._execute_operation("restore_window", "Document - Notepad", {}), 1000, 3000.0),
    ]

    for op_name, fn, iters, budg in win_ops:
        res = benchmark_op("WindowSkills", op_name, fn, iterations=iters, budget_us=budg)
        all_results.append(res)
        print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # 6. BrowserSkills Benchmarks (Mocked webbrowser.open)
    # =========================================================================
    print("\n[Category 6/6] BrowserSkills Benchmarks (Mocked Web Navigation & Dispatch)")
    print("-" * 88)

    browser_skills = BrowserSkills(security_policy=policy, confirmation_manager=conf_mgr)

    with patch("webbrowser.open", MagicMock(return_value=True)):
        browser_ops = [
            ("URL validation (https://...)", lambda: validate_url("https://github.com"), 2000, 100.0),
            ("Domain routing check (google.com)", lambda: is_url_or_domain("google.com"), 2000, 100.0),
            ("can_handle 'search web for ai'", lambda: browser_skills.can_handle("search web for ai"), 2000, 100.0),
            ("open_url dispatch (mocked)", lambda: browser_skills.execute({"action": "open_url", "target": "https://example.com"}), 1000, 500.0),
            ("search_web dispatch (mocked)", lambda: browser_skills.execute({"action": "search_web", "target": "python async"}), 1000, 500.0),
        ]

        for op_name, fn, iters, budg in browser_ops:
            res = benchmark_op("BrowserSkills", op_name, fn, iterations=iters, budget_us=budg)
            all_results.append(res)
            print(f"  {res['category']:<16} | {res['operation']:<42} | {res['mean_us']:>8.2f} µs | {res['status']}")

    # =========================================================================
    # Summary & JSON Artifact Generation
    # =========================================================================
    print("\n" + "=" * 88)
    total_benchmarks = len(all_results)
    passed_count = sum(1 for r in all_results if r["status"] == "PASS")
    mean_latencies = [r["mean_us"] for r in all_results]
    overall_mean = statistics.mean(mean_latencies)

    print(f"Summary: {passed_count}/{total_benchmarks} operations satisfied their performance budget.")
    print(f"Overall Average Operation Latency: {overall_mean:.2f} µs across all 6 skill modules.")
    print("Zero destructive operations executed. All native APIs strictly mocked or isolated.")
    print("=" * 88)

    output_data = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_benchmarks": total_benchmarks,
        "passed_benchmarks": passed_count,
        "overall_mean_us": round(overall_mean, 2),
        "results": all_results,
    }

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "system_skills_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Benchmark results successfully saved to: {json_path}")

    return output_data


if __name__ == "__main__":
    run_unified_benchmarks()
