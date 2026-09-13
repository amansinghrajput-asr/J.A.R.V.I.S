"""Performance benchmark suite for Phase 22.7 Browser & Web Launch Skills.

Measures safe operational latencies without launching external browsers or accessing the network:
1. URL Validation Latency (valid HTTP/HTTPS, blocked schemes, malformed URLs)
2. Domain & Target Routing / Normalization (google.com, www.site.org, invalid targets)
3. Action Recognition via can_handle & regex matching
4. Full Dispatch Latency for open_url, search_web, and open_browser (strictly mocked webbrowser.open)

Safety Invariants:
- All external calls to webbrowser.open() are strictly mocked.
- Zero network socket or HTTP traffic is generated.
- Benchmarks are deterministic and thread-safe.
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

# Silence logging during benchmarks
logging.getLogger("SYSTEM").setLevel(logging.WARNING)
logging.getLogger("SKILL").setLevel(logging.WARNING)
logging.getLogger("SKILL_MANAGER").setLevel(logging.WARNING)

from app.skills.system.browser_skills import (
    BrowserSkills,
    is_url_or_domain,
    validate_url,
)
from app.skills.system.security import (
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
)


def benchmark_op(
    name: str,
    fn: Callable[[], None],
    iterations: int = 1000,
    warmup: int = 50,
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
        "operation": name,
        "iterations": iterations,
        "mean_us": mean_us,
        "min_us": min_us,
        "max_us": max_us,
        "stdev_us": stdev_us,
        "budget_pass": mean_us < 100.0,
    }


def main() -> List[Dict[str, Any]]:
    print("=" * 80)
    print("J.A.R.V.I.S Phase 22.7 Browser Skills Performance Benchmark Suite")
    print("=" * 80)

    policy = SystemSecurityPolicy()
    conf_mgr = SystemConfirmationManager()
    skill = BrowserSkills(security_policy=policy, confirmation_manager=conf_mgr)

    # Patch webbrowser.open at module level to guarantee zero real browser execution
    mock_browser = MagicMock(return_value=True)

    results: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # 1. URL Validation Latencies
    # --------------------------------------------------------------------------
    print("\n[1] URL Validation Latency Benchmarks")
    print("-" * 80)

    def test_valid_https():
        validate_url("https://www.google.com/search?q=jarvis+ai")

    def test_valid_http():
        validate_url("http://example.com:8080/docs/index.html")

    def test_blocked_file():
        try:
            validate_url("file:///C:/Windows/System32/calc.exe")
        except SecurityPolicyViolationError:
            pass

    def test_blocked_javascript():
        try:
            validate_url("javascript:alert(document.cookie)")
        except SecurityPolicyViolationError:
            pass

    def test_blocked_data():
        try:
            validate_url("data:text/html,<html><body>pwn</body></html>")
        except SecurityPolicyViolationError:
            pass

    def test_malformed_url():
        try:
            validate_url("https://[invalid-ipv6-bracket")
        except Exception:
            pass

    val_ops = [
        ("URL Validation: Valid HTTPS URL", test_valid_https),
        ("URL Validation: Valid HTTP with Port", test_valid_http),
        ("URL Validation: Blocked file:// Scheme", test_blocked_file),
        ("URL Validation: Blocked javascript: Scheme", test_blocked_javascript),
        ("URL Validation: Blocked data: Scheme", test_blocked_data),
        ("URL Validation: Malformed URL Rejection", test_malformed_url),
    ]

    for name, fn in val_ops:
        res = benchmark_op(name, fn, iterations=2000)
        results.append(res)
        print(f"  {res['operation']:<45} : {res['mean_us']:>7.2f} µs (min: {res['min_us']:>6.2f}, max: {res['max_us']:>7.2f})")

    # --------------------------------------------------------------------------
    # 2. Domain / Target Routing & Normalization
    # --------------------------------------------------------------------------
    print("\n[2] Domain / Target Routing & Normalization Benchmarks")
    print("-" * 80)

    def test_domain_detect_bare():
        is_url_or_domain("google.com")

    def test_domain_detect_www():
        is_url_or_domain("www.youtube.com")

    def test_domain_detect_non_url():
        is_url_or_domain("notepad")

    def test_normalize_bare_domain():
        skill.parse_command("open github.com")

    def test_normalize_full_url():
        skill.parse_command("open https://docs.python.org/3/library/webbrowser.html")

    dom_ops = [
        ("Domain Detection: Bare Domain (google.com)", test_domain_detect_bare),
        ("Domain Detection: WWW Domain (www.youtube.com)", test_domain_detect_www),
        ("Domain Detection: Non-URL Rejection (notepad)", test_domain_detect_non_url),
        ("URL Normalization: Prepend https to domain", test_normalize_bare_domain),
        ("URL Normalization: Existing full HTTPS URL", test_normalize_full_url),
    ]

    for name, fn in dom_ops:
        res = benchmark_op(name, fn, iterations=2000)
        results.append(res)
        print(f"  {res['operation']:<45} : {res['mean_us']:>7.2f} µs (min: {res['min_us']:>6.2f}, max: {res['max_us']:>7.2f})")

    # --------------------------------------------------------------------------
    # 3. Action Recognition (can_handle routing)
    # --------------------------------------------------------------------------
    print("\n[3] Action Recognition (can_handle regex/intent matching)")
    print("-" * 80)

    def test_can_handle_open_url():
        skill.can_handle("open https://example.com")

    def test_can_handle_browse_domain():
        skill.can_handle("browse google.com")

    def test_can_handle_open_link():
        skill.can_handle("open link https://github.com")

    def test_can_handle_search_web():
        skill.can_handle("search the web for quantum algorithms")

    def test_can_handle_web_search():
        skill.can_handle("search for python dataclasses")

    def test_can_handle_open_browser():
        skill.can_handle("open browser")

    def test_can_handle_non_browser():
        skill.can_handle("terminate process notepad")

    rec_ops = [
        ("Action Recognition: 'open https://...'", test_can_handle_open_url),
        ("Action Recognition: 'browse google.com'", test_can_handle_browse_domain),
        ("Action Recognition: 'open link https://...'", test_can_handle_open_link),
        ("Action Recognition: 'search the web for...'", test_can_handle_search_web),
        ("Action Recognition: 'search for ...'", test_can_handle_web_search),
        ("Action Recognition: 'open browser'", test_can_handle_open_browser),
        ("Action Recognition: Negative / Non-Browser", test_can_handle_non_browser),
    ]

    for name, fn in rec_ops:
        res = benchmark_op(name, fn, iterations=2000)
        results.append(res)
        print(f"  {res['operation']:<45} : {res['mean_us']:>7.2f} µs (min: {res['min_us']:>6.2f}, max: {res['max_us']:>7.2f})")

    # --------------------------------------------------------------------------
    # 4. Full Execution Dispatch (with Mocked Browser Launch)
    # --------------------------------------------------------------------------
    print("\n[4] Full Command Execution Dispatch (Mocked Browser Launch)")
    print("-" * 80)

    with patch("webbrowser.open", mock_browser):
        cmd_open = {"action": "open_url", "target": "https://example.com"}
        cmd_search = {"action": "search_web", "target": "python async patterns"}
        cmd_browser = {"action": "open_browser"}

        def test_exec_open_url():
            skill.execute(cmd_open)

        def test_exec_search_web():
            skill.execute(cmd_search)

        def test_exec_open_browser():
            skill.execute(cmd_browser)

        exec_ops = [
            ("Execution Dispatch: open_url", test_exec_open_url),
            ("Execution Dispatch: search_web", test_exec_search_web),
            ("Execution Dispatch: open_browser", test_exec_open_browser),
        ]

        for name, fn in exec_ops:
            res = benchmark_op(name, fn, iterations=1000)
            results.append(res)
            print(f"  {res['operation']:<45} : {res['mean_us']:>7.2f} µs (min: {res['min_us']:>6.2f}, max: {res['max_us']:>7.2f})")

    print("\n" + "=" * 80)
    all_mean = [r["mean_us"] for r in results]
    print(f"Overall Average Operation Latency: {statistics.mean(all_mean):.2f} µs across {len(results)} benchmarks.")
    print("Zero real browser windows opened. Benchmark completed safely.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    main()
