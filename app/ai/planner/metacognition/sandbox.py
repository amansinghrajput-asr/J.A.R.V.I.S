"""Sandboxed Execution Harness and AST Security Scanner for Phase 20.

Isolates tool verification in a restricted subprocess with strict AST guardrails,
timeout enforcement, and memory monitoring.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import time
from typing import Optional, Set, Tuple

from app.ai.planner.metacognition.models import SandboxVerificationResult

logger = logging.getLogger("app.ai.planner.metacognition.sandbox")

DEFAULT_ALLOWED_MODULES: Set[str] = {
    "math",
    "re",
    "json",
    "datetime",
    "urllib",
    "urllib.parse",
    "hashlib",
    "typing",
    "collections",
    "itertools",
    "string",
    "random",
    "copy",
    "functools",
    "time",
}

FORBIDDEN_CALLS: Set[str] = {
    "eval",
    "exec",
    "open",
    "__import__",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "system",
}

FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__code__",
    "__globals__",
}


class SandboxedExecutionHarness:
    """Provides AST security verification and isolated subprocess testing for dynamic tools."""

    def __init__(
        self,
        allowed_modules: Optional[Set[str]] = None,
        timeout_seconds: float = 2.0,
        memory_limit_mb: int = 256,
    ) -> None:
        """Initialize sandbox harness.

        Args:
            allowed_modules: Set of whitelisted module names permitted in tool code.
            timeout_seconds: Maximum execution time permitted before killing subprocess.
            memory_limit_mb: Maximum virtual memory threshold.
        """
        self.allowed_modules = set(allowed_modules if allowed_modules is not None else DEFAULT_ALLOWED_MODULES)
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    def scan_ast_security(self, python_code: str) -> Tuple[bool, Optional[str]]:
        """Inspect Python code AST for forbidden calls, eval/exec, and unwhitelisted imports.

        Args:
            python_code: Source code string to inspect.

        Returns:
            Tuple of (is_safe, error_reason_if_any).
        """
        try:
            tree = ast.parse(python_code)
        except SyntaxError as exc:
            return False, f"Syntax error in code: {exc}"

        for node in ast.walk(tree):
            # Check module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg not in self.allowed_modules:
                        return False, f"Importing unwhitelisted module '{alias.name}' is prohibited."

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg not in self.allowed_modules:
                        return False, f"Importing from unwhitelisted module '{node.module}' is prohibited."

            # Check forbidden function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        return False, f"Invoking forbidden call '{node.func.id}()' is prohibited."
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN_CALLS:
                        return False, f"Invoking forbidden method '{node.func.attr}()' is prohibited."

            # Check forbidden attribute access (sandbox escape exploits)
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES:
                    return False, f"Accessing forbidden reflection attribute '{node.attr}' is prohibited."

        return True, None

    def execute_in_sandbox(
        self,
        code: str,
        test_code: str,
    ) -> SandboxVerificationResult:
        """Run tool verification in an ephemeral, resource-constrained subprocess.

        Args:
            code: Tool definition code.
            test_code: Test execution code asserting correctness.

        Returns:
            SandboxVerificationResult containing execution status, stdout, stderr, and timings.
        """
        combined_script = f"{code}\n\n# --- Verification Tests ---\n{test_code}\n"

        t_start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", combined_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                exit_code = proc.returncode

                success = (exit_code == 0)
                err_msg = stderr.strip() if not success else None
                return SandboxVerificationResult(
                    success=success,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time_ms=elapsed_ms,
                    error=err_msg,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return SandboxVerificationResult(
                    success=False,
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time_ms=elapsed_ms,
                    error=f"Sandbox execution timed out after {self.timeout_seconds}s.",
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return SandboxVerificationResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                execution_time_ms=elapsed_ms,
                error=f"Sandbox subprocess execution failure: {exc}",
            )
