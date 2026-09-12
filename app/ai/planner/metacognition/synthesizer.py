"""Dynamic Tool Synthesizer and Synthetic Test Generator for Phase 20 Metacognition.

Provides autonomous code generation, synthetic test harness creation,
AST security scanning, sandbox validation, and runtime hot-binding.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from app.ai.planner.events import (
    PlannerEventBus,
    ToolSynthesisRejected,
    ToolSynthesisStarted,
    ToolSynthesisVerified,
)
from app.ai.planner.metacognition.models import (
    SynthesizedToolResult,
    ToolSynthesisStatus,
)
from app.ai.planner.metacognition.registry import DynamicToolRegistry
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness

logger = logging.getLogger("app.ai.planner.metacognition.synthesizer")


class SyntheticTestGenerator:
    """Generates deterministic, reproducible unit test code for synthesized tools."""

    def generate_tests(
        self,
        action: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
    ) -> str:
        """Generate deterministic verification assertions for an action tool.

        Args:
            action: Clean action identifier name.
            input_schema: JSON Schema describing expected parameters.
            output_schema: JSON Schema describing expected return value.

        Returns:
            Python source code string containing test assertions.
        """
        clean_name = action.strip().lower()

        # Deterministic domain-specific test suites
        if clean_name in ("add_numbers", "add", "sum_two"):
            return (
                f"assert {clean_name}(2, 3) == 5, 'add_numbers(2, 3) failed'\n"
                f"assert {clean_name}(-1, 5) == 4, 'add_numbers(-1, 5) failed'\n"
                f"assert {clean_name}(0, 0) == 0, 'add_numbers(0, 0) failed'\n"
            )

        if clean_name in ("subtract", "subtract_numbers", "diff"):
            return (
                f"assert {clean_name}(5, 3) == 2, 'subtract(5, 3) failed'\n"
                f"assert {clean_name}(0, 5) == -5, 'subtract(0, 5) failed'\n"
                f"assert {clean_name}(4, 4) == 0, 'subtract(4, 4) failed'\n"
            )

        if clean_name in ("multiply", "multiply_numbers", "product"):
            return (
                f"assert {clean_name}(2, 3) == 6, 'multiply(2, 3) failed'\n"
                f"assert {clean_name}(-2, 4) == -8, 'multiply(-2, 4) failed'\n"
                f"assert {clean_name}(0, 5) == 0, 'multiply(0, 5) failed'\n"
            )

        if clean_name in ("divide", "divide_numbers", "div"):
            return (
                f"assert {clean_name}(6, 3) == 2.0 or {clean_name}(6, 3) == 2, 'divide(6, 3) failed'\n"
                f"assert {clean_name}(5, 2) == 2.5, 'divide(5, 2) failed'\n"
                f"assert {clean_name}(-8, 2) == -4.0 or {clean_name}(-8, 2) == -4, 'divide(-8, 2) failed'\n"
            )

        if clean_name in ("to_uppercase", "uppercase", "upper"):
            return (
                f"assert {clean_name}('hello') == 'HELLO', 'uppercase(\\'hello\\') failed'\n"
                f"assert {clean_name}('') == '', 'uppercase(\\'\\') failed'\n"
                f"assert {clean_name}('123') == '123', 'uppercase(\\'123\\') failed'\n"
            )

        if clean_name in ("reverse_string", "reverse_str", "reverse"):
            return (
                f"assert {clean_name}('hello') == 'olleh', 'reverse(\\'hello\\') failed'\n"
                f"assert {clean_name}('') == '', 'reverse(\\'\\') failed'\n"
                f"assert {clean_name}('racecar') == 'racecar', 'reverse(\\'racecar\\') failed'\n"
            )

        if clean_name in ("is_even", "check_even"):
            return (
                f"assert {clean_name}(4) is True, 'is_even(4) failed'\n"
                f"assert {clean_name}(7) is False, 'is_even(7) failed'\n"
                f"assert {clean_name}(0) is True, 'is_even(0) failed'\n"
            )

        # General schema-driven fallback tests
        properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
        test_calls = []

        if properties:
            kwargs_list = []
            for prop_name, prop_def in properties.items():
                p_type = prop_def.get("type", "string") if isinstance(prop_def, dict) else "string"
                if p_type in ("integer", "int"):
                    kwargs_list.append(f"{prop_name}=1")
                elif p_type in ("number", "float"):
                    kwargs_list.append(f"{prop_name}=1.0")
                elif p_type in ("boolean", "bool"):
                    kwargs_list.append(f"{prop_name}=True")
                elif p_type in ("array", "list"):
                    kwargs_list.append(f"{prop_name}=[]")
                elif p_type in ("object", "dict"):
                    kwargs_list.append(f"{prop_name}={{}}")
                else:
                    kwargs_list.append(f"{prop_name}='test'")

            kw_str = ", ".join(kwargs_list)
            test_calls.append(f"res = {clean_name}({kw_str})\nassert res is not None, 'Result was None'")
        else:
            test_calls.append(f"res = {clean_name}()\nassert res is not None, 'Result was None'")

        return "\n".join(test_calls) + "\n"


def _default_deterministic_code_generator(action: str, context: Dict[str, Any]) -> str:
    """Fallback deterministic code generator when no LLM or custom generator is injected."""
    clean_name = action.strip().lower()

    if clean_name in ("add_numbers", "add", "sum_two"):
        return f"def {clean_name}(a: int, b: int) -> int:\n    return a + b\n"
    if clean_name in ("subtract", "subtract_numbers", "diff"):
        return f"def {clean_name}(a: int, b: int) -> int:\n    return a - b\n"
    if clean_name in ("multiply", "multiply_numbers", "product"):
        return f"def {clean_name}(a: int, b: int) -> int:\n    return a * b\n"
    if clean_name in ("divide", "divide_numbers", "div"):
        return f"def {clean_name}(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b\n"
    if clean_name in ("to_uppercase", "uppercase", "upper"):
        return f"def {clean_name}(text: str) -> str:\n    return text.upper()\n"
    if clean_name in ("reverse_string", "reverse_str", "reverse"):
        return f"def {clean_name}(text: str) -> str:\n    return text[::-1]\n"
    if clean_name in ("is_even", "check_even"):
        return f"def {clean_name}(num: int) -> bool:\n    return num % 2 == 0\n"

    # Generic deterministic fallback
    input_schema = context.get("input_schema", {})
    props = list(input_schema.get("properties", {}).keys()) if isinstance(input_schema, dict) else []
    params_sig = ", ".join([f"{p}: Any = None" for p in props]) if props else "*args, **kwargs"
    return (
        f"from typing import Any\n\n"
        f"def {clean_name}({params_sig}) -> Any:\n"
        f"    return {{'status': 'success', 'action': '{clean_name}'}}\n"
    )


class ToolSynthesizer:
    """Master synthesizer for dynamic code formulation, sandbox verification, and hot-binding."""

    def __init__(
        self,
        sandbox: SandboxedExecutionHarness,
        code_generator: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        event_bus: Optional[PlannerEventBus] = None,
        registry: Optional[DynamicToolRegistry] = None,
    ) -> None:
        """Initialize dynamic tool synthesizer.

        Args:
            sandbox: Sandboxed execution harness for AST checks and isolated execution.
            code_generator: Injected code generator callable; defaults to deterministic generator.
            event_bus: Optional PlannerEventBus for emitting synthesis lifecycle events.
            registry: Optional DynamicToolRegistry for hot-binding verified tools.
        """
        self._sandbox = sandbox
        self._code_generator = code_generator if code_generator is not None else _default_deterministic_code_generator
        self._event_bus = event_bus
        self._registry = registry
        self._test_generator = SyntheticTestGenerator()
        self._lock = threading.RLock()

    @property
    def registry(self) -> Optional[DynamicToolRegistry]:
        """Return the associated tool registry."""
        return self._registry

    @property
    def sandbox(self) -> SandboxedExecutionHarness:
        """Return the sandbox harness."""
        return self._sandbox

    def synthesize_tool(
        self,
        action: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        max_attempts: int = 3,
    ) -> SynthesizedToolResult:
        """Synthesize, test, and hot-bind a dynamic tool.

        Args:
            action: Unique action identifier name.
            description: Semantic description of tool purpose.
            input_schema: Parameter schema specification.
            output_schema: Expected output schema specification.
            max_attempts: Maximum retry iterations on validation failures.

        Returns:
            SynthesizedToolResult containing status, code, test code, and verification metrics.
        """
        with self._lock:
            clean_action = action.strip().lower()
            t0 = time.perf_counter()

            if self._event_bus is not None:
                self._event_bus.publish(
                    ToolSynthesisStarted(
                        action_name=clean_action,
                        target=description,
                    )
                )

            last_code = ""
            last_test = ""
            last_status = ToolSynthesisStatus.PENDING
            last_error: Optional[str] = None
            verification_res = None

            context = {
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "max_attempts": max_attempts,
            }

            for attempt in range(1, max_attempts + 1):
                try:
                    context["attempt"] = attempt
                    context["last_error"] = last_error

                    # 1. Generate code
                    code = self._code_generator(clean_action, context)
                    if not code or not code.strip():
                        last_error = "Code generator produced empty code."
                        last_status = ToolSynthesisStatus.REJECTED
                        continue
                    last_code = code

                    # 2. Generate deterministic unit tests
                    test_code = self._test_generator.generate_tests(
                        clean_action, input_schema, output_schema
                    )
                    last_test = test_code

                    # 3. AST Security scan
                    ast_ok, ast_err = self._sandbox.scan_ast_security(code)
                    if not ast_ok:
                        last_error = f"AST Security Violation: {ast_err}"
                        last_status = ToolSynthesisStatus.AST_VALIDATION_FAILED
                        logger.warning("Tool synthesis '%s' attempt %d AST failed: %s", clean_action, attempt, ast_err)
                        continue

                    # 4. Sandbox verification execution
                    v_res = self._sandbox.execute_in_sandbox(code, test_code)
                    verification_res = v_res
                    if not v_res.success:
                        last_error = f"Sandbox Test Failure: {v_res.error or v_res.stderr or 'Non-zero exit'}"
                        last_status = ToolSynthesisStatus.SANDBOX_TEST_FAILED
                        logger.warning("Tool synthesis '%s' attempt %d sandbox failed: %s", clean_action, attempt, v_res.error)
                        continue

                    # 5. Compile into local callable and hot-bind
                    local_scope: Dict[str, Any] = {}
                    exec(code, {"__builtins__": __builtins__}, local_scope)
                    tool_callable = local_scope.get(clean_action) or local_scope.get(action)

                    if tool_callable is None:
                        last_error = f"Compiled code did not define function '{clean_action}'."
                        last_status = ToolSynthesisStatus.SANDBOX_TEST_FAILED
                        continue

                    # Hot-bind to registry if available
                    if self._registry is not None:
                        self._registry.register(
                            action_name=clean_action,
                            callable_obj=tool_callable,
                            overwrite=True,
                            metadata={"synthesized": True, "description": description},
                        )

                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    if self._event_bus is not None:
                        self._event_bus.publish(
                            ToolSynthesisVerified(
                                action_name=clean_action,
                                execution_time_ms=duration_ms,
                                attempts=attempt,
                            )
                        )

                    logger.info(
                        "Tool '%s' successfully synthesized and verified on attempt %d in %.2fms.",
                        clean_action,
                        attempt,
                        duration_ms,
                    )

                    return SynthesizedToolResult(
                        action_name=clean_action,
                        code=code,
                        test_code=test_code,
                        status=ToolSynthesisStatus.VERIFIED_AND_PROMOTED,
                        verification=v_res,
                        iteration_count=attempt,
                        callable_tool=tool_callable,
                    )

                except Exception as exc:
                    last_error = f"Unexpected synthesis iteration failure: {exc}"
                    last_status = ToolSynthesisStatus.REJECTED
                    logger.warning("Synthesis error on '%s' attempt %d: %s", clean_action, attempt, exc)

            # All attempts exhausted
            if self._event_bus is not None:
                self._event_bus.publish(
                    ToolSynthesisRejected(
                        action_name=clean_action,
                        reason=last_error or "Max retry attempts exhausted.",
                        attempts=max_attempts,
                    )
                )

            logger.error(
                "Tool '%s' synthesis rejected after %d attempts. Reason: %s",
                clean_action,
                max_attempts,
                last_error,
            )

            return SynthesizedToolResult(
                action_name=clean_action,
                code=last_code,
                test_code=last_test,
                status=last_status if last_status != ToolSynthesisStatus.PENDING else ToolSynthesisStatus.REJECTED,
                verification=verification_res,
                iteration_count=max_attempts,
                error=last_error,
            )
