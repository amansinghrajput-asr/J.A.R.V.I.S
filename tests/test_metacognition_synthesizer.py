"""Unit tests for Phase 20 Sprint 20.2 Dynamic Tool Synthesizer & Runtime Hot-Binding.

Covers:
- ToolSynthesizer verification workflow
- SyntheticTestGenerator determinism
- SandboxedExecutionHarness AST scanning and isolated subprocess execution
- DynamicToolRegistry registration, lookup, duplicates, and overwrite
- Retry loops and max attempt exhaustion
- PlannerEventBus lifecycle event emissions
- Dependency injection and zero global state
- Executor runtime dynamic tool resolution (Priority 3.5)
- Concurrent synthesis and thread safety
"""

import concurrent.futures
import threading
import unittest
from typing import Any, Dict, List

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    ToolSynthesisRejected,
    ToolSynthesisStarted,
    ToolSynthesisVerified,
)
from app.ai.planner.executor import Executor
from app.ai.planner.metacognition.models import (
    ToolSynthesisStatus,
)
from app.ai.planner.metacognition.registry import DynamicToolRegistry
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.metacognition.synthesizer import (
    SyntheticTestGenerator,
    ToolSynthesizer,
)
from app.ai.planner.models import Plan, Task


class TestDynamicToolRegistry(unittest.TestCase):
    """Test suite for DynamicToolRegistry."""

    def setUp(self) -> None:
        self.registry = DynamicToolRegistry()

    def test_registration_and_resolution(self) -> None:
        """Verify registering and resolving a tool callable."""
        def mock_tool(x: int) -> int:
            return x * 2

        self.assertTrue(self.registry.register("double", mock_tool))
        self.assertTrue(self.registry.contains("double"))
        resolved = self.registry.resolve("double")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved(5), 10)

    def test_duplicate_registration_prevention(self) -> None:
        """Verify registering a duplicate action name raises ValueError when overwrite is False."""
        self.registry.register("tool_a", lambda: 1)
        with self.assertRaises(ValueError):
            self.registry.register("tool_a", lambda: 2, overwrite=False)

    def test_overwrite_registration(self) -> None:
        """Verify registering with overwrite=True successfully updates the tool."""
        self.registry.register("tool_b", lambda: "original")
        self.registry.register("tool_b", lambda: "updated", overwrite=True)
        resolved = self.registry.resolve("tool_b")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved(), "updated")

    def test_unregister_and_list(self) -> None:
        """Verify unregistering removes tool and list_tools returns sorted keys."""
        self.registry.register("zebra", lambda: 1)
        self.registry.register("alpha", lambda: 2)
        self.assertEqual(self.registry.list_tools(), ["alpha", "zebra"])

        self.assertTrue(self.registry.unregister("zebra"))
        self.assertFalse(self.registry.contains("zebra"))
        self.assertEqual(self.registry.list_tools(), ["alpha"])

    def test_concurrent_registrations(self) -> None:
        """Verify thread-safety during high-concurrency registrations."""
        def _worker(idx: int) -> None:
            self.registry.register(f"tool_{idx}", lambda i=idx: i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_worker, i) for i in range(50)]
            for f in futures:
                f.result()

        self.assertEqual(len(self.registry.list_tools()), 50)


class TestSyntheticTestGenerator(unittest.TestCase):
    """Test suite for SyntheticTestGenerator determinism and reproducibility."""

    def setUp(self) -> None:
        self.generator = SyntheticTestGenerator()

    def test_deterministic_generation(self) -> None:
        """Verify repeated calls for same action produce identical test code."""
        t1 = self.generator.generate_tests("add_numbers", {}, {})
        t2 = self.generator.generate_tests("add_numbers", {}, {})
        self.assertEqual(t1, t2)
        self.assertIn("assert add_numbers(2, 3) == 5", t1)

    def test_schema_driven_fallback_generation(self) -> None:
        """Verify generic action generates valid assertion calls."""
        input_schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "name": {"type": "string"},
            },
        }
        tests = self.generator.generate_tests("custom_func", input_schema, {})
        self.assertIn("custom_func(count=1, name='test')", tests)
        self.assertIn("assert res is not None", tests)


class TestSandboxedExecutionHarness(unittest.TestCase):
    """Test suite for SandboxedExecutionHarness AST scanner and subprocess isolation."""

    def setUp(self) -> None:
        self.sandbox = SandboxedExecutionHarness(timeout_seconds=2.0)

    def test_ast_security_allows_safe_code(self) -> None:
        """Verify valid code with whitelisted modules passes AST validation."""
        code = "import math\ndef add(a, b):\n    return a + b\n"
        is_safe, err = self.sandbox.scan_ast_security(code)
        self.assertTrue(is_safe)
        self.assertIsNone(err)

    def test_ast_security_rejects_unwhitelisted_imports(self) -> None:
        """Verify importing unwhitelisted modules like 'os' or 'subprocess' is blocked."""
        code = "import os\ndef bad():\n    return os.listdir('.')\n"
        is_safe, err = self.sandbox.scan_ast_security(code)
        self.assertFalse(is_safe)
        self.assertIn("Importing unwhitelisted module 'os'", err)

    def test_ast_security_rejects_eval_and_exec(self) -> None:
        """Verify calling eval() or exec() is blocked."""
        code = "def bad(code_str):\n    return eval(code_str)\n"
        is_safe, err = self.sandbox.scan_ast_security(code)
        self.assertFalse(is_safe)
        self.assertIn("Invoking forbidden call 'eval()'", err)

    def test_ast_security_rejects_subclasses_exploit(self) -> None:
        """Verify accessing __subclasses__ reflection attribute is blocked."""
        code = "def bad():\n    return ''.__class__.__mro__[1].__subclasses__()\n"
        is_safe, err = self.sandbox.scan_ast_security(code)
        self.assertFalse(is_safe)
        self.assertIn("forbidden reflection attribute", err)

    def test_execute_in_sandbox_success(self) -> None:
        """Verify executing valid tool and tests inside sandbox subprocess succeeds."""
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(2, 3) == 5\n"
        res = self.sandbox.execute_in_sandbox(code, tests)
        self.assertTrue(res.success)
        self.assertEqual(res.exit_code, 0)
        self.assertGreater(res.execution_time_ms, 0.0)

    def test_execute_in_sandbox_assertion_failure(self) -> None:
        """Verify failed assertion in sandbox returns success=False with stderr."""
        code = "def add(a, b):\n    return a - b\n"
        tests = "assert add(2, 3) == 5\n"
        res = self.sandbox.execute_in_sandbox(code, tests)
        self.assertFalse(res.success)
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("AssertionError", res.stderr)


class TestToolSynthesizer(unittest.TestCase):
    """Test suite for ToolSynthesizer workflow, retries, and events."""

    def setUp(self) -> None:
        self.sandbox = SandboxedExecutionHarness(timeout_seconds=2.0)
        self.event_bus = PlannerEventBus()
        self.registry = DynamicToolRegistry()
        self.events_received: List[PlannerEvent] = []
        self.event_bus.subscribe(PlannerEvent, lambda e: self.events_received.append(e))

    def test_successful_synthesis_and_hot_binding(self) -> None:
        """Verify complete synthesis workflow produces promoted tool and hot-binds to registry."""
        synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            event_bus=self.event_bus,
            registry=self.registry,
        )

        res = synthesizer.synthesize_tool(
            action="add_numbers",
            description="Add two integers",
            input_schema={"type": "object"},
            output_schema={"type": "integer"},
        )

        self.assertEqual(res.status, ToolSynthesisStatus.VERIFIED_AND_PROMOTED)
        self.assertIsNotNone(res.callable_tool)
        self.assertEqual(res.callable_tool(10, 20), 30)

        # Verify hot-binding in registry
        self.assertTrue(self.registry.contains("add_numbers"))
        resolved = self.registry.resolve("add_numbers")
        self.assertEqual(resolved(4, 5), 9)

        # Verify events emitted
        event_types = [type(e) for e in self.events_received]
        self.assertIn(ToolSynthesisStarted, event_types)
        self.assertIn(ToolSynthesisVerified, event_types)

    def test_ast_rejection_and_retry(self) -> None:
        """Verify AST violation triggers retry and reports AST_VALIDATION_FAILED if uncorrected."""
        def _bad_code_gen(action: str, ctx: Dict[str, Any]) -> str:
            # Emits code calling eval
            return f"def {action}(x):\n    return eval(x)\n"

        synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            code_generator=_bad_code_gen,
            event_bus=self.event_bus,
            registry=self.registry,
        )

        res = synthesizer.synthesize_tool(
            action="danger_tool",
            description="Runs eval",
            input_schema={},
            output_schema={},
            max_attempts=2,
        )

        self.assertEqual(res.status, ToolSynthesisStatus.AST_VALIDATION_FAILED)
        self.assertIn("AST Security Violation", res.error)
        self.assertEqual(res.iteration_count, 2)
        self.assertFalse(self.registry.contains("danger_tool"))

        # Verify rejection event emitted
        event_types = [type(e) for e in self.events_received]
        self.assertIn(ToolSynthesisRejected, event_types)

    def test_sandbox_failure_and_retry_recovery(self) -> None:
        """Verify self-healing retry loop recovers when second attempt provides correct code."""
        attempt_tracker = {"count": 0}

        def _healing_code_gen(action: str, ctx: Dict[str, Any]) -> str:
            attempt_tracker["count"] += 1
            if attempt_tracker["count"] == 1:
                # First attempt produces bug
                return f"def {action}(a, b):\n    return a - b\n"
            # Second attempt fixes bug
            return f"def {action}(a, b):\n    return a + b\n"

        synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            code_generator=_healing_code_gen,
            event_bus=self.event_bus,
            registry=self.registry,
        )

        res = synthesizer.synthesize_tool(
            action="add_numbers",
            description="Self-healing adder",
            input_schema={},
            output_schema={},
            max_attempts=3,
        )

        self.assertEqual(res.status, ToolSynthesisStatus.VERIFIED_AND_PROMOTED)
        self.assertEqual(res.iteration_count, 2)
        self.assertTrue(self.registry.contains("add_numbers"))

    def test_executor_integration_priority_3_5(self) -> None:
        """Verify Executor resolves unknown action dynamically via ToolSynthesizer (Priority 3.5)."""
        synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            event_bus=self.event_bus,
            registry=self.registry,
        )
        executor = Executor(
            tool_synthesizer=synthesizer,
            dynamic_tool_registry=self.registry,
            auto_register_in_container=False,
        )

        plan = Plan(
            query="test add",
            tasks=[Task(action="add_numbers", parameters={"a": 15, "b": 25})],
        )

        exec_res = executor.execute_plan(plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(len(exec_res.completed_tasks), 1)
        self.assertEqual(len(exec_res.failed_tasks), 0)

    def test_concurrent_synthesis_isolation(self) -> None:
        """Verify multiple threads synthesizing tools concurrently do not collide."""
        synthesizer = ToolSynthesizer(
            sandbox=self.sandbox,
            registry=self.registry,
        )

        actions = ["add_numbers", "subtract", "multiply", "to_uppercase", "is_even"]

        def _run_synth(act: str) -> ToolSynthesisStatus:
            res = synthesizer.synthesize_tool(
                action=act,
                description=f"Action {act}",
                input_schema={},
                output_schema={},
            )
            return res.status

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_run_synth, a) for a in actions]
            statuses = [f.result() for f in futures]

        for s in statuses:
            self.assertEqual(s, ToolSynthesisStatus.VERIFIED_AND_PROMOTED)

        for a in actions:
            self.assertTrue(self.registry.contains(a))


if __name__ == "__main__":
    unittest.main()
