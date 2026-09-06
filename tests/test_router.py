"""Comprehensive unit tests for the J.A.R.V.I.S Command Router Subsystem."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from app.core.config import Settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import Event, EventBus, event_bus
from app.router.intent import Intent
from app.router.router import (
    CommandPreprocessError,
    CommandRouter,
    InvalidCommandError,
    MiddlewareError,
    RouterError,
    RoutingError,
    command_router,
)
from app.skills.base import BaseSkill
from app.skills.manager import SkillManager


class DummyStringSkill(BaseSkill):
    """Mock skill that handles string commands starting with 'greet'."""

    name = "dummy_string"
    description = "Handles greeting commands"
    priority = 100

    def can_handle(self, command: Any) -> bool:
        if isinstance(command, str):
            return command.startswith("greet") or command.startswith("hello")
        return False

    def execute(self, command: Any) -> str:
        return f"Greetings: {command}"


class DummyIntentSkill(BaseSkill):
    """Mock skill that specifically handles Intent objects with intent_name='calculator'."""

    name = "dummy_intent"
    description = "Handles calculator intent"
    priority = 150

    def can_handle(self, command: Any) -> bool:
        if isinstance(command, Intent):
            return command.intent_name == "calculator"
        return False

    def execute(self, command: Any) -> int:
        if isinstance(command, Intent):
            a = command.parameters.get("a", 0)
            b = command.parameters.get("b", 0)
            return a + b
        return 0


class DummyAsyncSkill(BaseSkill):
    """Mock skill that executes asynchronously via coroutine."""

    name = "dummy_async"
    description = "Asynchronous execution skill"
    priority = 100

    def can_handle(self, command: Any) -> bool:
        if isinstance(command, Intent):
            return command.intent_name == "async_task"
        if isinstance(command, str):
            return command.startswith("async_task")
        return False

    async def execute(self, command: Any) -> str:
        await asyncio.sleep(0.01)
        return "async_result_done"


class DummyFailingSkill(BaseSkill):
    """Mock skill that raises an exception during execution."""

    name = "dummy_failing"
    description = "Always fails"
    priority = 200

    def can_handle(self, command: Any) -> bool:
        if isinstance(command, (str, Intent)):
            text = command if isinstance(command, str) else command.raw_command
            return "fail" in text
        return False

    def execute(self, command: Any) -> Any:
        raise ValueError("Deliberate skill failure for testing.")


class TestIntent(unittest.TestCase):
    """Unit tests for the Intent dataclass."""

    def test_intent_defaults(self) -> None:
        intent = Intent(raw_command="Open Chrome")
        self.assertEqual(intent.raw_command, "Open Chrome")
        self.assertEqual(intent.normalized_command, "open chrome")
        self.assertEqual(intent.intent_name, "unknown")
        self.assertEqual(intent.confidence, 1.0)
        self.assertEqual(intent.parameters, {})
        self.assertIsInstance(intent.timestamp, float)
        self.assertGreater(intent.timestamp, 0)
        self.assertEqual(intent.source, "text")
        self.assertEqual(intent.context, {})
        # UUID check
        self.assertTrue(uuid.UUID(intent.id))

    def test_intent_custom_values(self) -> None:
        custom_id = str(uuid.uuid4())
        intent = Intent(
            raw_command="Search AI",
            normalized_command="search ai",
            intent_name="web_search",
            confidence=0.95,
            parameters={"query": "AI"},
            timestamp=123456.0,
            id=custom_id,
            source="voice",
            context={"conversation_id": "conv-42", "window": "desktop"},
        )
        self.assertEqual(intent.raw_command, "Search AI")
        self.assertEqual(intent.normalized_command, "search ai")
        self.assertEqual(intent.intent_name, "web_search")
        self.assertEqual(intent.confidence, 0.95)
        self.assertEqual(intent.parameters, {"query": "AI"})
        self.assertEqual(intent.timestamp, 123456.0)
        self.assertEqual(intent.id, custom_id)
        self.assertEqual(intent.source, "voice")
        self.assertEqual(intent.context, {"conversation_id": "conv-42", "window": "desktop"})

    def test_intent_confidence_clamping(self) -> None:
        intent_high = Intent(raw_command="test", confidence=1.5)
        self.assertEqual(intent_high.confidence, 1.0)

        intent_low = Intent(raw_command="test", confidence=-0.5)
        self.assertEqual(intent_low.confidence, 0.0)

        intent_invalid = Intent(raw_command="test", confidence="invalid")  # type: ignore
        self.assertEqual(intent_invalid.confidence, 0.0)

    def test_intent_to_dict_and_from_dict(self) -> None:
        original = Intent(
            raw_command="Volume up",
            normalized_command="volume up",
            intent_name="media_control",
            confidence=0.9,
            parameters={"step": 5},
            timestamp=100.0,
            id=str(uuid.uuid4()),
            source="cli",
            context={"battery": 95},
        )
        data = original.to_dict()
        self.assertEqual(data["id"], original.id)
        self.assertEqual(data["raw_command"], "Volume up")
        self.assertEqual(data["normalized_command"], "volume up")
        self.assertEqual(data["intent_name"], "media_control")
        self.assertEqual(data["confidence"], 0.9)
        self.assertEqual(data["parameters"], {"step": 5})
        self.assertEqual(data["timestamp"], 100.0)
        self.assertEqual(data["source"], "cli")
        self.assertEqual(data["context"], {"battery": 95})

        restored = Intent.from_dict(data)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.raw_command, original.raw_command)
        self.assertEqual(restored.normalized_command, original.normalized_command)
        self.assertEqual(restored.intent_name, original.intent_name)
        self.assertEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.parameters, original.parameters)
        self.assertEqual(restored.timestamp, original.timestamp)
        self.assertEqual(restored.source, original.source)
        self.assertEqual(restored.context, original.context)

    def test_intent_from_dict_invalid(self) -> None:
        with self.assertRaises(TypeError):
            Intent.from_dict("not a dict")  # type: ignore

    def test_intent_repr(self) -> None:
        intent = Intent(raw_command="test", intent_name="test_intent")
        self.assertIn("test_intent", repr(intent))
        self.assertIn("text", repr(intent))


class TestCommandRouter(unittest.TestCase):
    """Unit tests for the CommandRouter class."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.skill_manager = SkillManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.router = CommandRouter(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            skill_manager_instance=self.skill_manager,
            auto_register_in_container=False,
        )

    def tearDown(self) -> None:
        self.router.clear_aliases()
        self.router.clear_preprocessors()
        self.router.clear_middleware()
        self.router.clear_intent_rules()
        self.skill_manager.clear()

    # --------------------------------------------------------------------------
    # Initialization and Integrations
    # --------------------------------------------------------------------------

    def test_router_initialization_with_injected_dependencies(self) -> None:
        self.assertIs(self.router.container, self.container)
        self.assertIs(self.router.event_bus, self.event_bus)
        self.assertIs(self.router.skill_manager, self.skill_manager)
        self.assertIsNotNone(self.router.logger)
        self.assertIsNotNone(self.router.config)

    def test_router_auto_registers_in_container(self) -> None:
        test_container = ServiceContainer()
        router_inst = CommandRouter(
            container_instance=test_container,
            auto_register_in_container=True,
        )
        self.assertTrue(test_container.exists("command_router"))
        self.assertTrue(test_container.exists("router"))
        self.assertIs(test_container.resolve("command_router"), router_inst)
        self.assertIs(test_container.resolve("router"), router_inst)

    def test_global_singleton_instance(self) -> None:
        self.assertIsInstance(command_router, CommandRouter)

    # --------------------------------------------------------------------------
    # Command Normalization
    # --------------------------------------------------------------------------

    def test_normalize_whitespace_and_casing(self) -> None:
        cmd = "   HELLO    WORLD   \t  \n  "
        norm = self.router.normalize(cmd)
        self.assertEqual(norm, "hello world")

    def test_normalize_trailing_punctuation(self) -> None:
        self.assertEqual(self.router.normalize("What is the time?"), "what is the time")
        self.assertEqual(self.router.normalize("Turn off lights!"), "turn off lights")
        self.assertEqual(self.router.normalize("Check status..."), "check status")

    def test_normalize_wake_word_stripping(self) -> None:
        self.assertEqual(self.router.normalize("Hey Jarvis, what is the weather?"), "what is the weather")
        self.assertEqual(self.router.normalize("Jarvis: open browser"), "open browser")
        self.assertEqual(self.router.normalize("Jarvis open browser"), "open browser")
        self.assertEqual(self.router.normalize("hey jarvis"), "")

    def test_normalize_empty_and_non_string(self) -> None:
        self.assertEqual(self.router.normalize(""), "")
        self.assertEqual(self.router.normalize("   "), "")
        self.assertEqual(self.router.normalize(None), "")  # type: ignore
        self.assertEqual(self.router.normalize(1234), "1234")  # type: ignore

    # --------------------------------------------------------------------------
    # Alias Management
    # --------------------------------------------------------------------------

    def test_register_and_has_alias(self) -> None:
        self.router.register_alias("cls", "clear screen")
        self.assertTrue(self.router.has_alias("cls"))
        self.assertTrue(self.router.has_alias("CLS"))
        self.assertEqual(self.router.get_alias("cls"), "clear screen")

    def test_unregister_alias(self) -> None:
        self.router.register_alias("np", "notepad")
        self.assertTrue(self.router.unregister_alias("np"))
        self.assertFalse(self.router.has_alias("np"))
        self.assertFalse(self.router.unregister_alias("np"))

    def test_register_alias_invalid(self) -> None:
        with self.assertRaises(ValueError):
            self.router.register_alias("", "target")
        with self.assertRaises(ValueError):
            self.router.register_alias("alias", "")

    def test_list_and_clear_aliases(self) -> None:
        self.router.register_alias("a1", "target 1")
        self.router.register_alias("a2", "target 2")
        self.assertEqual(len(self.router.list_aliases()), 2)

        self.router.clear_aliases()
        self.assertEqual(len(self.router.list_aliases()), 0)

    def test_resolve_exact_alias(self) -> None:
        self.router.register_alias("gh", "github")
        self.assertEqual(self.router.resolve_alias("gh"), "github")
        self.assertEqual(self.router.resolve_alias("GH"), "github")

    def test_resolve_prefix_alias_with_arguments(self) -> None:
        self.router.register_alias("g", "search google")
        resolved = self.router.resolve_alias("g python tutorials")
        self.assertEqual(resolved, "search google python tutorials")

    def test_resolve_alias_recursion_cycle_protection(self) -> None:
        self.router.register_alias("a", "b")
        self.router.register_alias("b", "a")
        result = self.router.resolve_alias("a")
        self.assertIn(result, ["a", "b"])

    # --------------------------------------------------------------------------
    # Command Preprocessing
    # --------------------------------------------------------------------------

    def test_preprocessor_execution_and_priority(self) -> None:
        execution_order = []

        def prep_low(cmd: str) -> str:
            execution_order.append("low")
            return cmd + " [low]"

        def prep_high(cmd: str) -> str:
            execution_order.append("high")
            return cmd + " [high]"

        self.router.register_preprocessor(prep_low, priority=50)
        self.router.register_preprocessor(prep_high, priority=200)

        processed = self.router._apply_preprocessors("test")
        self.assertEqual(execution_order, ["high", "low"])
        self.assertEqual(processed, "test [high] [low]")

    def test_preprocessor_decorator_syntax(self) -> None:
        @self.router.register_preprocessor(priority=150)
        def custom_prep(cmd: str) -> str:
            return cmd.replace("foo", "bar")

        processed = self.router._apply_preprocessors("hello foo")
        self.assertEqual(processed, "hello bar")

    def test_unregister_preprocessor(self) -> None:
        def my_prep(cmd: str) -> str:
            return cmd.upper()

        unsub = self.router.register_preprocessor(my_prep)
        self.assertEqual(self.router._apply_preprocessors("hi"), "HI")

        self.assertTrue(unsub())
        self.assertEqual(self.router._apply_preprocessors("hi"), "hi")

    def test_preprocessor_error_handling(self) -> None:
        def broken_prep(cmd: str) -> str:
            raise RuntimeError("Preprocessor explosion")

        self.router.register_preprocessor(broken_prep)

        with self.assertRaises(CommandPreprocessError):
            self.router._apply_preprocessors("test command")

    # --------------------------------------------------------------------------
    # Middleware Pipeline
    # --------------------------------------------------------------------------

    def test_middleware_registration_and_execution(self) -> None:
        history = []

        def logging_middleware(intent: Intent, next_handler: Callable[[Intent], Any]) -> Any:
            history.append(f"before:{intent.intent_name}")
            res = next_handler(intent)
            history.append(f"after:{res}")
            return res

        unsub = self.router.register_middleware(logging_middleware)
        self.assertEqual(len(self.router.list_middleware()), 1)

        self.skill_manager.register(DummyStringSkill())
        result = self.router.route("greet Alice")

        self.assertEqual(result, "Greetings: greet alice")
        self.assertEqual(history, ["before:greet", "after:Greetings: greet alice"])

        self.assertTrue(unsub())
        self.assertEqual(len(self.router.list_middleware()), 0)

    def test_middleware_priority_chain(self) -> None:
        order = []

        def mid_low(intent: Intent, next_handler: Callable[[Intent], Any]) -> Any:
            order.append("low_enter")
            res = next_handler(intent)
            order.append("low_exit")
            return res

        def mid_high(intent: Intent, next_handler: Callable[[Intent], Any]) -> Any:
            order.append("high_enter")
            res = next_handler(intent)
            order.append("high_exit")
            return res

        self.router.register_middleware(mid_low, priority=50)
        self.router.register_middleware(mid_high, priority=200)

        self.skill_manager.register(DummyStringSkill())
        self.router.route("greet Bob")

        self.assertEqual(order, ["high_enter", "low_enter", "low_exit", "high_exit"])

    def test_middleware_decorating_intent_context(self) -> None:
        def telemetry_middleware(intent: Intent, next_handler: Callable[[Intent], Any]) -> Any:
            intent.context["telemetry_injected"] = True
            return next_handler(intent)

        self.router.register_middleware(telemetry_middleware)
        self.skill_manager.register(DummyStringSkill())

        received_events: list[Event] = []
        self.event_bus.subscribe("command.completed", lambda e: received_events.append(e))

        self.router.route("greet Charlie")

        completed_evt = received_events[0]
        self.assertTrue(completed_evt.payload["intent"]["context"]["telemetry_injected"])

    # --------------------------------------------------------------------------
    # Intent Classification Rules (Deterministic)
    # --------------------------------------------------------------------------

    def test_register_intent_rule_with_named_groups(self) -> None:
        pattern = r"^open\s+(?P<app>[a-z0-9_\-]+)$"
        self.router.register_intent_rule(
            intent_name="open_application",
            pattern=pattern,
        )

        intent = self.router.parse_intent("open chrome")
        self.assertEqual(intent.intent_name, "open_application")
        self.assertEqual(intent.parameters, {"app": "chrome"})
        self.assertEqual(intent.confidence, 1.0)
        self.assertEqual(intent.source, "text")

    def test_register_intent_rule_with_custom_extractor(self) -> None:
        pattern = r"^add\s+(\d+)\s+and\s+(\d+)$"

        def extract_nums(match: re.Match[str]) -> dict[str, Any]:
            return {"a": int(match.group(1)), "b": int(match.group(2))}

        self.router.register_intent_rule(
            intent_name="calculator",
            pattern=pattern,
            parameters_extractor=extract_nums,
        )

        intent = self.router.parse_intent("add 15 and 25")
        self.assertEqual(intent.intent_name, "calculator")
        self.assertEqual(intent.parameters, {"a": 15, "b": 25})

    def test_parse_intent_fallback_to_first_token(self) -> None:
        intent = self.router.parse_intent("weather in Tokyo tomorrow", source="voice")
        self.assertEqual(intent.intent_name, "weather")
        self.assertEqual(intent.parameters, {"args": ["in", "tokyo", "tomorrow"]})
        self.assertEqual(intent.source, "voice")

    # --------------------------------------------------------------------------
    # Synchronous Command Routing
    # --------------------------------------------------------------------------

    def test_route_to_string_based_skill(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        result = self.router.route("greet Alice")
        self.assertEqual(result, "Greetings: greet alice")

    def test_route_to_intent_based_skill(self) -> None:
        self.skill_manager.register(DummyIntentSkill())
        self.router.register_intent_rule(
            intent_name="calculator",
            pattern=r"^add\s+(\d+)\s+and\s+(\d+)$",
            parameters_extractor=lambda m: {"a": int(m.group(1)), "b": int(m.group(2))},
        )

        result = self.router.route("add 7 and 8")
        self.assertEqual(result, 15)

    def test_route_with_explicit_skill_name(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        result = self.router.route("arbitrary text", skill_name="dummy_string")
        self.assertIn("Greetings:", result)

    def test_route_with_source_and_context(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        received_events: list[Event] = []
        self.event_bus.subscribe("command.completed", lambda e: received_events.append(e))

        self.router.route(
            "greet Dave",
            source="voice",
            context={"conversation_id": "conv-101"},
        )

        evt_payload = received_events[0].payload
        self.assertEqual(evt_payload["intent"]["source"], "voice")
        self.assertEqual(evt_payload["intent"]["context"]["conversation_id"], "conv-101")

    def test_route_empty_command_raises_invalid_command_error(self) -> None:
        with self.assertRaises(InvalidCommandError):
            self.router.route("")

        with self.assertRaises(InvalidCommandError):
            self.router.route("   ")

    def test_route_no_capable_skill_raises_routing_error(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        with self.assertRaises(RoutingError):
            self.router.route("unknown command that no skill handles")

    def test_route_failing_skill_raises_routing_error(self) -> None:
        self.skill_manager.register(DummyFailingSkill())
        with self.assertRaises(RoutingError):
            self.router.route("please fail now")

    def test_route_with_alias_expansion(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        self.router.register_alias("hi", "greet")

        result = self.router.route("hi Bob")
        self.assertEqual(result, "Greetings: greet bob")

    # --------------------------------------------------------------------------
    # Event Publishing Lifecycle (received, routed, completed, failed)
    # --------------------------------------------------------------------------

    def test_events_published_on_successful_route(self) -> None:
        received_events: list[Event] = []

        self.event_bus.subscribe("command.received", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.routed", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.completed", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.failed", lambda e: received_events.append(e))

        self.skill_manager.register(DummyStringSkill())
        res = self.router.route("greet Charlie")
        self.assertEqual(res, "Greetings: greet charlie")

        event_names = [e.name for e in received_events]
        self.assertIn("command.received", event_names)
        self.assertIn("command.routed", event_names)
        self.assertIn("command.completed", event_names)
        self.assertNotIn("command.failed", event_names)

        # Check routed event payload
        routed_event = next(e for e in received_events if e.name == "command.routed")
        self.assertEqual(routed_event.payload["command"], "greet charlie")
        self.assertEqual(routed_event.payload["raw_command"], "greet Charlie")
        self.assertEqual(routed_event.payload["skill_name"], "dummy_string")

        # Check completed event payload
        completed_event = next(e for e in received_events if e.name == "command.completed")
        self.assertTrue(completed_event.payload["success"])
        self.assertEqual(completed_event.payload["result"], "Greetings: greet charlie")
        self.assertEqual(completed_event.payload["skill_name"], "dummy_string")
        self.assertIn("duration", completed_event.payload)

    def test_events_published_on_failed_route(self) -> None:
        received_events: list[Event] = []

        self.event_bus.subscribe("command.received", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.routed", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.completed", lambda e: received_events.append(e))
        self.event_bus.subscribe("command.failed", lambda e: received_events.append(e))

        with self.assertRaises(RoutingError):
            self.router.route("unhandled command")

        event_names = [e.name for e in received_events]
        self.assertIn("command.received", event_names)
        self.assertIn("command.failed", event_names)
        self.assertIn("command.completed", event_names)

        # Verify completed event has success=False
        completed_event = next(e for e in received_events if e.name == "command.completed")
        self.assertFalse(completed_event.payload["success"])
        self.assertIsNone(completed_event.payload["result"])

        failed_event = next(e for e in received_events if e.name == "command.failed")
        self.assertEqual(failed_event.payload["command"], "unhandled command")
        self.assertIn("error", failed_event.payload)

    # --------------------------------------------------------------------------
    # Asynchronous Command Routing
    # --------------------------------------------------------------------------

    def test_route_async_successful(self) -> None:
        async def run_test() -> None:
            self.skill_manager.register(DummyAsyncSkill())

            result = await self.router.route_async("async_task run now")
            self.assertEqual(result, "async_result_done")

        asyncio.run(run_test())

    def test_route_async_with_async_middleware(self) -> None:
        async def run_test() -> None:
            log_entries = []

            async def async_mid(intent: Intent, next_h: Callable[[Intent], Any]) -> Any:
                log_entries.append("async_mid_in")
                res = await next_h(intent)
                log_entries.append("async_mid_out")
                return res

            self.router.register_middleware(async_mid)
            self.skill_manager.register(DummyAsyncSkill())

            result = await self.router.route_async("async_task go")
            self.assertEqual(result, "async_result_done")
            self.assertEqual(log_entries, ["async_mid_in", "async_mid_out"])

        asyncio.run(run_test())

    def test_route_async_with_async_preprocessor(self) -> None:
        async def run_test() -> None:
            self.skill_manager.register(DummyStringSkill())

            async def async_prep(cmd: str) -> str:
                await asyncio.sleep(0.01)
                return cmd.replace("salute", "greet")

            self.router.register_preprocessor(async_prep)

            result = await self.router.route_async("salute World")
            self.assertEqual(result, "Greetings: greet world")

        asyncio.run(run_test())

    def test_route_async_events_published(self) -> None:
        async def run_test() -> None:
            events = []
            self.event_bus.subscribe("command.received", lambda e: events.append(e.name))
            self.event_bus.subscribe("command.routed", lambda e: events.append(e.name))
            self.event_bus.subscribe("command.completed", lambda e: events.append(e.name))

            self.skill_manager.register(DummyAsyncSkill())
            await self.router.route_async("async_task hello")

            self.assertIn("command.received", events)
            self.assertIn("command.routed", events)
            self.assertIn("command.completed", events)

        asyncio.run(run_test())

    # --------------------------------------------------------------------------
    # Thread Safety & Concurrency
    # --------------------------------------------------------------------------

    def test_concurrent_command_routing(self) -> None:
        self.skill_manager.register(DummyStringSkill())
        concurrency = 20
        results = []

        def worker(idx: int) -> str:
            return self.router.route(f"greet worker_{idx}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(concurrency)]
            for future in as_completed(futures):
                results.append(future.result())

        self.assertEqual(len(results), concurrency)
        for i in range(concurrency):
            self.assertIn(f"Greetings: greet worker_{i}", results)

    def test_concurrent_alias_registration_and_lookup(self) -> None:
        num_aliases = 50

        def register_worker(idx: int) -> None:
            self.router.register_alias(f"alias_{idx}", f"target_{idx}")

        def lookup_worker(idx: int) -> None:
            self.router.resolve_alias(f"alias_{idx}")
            self.router.list_aliases()

        with ThreadPoolExecutor(max_workers=8) as executor:
            reg_futures = [executor.submit(register_worker, i) for i in range(num_aliases)]
            look_futures = [executor.submit(lookup_worker, i) for i in range(num_aliases)]
            for f in as_completed(reg_futures + look_futures):
                f.result()

        aliases = self.router.list_aliases()
        self.assertEqual(len(aliases), num_aliases)


if __name__ == "__main__":
    unittest.main()
