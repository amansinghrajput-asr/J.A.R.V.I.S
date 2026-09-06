"""Comprehensive unit and integration tests for the J.A.R.V.I.S Application Runner."""

from __future__ import annotations

import io
import logging
import sys
import unittest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from app.ai.manager import AIManager
from app.ai.models import AIResponse
from app.application import (
    EVENT_APPLICATION_READY,
    EVENT_APPLICATION_STOPPED,
    JarvisApplication,
    READY_MESSAGE,
)
from app.core.config import Settings, load_settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.memory.manager import MemoryManager
from app.router.intent import Intent
from app.router.router import CommandRouter
from app.skills.ai_skill import AISkill
from app.skills.base import BaseSkill
from app.skills.manager import SkillManager
from app.voice.pipeline import VoicePipeline
from main import main


class MockSpecializedSkill(BaseSkill):
    """Mock specialized skill with high priority to test precedence over AISkill."""

    name: str = "mock_calc"
    description: str = "Calculates mathematical expressions"
    version: str = "1.0.0"
    priority: int = 200

    def can_handle(self, command: Any) -> bool:
        cmd_str = command.raw_command if isinstance(command, Intent) else str(command)
        return "calc" in cmd_str.lower() or "calculate" in cmd_str.lower()

    def execute(self, command: Any) -> Any:
        return "Calculated result: 42"


class TestJarvisApplication(unittest.TestCase):
    """Test suite for JarvisApplication lifecycle, routing, and console interaction."""

    def setUp(self) -> None:
        """Create fresh, isolated subsystems for testing."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.logger = logging.getLogger("TEST_APPLICATION")
        self.settings = load_settings()

    def test_application_initialization_and_subsystem_wiring(self) -> None:
        """Verify JarvisApplication initializes all required subsystems and registers them in the container."""
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            app = JarvisApplication(
                container=self.container,
                config=self.settings,
                logger=self.logger,
                event_bus=self.event_bus,
                print_ready=True,
            )

        # 1. Verification of initialized properties
        self.assertIsInstance(app.container, ServiceContainer)
        self.assertIsInstance(app.ai_manager, AIManager)
        self.assertIsInstance(app.skill_manager, SkillManager)
        self.assertIsInstance(app.command_router, CommandRouter)
        self.assertIsInstance(app.voice_pipeline, VoicePipeline)
        self.assertIsInstance(app.memory_manager, MemoryManager)
        self.assertFalse(app.is_running)

        # 2. Container service registration verification
        self.assertTrue(self.container.exists("ai_manager"))
        self.assertTrue(self.container.exists("skill_manager"))
        self.assertTrue(self.container.exists("command_router"))
        self.assertTrue(self.container.exists("router"))
        self.assertTrue(self.container.exists("voice_pipeline"))
        self.assertTrue(self.container.exists("application"))
        self.assertTrue(self.container.exists("app"))
        self.assertIs(self.container.resolve("application"), app)

        # 3. Ready message printed to stdout
        self.assertIn(READY_MESSAGE, captured_stdout.getvalue())

    def test_print_ready_flag_controls_stdout_output(self) -> None:
        """Verify print_ready=False suppresses printing the ready banner."""
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            _ = JarvisApplication(
                container=self.container,
                config=self.settings,
                logger=self.logger,
                event_bus=self.event_bus,
                print_ready=False,
            )

        self.assertNotIn(READY_MESSAGE, captured_stdout.getvalue())

    def test_property_aliases(self) -> None:
        """Verify convenience property aliases."""
        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            print_ready=False,
        )

        self.assertIs(app.router, app.command_router)
        self.assertIs(app.pipeline, app.voice_pipeline)
        self.assertIs(app.memory, app.memory_manager)
        self.assertIs(app.config, app.settings)

    def test_auto_discover_skills_registers_fallback_ai_skill(self) -> None:
        """Verify automatic skill discovery discovers skills and ensures AISkill is registered."""
        app = JarvisApplication(
            container=self.container,
            config=self.settings,
            logger=self.logger,
            event_bus=self.event_bus,
            auto_discover_skills=True,
            print_ready=False,
        )

        self.assertTrue(app.skill_manager.has_skill("ai"))
        ai_skill_inst = app.skill_manager.get("ai")
        self.assertIsInstance(ai_skill_inst, AISkill)
        self.assertEqual(ai_skill_inst.priority, -100)

    def test_interactive_console_loop_exit_and_quit(self) -> None:
        """Verify interactive console loop terminates cleanly on 'exit' and 'quit'."""
        for exit_cmd in ("exit", "quit", "EXIT", "Quit", "  exit  "):
            app = JarvisApplication(
                container=ServiceContainer(),
                config=self.settings,
                logger=self.logger,
                event_bus=EventBus(),
                print_ready=False,
            )

            inputs = [exit_cmd]
            outputs: list[str] = []

            def mock_input(prompt: str) -> str:
                self.assertEqual(prompt, "You > ")
                return inputs.pop(0)

            def mock_output(msg: str) -> None:
                outputs.append(msg)

            exit_code = app.run(input_fn=mock_input, output_fn=mock_output)
            self.assertEqual(exit_code, 0)
            self.assertFalse(app.is_running)

    def test_interactive_console_loop_command_routing_and_response_format(self) -> None:
        """Verify user commands are routed through CommandRouter and displayed to user."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )

        # Mock command router route response
        app.command_router.route = MagicMock(
            return_value=AIResponse(content="Hello! How can I help you?")
        )

        inputs = ["hello", "exit"]
        outputs: list[str] = []

        def mock_input(prompt: str) -> str:
            return inputs.pop(0)

        def mock_output(msg: str) -> None:
            outputs.append(msg)

        exit_code = app.run(input_fn=mock_input, output_fn=mock_output)

        self.assertEqual(exit_code, 0)
        app.command_router.route.assert_called_once_with("hello")
        self.assertTrue(any("Jarvis > Hello! How can I help you?" in out for out in outputs))

    def test_specialized_skill_priority_over_ai_skill(self) -> None:
        """Verify specialized skills execute first and AISkill is used as conversational fallback."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )

        # Register specialized skill
        specialized_skill = MockSpecializedSkill()
        app.skill_manager.register(specialized_skill)

        # Mock AI Manager generation for fallback
        app.ai_manager.generate = MagicMock(
            return_value=AIResponse(content="AI conversational fallback answer.")
        )

        # 1. Specialized command routes to MockSpecializedSkill
        calc_result = app.process_command("calculate 20 + 22")
        self.assertEqual(calc_result, "Calculated result: 42")
        app.ai_manager.generate.assert_not_called()

        # 2. General conversational query routes to AISkill fallback
        fallback_result = app.process_command("Tell me about quantum computing")
        self.assertIsInstance(fallback_result, AIResponse)
        self.assertEqual(fallback_result.content, "AI conversational fallback answer.")
        app.ai_manager.generate.assert_called_once()

    def test_interactive_loop_exception_recovery(self) -> None:
        """Verify that errors during command execution are caught and displayed without crashing."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )

        # Mock command router route to raise an error
        app.command_router.route = MagicMock(side_effect=RuntimeError("Simulated provider failure"))

        inputs = ["failing command", "exit"]
        outputs: list[str] = []

        def mock_input(prompt: str) -> str:
            return inputs.pop(0)

        def mock_output(msg: str) -> None:
            outputs.append(msg)

        exit_code = app.run(input_fn=mock_input, output_fn=mock_output)

        self.assertEqual(exit_code, 0)
        self.assertTrue(any("I encountered an error: Simulated provider failure" in out for out in outputs))

    def test_interactive_loop_handles_eof_and_keyboard_interrupt(self) -> None:
        """Verify EOFError and KeyboardInterrupt terminate the loop gracefully."""
        for exc_class in (EOFError, KeyboardInterrupt):
            app = JarvisApplication(
                container=ServiceContainer(),
                config=self.settings,
                logger=self.logger,
                event_bus=EventBus(),
                print_ready=False,
            )

            def mock_input(prompt: str) -> str:
                raise exc_class()

            outputs: list[str] = []
            exit_code = app.run(input_fn=mock_input, output_fn=outputs.append)
            self.assertEqual(exit_code, 0)
            self.assertFalse(app.is_running)

    def test_interactive_loop_empty_inputs_skipped(self) -> None:
        """Verify empty or whitespace-only inputs are ignored without error."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )
        app.command_router.route = MagicMock()

        inputs = ["", "   ", "\t", "exit"]

        def mock_input(prompt: str) -> str:
            return inputs.pop(0)

        outputs: list[str] = []
        exit_code = app.run(input_fn=mock_input, output_fn=outputs.append)

        self.assertEqual(exit_code, 0)
        app.command_router.route.assert_not_called()

    def test_stop_method_terminates_running_loop(self) -> None:
        """Verify stop() method terminates the interactive execution loop."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )

        loop_count = 0

        def mock_input(prompt: str) -> str:
            nonlocal loop_count
            loop_count += 1
            if loop_count >= 2:
                app.stop()
            return "dummy message"

        app.command_router.route = MagicMock(return_value="OK")
        exit_code = app.run(input_fn=mock_input, output_fn=lambda msg: None)

        self.assertEqual(exit_code, 0)
        self.assertFalse(app.is_running)
        self.assertEqual(loop_count, 2)

    def test_format_response_helper(self) -> None:
        """Verify _format_response handles diverse return types gracefully."""
        app = JarvisApplication(
            container=ServiceContainer(),
            config=self.settings,
            logger=self.logger,
            event_bus=EventBus(),
            print_ready=False,
        )

        # None
        self.assertEqual(app._format_response(None), "Command executed successfully.")
        # AIResponse / object with .content
        self.assertEqual(app._format_response(AIResponse(content="Response text")), "Response text")
        # Dict with response
        self.assertEqual(app._format_response({"response": "Dict message"}), "Dict message")
        # Dict with message
        self.assertEqual(app._format_response({"message": "Info message"}), "Info message")
        # Plain string
        self.assertEqual(app._format_response("Plain text response"), "Plain text response")
        # Primitive integer
        self.assertEqual(app._format_response(42), "42")

    def test_main_entry_point_launches_application(self) -> None:
        """Verify main.py launches JarvisApplication and exits cleanly."""
        mock_app = MagicMock(spec=JarvisApplication)
        mock_app.run.return_value = 0

        with patch("main.bootstrap", return_value=True):
            exit_code = main(app_instance=mock_app, interactive=True)

        self.assertEqual(exit_code, 0)
        mock_app.run.assert_called_once()

    def test_main_entry_point_bootstrap_failure_returns_one(self) -> None:
        """Verify main() returns 1 if bootstrap() returns failure."""
        with patch("main.bootstrap", return_value=False):
            exit_code = main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
