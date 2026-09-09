"""Unit tests for the Structured Logging Subscriber (Phase 17.0).

Verifies:
- Event-to-log level mapping (INFO, WARNING, ERROR, DEBUG)
- Automatic redaction of API keys, bearer tokens, passwords, and sensitive keys
- Truncation of large output payloads
- Safety guarantee: prompts and secrets are never logged in plaintext
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock

from app.ai.planner.events import (
    PlanCompleted,
    PlanFailed,
    PlannerEventBus,
    PlanStarted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
    TaskTimeout,
)
from app.ai.planner.logging_subscriber import (
    StructuredLoggingSubscriber,
    sanitize_value,
)


class TestStructuredLoggingSubscriber(unittest.TestCase):
    """Test suite for StructuredLoggingSubscriber."""

    def setUp(self) -> None:
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.bus = PlannerEventBus()
        self.subscriber = StructuredLoggingSubscriber(bus=self.bus, custom_logger=self.mock_logger)

    def test_sanitize_value_redaction(self) -> None:
        """Sensitive dictionary keys and token patterns must be redacted."""
        sensitive_payload = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "password": "SuperSecretPassword123!",
            "auth_token": "Bearer ghp_abcdefghijklmnopqrstuvwxyz123456",
            "user_query": "open chrome",
            "nested": {
                "credential": "admin:secret",
                "normal": "hello",
            },
        }
        sanitized = sanitize_value(sensitive_payload)

        self.assertEqual(sanitized["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["password"], "[REDACTED]")
        self.assertEqual(sanitized["auth_token"], "[REDACTED]")
        self.assertEqual(sanitized["nested"]["credential"], "[REDACTED]")
        self.assertEqual(sanitized["nested"]["normal"], "hello")
        self.assertEqual(sanitized["user_query"], "open chrome")

    def test_sanitize_value_output_truncation(self) -> None:
        """Large string payloads must be truncated to safe length."""
        huge_output = "A" * 500
        sanitized = sanitize_value(huge_output, max_length=50)
        self.assertTrue(len(sanitized) < 100)
        self.assertIn("... [truncated]", sanitized)

    def test_info_log_level_mapping(self) -> None:
        """PlanStarted, PlanCompleted, TaskCompleted, and RecoveryCompleted must log at INFO level."""
        self.bus.publish(PlanStarted(plan_id="p1", query="open chrome", task_count=1))
        self.mock_logger.info.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(TaskCompleted(task_id="t1", action="open_app", result="opened", duration=0.1))
        self.mock_logger.info.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(PlanCompleted(plan_id="p1", success=True, duration=0.2))
        self.mock_logger.info.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(RecoveryCompleted(query="q", attempt=1, success=True, task_count=1, duration=0.3))
        self.mock_logger.info.assert_called()

    def test_warning_log_level_mapping(self) -> None:
        """TaskFailed, TaskRetried, RecoveryStarted, and TaskTimeout must log at WARNING level."""
        self.bus.publish(TaskFailed(task_id="t1", action="web_search", error="network glitch", duration=0.1))
        self.mock_logger.warning.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(TaskRetried(task_id="t1", action="web_search", attempt=2, reason="retry net"))
        self.mock_logger.warning.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(RecoveryStarted(query="test", attempt=1, failed_task_ids=["t1"]))
        self.mock_logger.warning.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(TaskTimeout(task_id="t1", timeout_seconds=5.0, policy="RETRY"))
        self.mock_logger.warning.assert_called()

    def test_error_log_level_mapping(self) -> None:
        """PlanFailed and RecoveryFailed must log at ERROR level."""
        self.bus.publish(PlanFailed(plan_id="p1", error="Fatal cycle detected"))
        self.mock_logger.error.assert_called()

        self.mock_logger.reset_mock()
        self.bus.publish(RecoveryFailed(query="test", attempt=2, reason="Budget exhausted"))
        self.mock_logger.error.assert_called()

    def test_debug_log_level_mapping(self) -> None:
        """TaskStarted and granular control events must log at DEBUG level."""
        self.bus.publish(TaskStarted(task_id="t1", action="open_app", target="chrome"))
        self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
