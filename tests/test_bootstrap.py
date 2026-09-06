"""Unit tests for the J.A.R.V.I.S Bootstrap Subsystem."""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.bootstrap import (
    BANNER_ART,
    BootstrapResult,
    _display_banner,
    _display_failure_message,
    _verify_configuration,
    _verify_directories,
    _verify_logger,
    bootstrap,
    get_banner,
)
from app.core.config import Settings, load_settings
from main import main


class TestBootstrap(unittest.TestCase):
    """Comprehensive test suite for application bootstrap sequence."""

    def setUp(self) -> None:
        """Set up an isolated temporary workspace for bootstrap tests."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.models_dir = self.test_dir / "models"
        self.logs_dir = self.test_dir / "logs"
        self.configs_dir = self.test_dir / "configs"

        # Preserve environment variables
        self.env_keys = [
            "APP_NAME",
            "VERSION",
            "ENVIRONMENT",
            "DEBUG",
            "LOG_LEVEL",
            "DATA_DIR",
            "MODELS_DIR",
            "LOGS_DIR",
            "CONFIG_DIR",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        self.saved_env = {k: os.environ.get(k) for k in self.env_keys}

        # Point environment paths to temporary directory
        os.environ["DATA_DIR"] = str(self.data_dir)
        os.environ["MODELS_DIR"] = str(self.models_dir)
        os.environ["LOGS_DIR"] = str(self.logs_dir)
        os.environ["CONFIG_DIR"] = str(self.configs_dir)
        os.environ["ENVIRONMENT"] = "development"

    def tearDown(self) -> None:
        """Clean up logging handlers, temporary directories, and restore environment."""
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            handler.close()
            root_logger.removeHandler(handler)

        # Restore environment
        for k, v in self.saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_bootstrap_result_properties(self) -> None:
        """Verify BootstrapResult truthiness and comparison semantics."""
        success_result = BootstrapResult(success=True)
        failure_result = BootstrapResult(
            success=False, error_message="Initialization failed"
        )

        # Boolean evaluation
        self.assertTrue(bool(success_result))
        self.assertFalse(bool(failure_result))

        # Equality with booleans
        self.assertEqual(success_result, True)
        self.assertEqual(failure_result, False)
        self.assertNotEqual(success_result, False)
        self.assertNotEqual(failure_result, True)

    def test_bootstrap_success_execution(self) -> None:
        """Verify successful bootstrap creates directories, initializes loggers, and returns success."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)

        result = bootstrap(
            settings_instance=test_settings,
            exit_on_failure=False,
            display_banner=False,
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.settings)
        self.assertIsNotNone(result.logger)
        self.assertIsNone(result.error_message)

        # Verify directories created
        self.assertTrue(self.data_dir.exists())
        self.assertTrue(self.models_dir.exists())
        self.assertTrue(self.logs_dir.exists())
        self.assertTrue(self.configs_dir.exists())

    def test_banner_content(self) -> None:
        """Verify the banner displays Application Name, Version, and Environment."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)
        banner_text = get_banner(test_settings)

        # 3. Startup banner displayed
        self.assertIn("Just A Rather Very Intelligent System", banner_text)

        # 4. Display: Application Name, Version, Environment
        self.assertIn(f"Application : {test_settings.app_name}", banner_text)
        self.assertIn(f"Version     : {test_settings.version}", banner_text)
        self.assertIn(
            f"Environment : {test_settings.environment}", banner_text
        )
        self.assertIn(f"Log Level   : {test_settings.log_level}", banner_text)

    def test_display_banner_output(self) -> None:
        """Verify banner prints cleanly to stdout."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            _display_banner(test_settings)

        output = captured_stdout.getvalue()
        self.assertIn(test_settings.app_name, output)
        self.assertIn(test_settings.version, output)
        self.assertIn(test_settings.environment, output)

    def test_verify_configuration_valid(self) -> None:
        """Verify valid configuration passes verification without error."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)
        _verify_configuration(test_settings)

    def test_verify_configuration_invalid(self) -> None:
        """Verify invalid configuration raises RuntimeError."""
        # Non-settings instance
        with self.assertRaises(RuntimeError):
            _verify_configuration(cast_obj := None)  # type: ignore[arg-type]

        # Invalid environment
        os.environ["ENVIRONMENT"] = "invalid_env"
        with self.assertRaises(RuntimeError):
            bad_settings = load_settings(
                base_dir=self.test_dir, env_file=None, validate_on_load=False
            )
            _verify_configuration(bad_settings)

    def test_verify_logger(self) -> None:
        """Verify logger validation succeeds when handlers are present."""
        logger = logging.getLogger("SYSTEM")
        logger.addHandler(logging.NullHandler())
        _verify_logger(logger)

        # Non-logger instance fails
        with self.assertRaises(RuntimeError):
            _verify_logger("not_a_logger")  # type: ignore[arg-type]

    def test_verify_directories(self) -> None:
        """Verify directory verification creates and validates all folders."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)
        _verify_directories(test_settings)

        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.models_dir.is_dir())
        self.assertTrue(self.logs_dir.is_dir())
        self.assertTrue(self.configs_dir.is_dir())

    def test_verify_directories_failure_when_file_exists(self) -> None:
        """Verify failure when a directory path exists as a regular file."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)
        # Create a file where data_dir should be
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.write_text("blocking file")

        with self.assertRaises(RuntimeError):
            _verify_directories(test_settings)

    def test_display_failure_message(self) -> None:
        """Verify failure message formatting to stderr."""
        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            _display_failure_message("Simulated critical failure")

        err_output = captured_stderr.getvalue()
        self.assertIn("Startup Initialization Failed", err_output)
        self.assertIn("Simulated critical failure", err_output)
        self.assertIn("Suggested Steps", err_output)

    def test_bootstrap_failure_exits_gracefully(self) -> None:
        """Verify bootstrap logs, prints message, and invokes sys.exit(1) on failure."""
        bad_settings = MagicMock(spec=Settings)
        bad_settings.app_name = "J.A.R.V.I.S"
        bad_settings.version = "0.1.0"
        bad_settings.environment = "development"
        bad_settings.log_level = "INFO"
        bad_settings.log_directory = self.logs_dir
        # Mock validate to raise ValueError
        bad_settings.validate.side_effect = ValueError("Invalid configuration parameter")

        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            with self.assertRaises(SystemExit) as exit_ctx:
                bootstrap(
                    settings_instance=bad_settings,
                    exit_on_failure=True,
                    display_banner=False,
                )

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("Invalid configuration parameter", captured_stderr.getvalue())

    def test_bootstrap_failure_without_exit(self) -> None:
        """Verify bootstrap returns failure result without terminating when exit_on_failure=False."""
        bad_settings = MagicMock(spec=Settings)
        bad_settings.app_name = "J.A.R.V.I.S"
        bad_settings.version = "0.1.0"
        bad_settings.environment = "development"
        bad_settings.log_level = "INFO"
        bad_settings.log_directory = self.logs_dir
        bad_settings.validate.side_effect = ValueError("Corrupt configuration")

        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            result = bootstrap(
                settings_instance=bad_settings,
                exit_on_failure=False,
                display_banner=False,
            )

        self.assertFalse(result.success)
        self.assertFalse(bool(result))
        self.assertIsNotNone(result.error_message)
        self.assertIn("Corrupt configuration", result.error_message or "")


    def test_main_function(self) -> None:
        """Verify main() entry point invokes bootstrap and returns 0 exit code on success."""
        test_settings = load_settings(base_dir=self.test_dir, env_file=None)

        with patch("app.core.bootstrap.load_settings", return_value=test_settings):
            with patch("sys.stdout", io.StringIO()):
                exit_code = main()

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
