"""Unit tests for the J.A.R.V.I.S Configuration Manager."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import Settings, load_settings
from app.core.constants import (
    DEFAULT_APP_NAME,
    DEFAULT_ENVIRONMENT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_TTS_VOICE_EN,
    DEFAULT_TTS_VOICE_HI,
    DEFAULT_VERSION,
    DEFAULT_WAKE_PHRASES,
    DEFAULT_WAKE_WORD,
)


class TestConfigManager(unittest.TestCase):
    """Test suite for configuration loading and validation."""

    def setUp(self) -> None:
        """Clear relevant env vars before each test."""
        self.env_vars_to_clear = [
            "APP_NAME",
            "VERSION",
            "ENVIRONMENT",
            "DEBUG",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "LANGUAGE",
            "WAKE_WORD",
            "WAKE_PHRASES",
            "TTS_VOICE",
            "LOG_LEVEL",
            "DATA_DIR",
            "MODELS_DIR",
            "LOGS_DIR",
            "CONFIG_DIR",
        ]
        self._saved_env = {
            k: os.environ.get(k) for k in self.env_vars_to_clear
        }
        for k in self.env_vars_to_clear:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        """Restore environment after each test."""
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_default_values(self) -> None:
        """Verify fallback to defaults when no environment variables exist."""
        test_settings = load_settings(env_file=None, validate_on_load=True)

        self.assertEqual(test_settings.app_name, DEFAULT_APP_NAME)
        self.assertEqual(test_settings.version, DEFAULT_VERSION)
        self.assertEqual(test_settings.environment, DEFAULT_ENVIRONMENT)
        self.assertFalse(test_settings.debug)
        self.assertEqual(test_settings.gemini_api_key, "")
        self.assertEqual(test_settings.gemini_model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(test_settings.openrouter_api_key, "")
        self.assertEqual(
            test_settings.openrouter_model, DEFAULT_OPENROUTER_MODEL
        )
        self.assertEqual(test_settings.language, "en")
        self.assertEqual(test_settings.wake_word, DEFAULT_WAKE_WORD)
        self.assertEqual(test_settings.wake_phrases, DEFAULT_WAKE_PHRASES)
        self.assertEqual(test_settings.tts_voice, DEFAULT_TTS_VOICE_EN)
        self.assertEqual(test_settings.log_level, DEFAULT_LOG_LEVEL)

    def test_custom_environment_overrides(self) -> None:
        """Verify environment variables override defaults properly."""
        with patch.dict(
            os.environ,
            {
                "APP_NAME": "CustomJARVIS",
                "VERSION": "2.0.0",
                "ENVIRONMENT": "testing",
                "DEBUG": "true",
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL": "gemini-1.5-flash",
                "OPENROUTER_API_KEY": "test_openrouter_key",
                "OPENROUTER_MODEL": "meta-llama/llama-3-8b",
                "LANGUAGE": "hi",
                "WAKE_WORD": "hey jarvis",
                "WAKE_PHRASES": "Computer, Hello Jarvis, Wake Up",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            test_settings = load_settings(env_file=None, validate_on_load=True)

            self.assertEqual(test_settings.app_name, "CustomJARVIS")
            self.assertEqual(test_settings.version, "2.0.0")
            self.assertEqual(test_settings.environment, "testing")
            self.assertTrue(test_settings.debug)
            self.assertEqual(test_settings.gemini_api_key, "test_gemini_key")
            self.assertEqual(test_settings.gemini_model, "gemini-1.5-flash")
            self.assertEqual(
                test_settings.openrouter_api_key, "test_openrouter_key"
            )
            self.assertEqual(
                test_settings.openrouter_model, "meta-llama/llama-3-8b"
            )
            self.assertEqual(test_settings.language, "hi")
            self.assertEqual(test_settings.wake_word, "hey jarvis")
            self.assertEqual(
                test_settings.wake_phrases, ("Computer", "Hello Jarvis", "Wake Up")
            )
            self.assertEqual(test_settings.tts_voice, DEFAULT_TTS_VOICE_HI)
            self.assertEqual(test_settings.log_level, "DEBUG")

    def test_validation_invalid_environment(self) -> None:
        """Verify invalid ENVIRONMENT raises ValueError."""
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"}):
            with self.assertRaises(ValueError) as ctx:
                load_settings(env_file=None, validate_on_load=True)
            self.assertIn("Invalid ENVIRONMENT", str(ctx.exception))

    def test_validation_invalid_log_level(self) -> None:
        """Verify invalid LOG_LEVEL raises ValueError."""
        with patch.dict(os.environ, {"LOG_LEVEL": "VERBOSE"}):
            with self.assertRaises(ValueError) as ctx:
                load_settings(env_file=None, validate_on_load=True)
            self.assertIn("Invalid LOG_LEVEL", str(ctx.exception))

    def test_validation_invalid_language(self) -> None:
        """Verify invalid LANGUAGE raises ValueError."""
        with patch.dict(os.environ, {"LANGUAGE": "french"}):
            with self.assertRaises(ValueError) as ctx:
                load_settings(env_file=None, validate_on_load=True)
            self.assertIn("Invalid LANGUAGE", str(ctx.exception))

    def test_immutability(self) -> None:
        """Verify Settings instance is frozen and cannot be mutated."""
        test_settings = load_settings(env_file=None)
        with self.assertRaises((AttributeError, TypeError)):
            test_settings.app.app_name = "NewName"  # type: ignore[misc]

    def test_paths_configuration(self) -> None:
        """Verify paths are correctly resolved as Path instances."""
        test_settings = load_settings(env_file=None)
        self.assertIsInstance(test_settings.data_dir, Path)
        self.assertIsInstance(test_settings.models_dir, Path)
        self.assertIsInstance(test_settings.logs_dir, Path)
        self.assertIsInstance(test_settings.config_dir, Path)
        self.assertTrue(test_settings.data_dir.is_absolute())


if __name__ == "__main__":
    unittest.main()
