"""Unit tests for the J.A.R.V.I.S Logging Subsystem."""

import logging
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from app.core import logger as jarvis_logger
from app.core.logger import get_logger


class TestLogger(unittest.TestCase):
    """Test suite for the logging system."""

    def setUp(self) -> None:
        """Create a temporary directory for test logs and reset logger state."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.logs_dir = self.test_dir / "test_logs"

        # Reset logger module state
        jarvis_logger._INITIALIZED = False
        jarvis_logger._setup_root_logging(
            log_dir=self.logs_dir,
            log_level_name="DEBUG",
        )

    def tearDown(self) -> None:
        """Clean up test handlers and temporary directory."""
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            handler.close()
            root_logger.removeHandler(handler)

        jarvis_logger._INITIALIZED = False
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_logs_directory_auto_created(self) -> None:
        """Verify the log directory is automatically created."""
        self.assertTrue(self.logs_dir.exists())
        self.assertTrue(self.logs_dir.is_dir())

    def test_get_logger_returns_named_logger(self) -> None:
        """Verify get_logger returns logger with requested name."""
        voice_logger = get_logger("VOICE")
        self.assertEqual(voice_logger.name, "VOICE")
        self.assertIsInstance(voice_logger, logging.Logger)

    def test_log_file_created_and_written(self) -> None:
        """Verify messages are written to the daily log file with expected format."""
        test_logger = get_logger("VOICE")
        test_message = "Listening for wake word..."
        test_logger.info(test_message)

        # Flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = self.logs_dir / "jarvis.log"
        self.assertTrue(log_file.exists())

        content = log_file.read_text(encoding="utf-8")
        self.assertIn("| INFO | VOICE | Listening for wake word...", content)

    def test_all_log_levels(self) -> None:
        """Verify DEBUG, INFO, WARNING, ERROR, and CRITICAL levels are recorded."""
        test_logger = get_logger("BRAIN")

        test_logger.debug("Debug event")
        test_logger.info("Info event")
        test_logger.warning("Warning event")
        test_logger.error("Error event")
        test_logger.critical("Critical event")

        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = self.logs_dir / "jarvis.log"
        content = log_file.read_text(encoding="utf-8")

        self.assertIn("| DEBUG | BRAIN | Debug event", content)
        self.assertIn("| INFO | BRAIN | Info event", content)
        self.assertIn("| WARNING | BRAIN | Warning event", content)
        self.assertIn("| ERROR | BRAIN | Error event", content)
        self.assertIn("| CRITICAL | BRAIN | Critical event", content)

    def test_utf8_multilingual_logging(self) -> None:
        """Verify Hindi and multilingual messages are properly encoded."""
        test_logger = get_logger("VOICE_HI")
        hindi_message = "नमस्ते जार्विस, क्या हाल है?"
        test_logger.info(hindi_message)

        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = self.logs_dir / "jarvis.log"
        content = log_file.read_text(encoding="utf-8")
        self.assertIn(hindi_message, content)

    def test_thread_safety(self) -> None:
        """Verify multiple threads logging concurrently do not corrupt output."""
        thread_count = 10
        messages_per_thread = 20

        def worker(thread_id: int) -> None:
            t_logger = get_logger(f"THREAD_{thread_id}")
            for msg_id in range(messages_per_thread):
                t_logger.info(f"Message {msg_id} from thread {thread_id}")

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = self.logs_dir / "jarvis.log"
        lines = [
            line
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_total_messages = thread_count * messages_per_thread
        self.assertEqual(len(lines), expected_total_messages)


if __name__ == "__main__":
    unittest.main()
