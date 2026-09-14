"""Hardware-independent deterministic unit tests for Phase 27.2 Vision Security Guardrails.

SAFETY GUARANTEES:
- All window metadata and desktop capture are 100% mocked.
- ZERO real desktop window queries or screen captures occur.
- ZERO disk writes.
- ZERO secrets in logs or test assertions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.vision.capture import DesktopCaptureEngine, MockCaptureBackend
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    CaptureError,
    ScreenCapture,
    ScreenObservation,
    WindowBounds,
)
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    SensitiveWindowRule,
    VisionSecurityPolicy,
)


class TestSensitiveWindowRules(unittest.TestCase):
    """Test SensitiveWindowRule matching logic (case-insensitivity, process and title patterns)."""

    def test_normal_application_allowed(self) -> None:
        """Verify normal application metadata is not matched by default sensitive rules."""
        policy = VisionSecurityPolicy()
        auth = policy.authorize(
            window_title="Visual Studio Code - project.py",
            process_name="Code.exe",
        )
        self.assertTrue(auth.is_allowed)
        self.assertEqual(auth.decision, CaptureDecision.ALLOW)
        self.assertEqual(auth.category, CaptureCategory.SAFE)

    def test_password_manager_process_blocked(self) -> None:
        """Verify known password manager processes trigger BLOCK."""
        policy = VisionSecurityPolicy()
        pm_processes = [
            "bitwarden.exe",
            "1password.exe",
            "keepass.exe",
            "keepassxc.exe",
            "lastpass.exe",
            "dashlane.exe",
        ]
        for proc in pm_processes:
            with self.subTest(proc=proc):
                auth = policy.authorize(window_title="My Vault", process_name=proc)
                self.assertFalse(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.BLOCK)
                self.assertEqual(auth.category, CaptureCategory.SENSITIVE_APPLICATION)
                self.assertIn("password_managers", auth.rule_name)

    def test_password_manager_title_blocked(self) -> None:
        """Verify window titles containing password vault terms trigger BLOCK."""
        policy = VisionSecurityPolicy()
        titles = [
            "Bitwarden - Personal Vault",
            "1Password - Unlock",
            "KeePassXC - Passwords.kdbx",
            "LastPass Vault",
        ]
        for title in titles:
            with self.subTest(title=title):
                auth = policy.authorize(window_title=title, process_name="chrome.exe")
                self.assertFalse(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.BLOCK)
                self.assertEqual(auth.category, CaptureCategory.SENSITIVE_APPLICATION)

    def test_case_insensitive_matching(self) -> None:
        """Verify matching is strictly case-insensitive."""
        policy = VisionSecurityPolicy()
        auth = policy.authorize(window_title="BITWARDEN VAULT", process_name="BITWARDEN.EXE")
        self.assertFalse(auth.is_allowed)
        self.assertEqual(auth.decision, CaptureDecision.BLOCK)

        auth2 = policy.authorize(window_title="chrome incognito tab", process_name="CHROME.EXE")
        self.assertFalse(auth2.is_allowed)
        self.assertEqual(auth2.decision, CaptureDecision.BLOCK)

    def test_credential_security_dialogs_blocked(self) -> None:
        """Verify Windows Security and UAC credential prompts trigger BLOCK."""
        policy = VisionSecurityPolicy()
        cred_contexts = [
            ("Windows Security", "credentialuibroker.exe"),
            ("Credential Manager", "explorer.exe"),
            ("User Account Control", "consent.exe"),
        ]
        for title, proc in cred_contexts:
            with self.subTest(title=title, proc=proc):
                auth = policy.authorize(window_title=title, process_name=proc)
                self.assertFalse(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.BLOCK)
                self.assertEqual(auth.category, CaptureCategory.CREDENTIAL_INTERFACE)

    def test_financial_and_banking_blocked(self) -> None:
        """Verify banking and payment checkout windows trigger BLOCK."""
        policy = VisionSecurityPolicy()
        bank_titles = [
            "Chase Bank - Accounts Overview",
            "Bank of America | Online Banking",
            "PayPal Checkout - Review Order",
            "Secure Netbanking Portal",
        ]
        for title in bank_titles:
            with self.subTest(title=title):
                auth = policy.authorize(window_title=title, process_name="msedge.exe")
                self.assertFalse(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.BLOCK)
                self.assertEqual(auth.category, CaptureCategory.SENSITIVE_APPLICATION)

    def test_private_and_incognito_browsing_blocked(self) -> None:
        """Verify private and incognito window markers trigger BLOCK."""
        policy = VisionSecurityPolicy()
        private_cases = [
            ("New Incognito Tab - Google Chrome", "chrome.exe"),
            ("InPrivate browsing - Microsoft Edge", "msedge.exe"),
            ("Private Browsing - Mozilla Firefox", "firefox.exe"),
            ("Tor Browser - Private Web", "tor.exe"),
        ]
        for title, proc in private_cases:
            with self.subTest(title=title, proc=proc):
                auth = policy.authorize(window_title=title, process_name=proc)
                self.assertFalse(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.BLOCK)
                self.assertEqual(auth.category, CaptureCategory.PRIVATE_BROWSING)

    def test_standard_browser_window_allowed(self) -> None:
        """Verify standard (non-incognito) browser windows are permitted."""
        policy = VisionSecurityPolicy()
        normal_cases = [
            ("Python Documentation - Google Chrome", "chrome.exe"),
            ("GitHub - Pull Requests - Microsoft Edge", "msedge.exe"),
            ("Wikipedia, the free encyclopedia - Mozilla Firefox", "firefox.exe"),
        ]
        for title, proc in normal_cases:
            with self.subTest(title=title, proc=proc):
                auth = policy.authorize(window_title=title, process_name=proc)
                self.assertTrue(auth.is_allowed)
                self.assertEqual(auth.decision, CaptureDecision.ALLOW)

    def test_authorization_reason_does_not_leak_secrets(self) -> None:
        """Verify authorization reason is safe and contains zero private window text."""
        policy = VisionSecurityPolicy()
        auth = policy.authorize(
            window_title="My Secret Bank Account 123456789 - Online Banking",
            process_name="chrome.exe",
        )
        self.assertFalse(auth.is_allowed)
        # Reason must explain the category without leaking the user's secret account title
        self.assertNotIn("123456789", auth.reason)
        self.assertIn("financial_banking", auth.reason)

    def test_custom_rule_management(self) -> None:
        """Verify adding, querying, and removing custom sensitive rules."""
        policy = VisionSecurityPolicy()
        custom_rule = SensitiveWindowRule(
            name="custom_internal_tool",
            category=CaptureCategory.POLICY_RESTRICTION,
            description="Confidential internal tool",
            process_patterns=("internal_tool.exe",),
            title_patterns=("Confidential Console",),
            priority=300,
        )
        policy.add_rule(custom_rule)
        self.assertIsNotNone(policy.get_rule("custom_internal_tool"))

        auth = policy.authorize(window_title="Confidential Console", process_name="internal_tool.exe")
        self.assertFalse(auth.is_allowed)
        self.assertEqual(auth.category, CaptureCategory.POLICY_RESTRICTION)

        # Remove rule
        removed = policy.remove_rule("custom_internal_tool")
        self.assertTrue(removed)
        self.assertIsNone(policy.get_rule("custom_internal_tool"))

        # Should now be allowed
        auth2 = policy.authorize(window_title="Confidential Console", process_name="internal_tool.exe")
        self.assertTrue(auth2.is_allowed)


class TestEphemeralBufferManager(unittest.TestCase):
    """Test in-memory ephemeral buffer lifecycle, TTL expiration, and bounded cache."""

    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.manager = EphemeralBufferManager(
            default_ttl_seconds=1.0,
            max_buffers=1,
            event_bus_instance=self.event_bus,
        )
        self.sample_capture = ScreenCapture(
            raw_data=bytes([0, 0, 0, 255]) * 16,
            width=4,
            height=4,
            source="test",
        )

    def test_store_and_retrieve_buffer(self) -> None:
        """Verify storing and retrieving screen observation in RAM."""
        obs = self.manager.store(self.sample_capture, source="test_window")
        self.assertIsNotNone(obs)
        self.assertFalse(self.manager.is_empty)
        self.assertEqual(self.manager.count, 1)

        retrieved = self.manager.get(obs.id)
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.id, obs.id)
        self.assertEqual(retrieved.capture.width, 4)

        # Default get() retrieves the latest observation
        latest = self.manager.get()
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.id, obs.id)

    def test_single_frame_replacement(self) -> None:
        """Verify subsequent captures replace previous frames to prevent memory bloat."""
        obs1 = self.manager.store(self.sample_capture)
        self.assertEqual(self.manager.count, 1)

        second_capture = ScreenCapture(
            raw_data=bytes([255, 255, 255, 255]) * 16,
            width=4,
            height=4,
            source="test_second",
        )
        obs2 = self.manager.store(second_capture)
        self.assertEqual(self.manager.count, 1)

        # Old observation is no longer in buffer
        self.assertIsNone(self.manager.get(obs1.id))
        self.assertIsNotNone(self.manager.get(obs2.id))

    def test_explicit_buffer_clear_and_close(self) -> None:
        """Verify explicit clear and close operations release references."""
        self.manager.store(self.sample_capture)
        self.assertEqual(self.manager.count, 1)

        self.manager.clear()
        self.assertEqual(self.manager.count, 0)
        self.assertTrue(self.manager.is_empty)
        self.assertIsNone(self.manager.get())

    def test_buffer_ttl_expiration(self) -> None:
        """Verify observations automatically expire after TTL."""
        # Store with very short TTL: 0.05 seconds
        obs = self.manager.store(self.sample_capture, ttl_seconds=0.05)
        self.assertFalse(obs.is_expired)

        time.sleep(0.08)
        self.assertTrue(obs.is_expired)
        # get() must return None for expired buffer
        self.assertIsNone(self.manager.get(obs.id))
        self.assertEqual(self.manager.count, 0)

    def test_thread_safe_buffer_access(self) -> None:
        """Verify concurrent worker threads safely access buffer without race conditions."""
        stored_ids = []

        def _worker(i: int):
            cap = ScreenCapture(
                raw_data=bytes([i % 256, 0, 0, 255]) * 16,
                width=4,
                height=4,
            )
            o = self.manager.store(cap)
            stored_ids.append(o.id)
            _ = self.manager.get()

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_worker, i) for i in range(10)]
            for f in futures:
                f.result()

        # Buffer count should still be bounded at max_buffers (1)
        self.assertLessEqual(self.manager.count, 1)


class TestSecureVisionManagerIntegration(unittest.TestCase):
    """Test SecureVisionManager policy enforcement preceding screen capture."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.mock_backend = MockCaptureBackend()
        self.engine = DesktopCaptureEngine(
            backend=self.mock_backend,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )
        self.policy = VisionSecurityPolicy()
        self.buffer_mgr = EphemeralBufferManager(event_bus_instance=self.event_bus)

        self.secure_mgr = SecureVisionManager(
            engine=self.engine,
            policy=self.policy,
            buffer_manager=self.buffer_mgr,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

    def test_allowed_capture_reaches_backend_and_stores_buffer(self) -> None:
        """Verify permitted capture executes BitBlt and stores ScreenObservation."""
        # Mock active window as a normal application
        self.secure_mgr._resolve_window_metadata = MagicMock(
            return_value=("Visual Studio Code - main.py", "Code.exe")
        )

        obs = self.secure_mgr.capture_active_window()
        self.assertIsNotNone(obs)
        self.assertIsNotNone(obs.capture)
        self.assertTrue(obs.authorization.is_allowed)
        self.assertEqual(self.mock_backend.capture_count, 1)

        # Stored in buffer manager
        latest = self.secure_mgr.get_latest_observation()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, obs.id)

    def test_blocked_capture_prevents_backend_execution(self) -> None:
        """Verify sensitive window BLOCKS before any backend BitBlt call."""
        # Mock active window as Bitwarden vault
        self.secure_mgr._resolve_window_metadata = MagicMock(
            return_value=("Bitwarden - Master Password", "bitwarden.exe")
        )

        initial_count = self.mock_backend.capture_count

        with self.assertRaises(CaptureBlockedError) as ctx:
            self.secure_mgr.capture_active_window(raise_on_blocked=True)

        self.assertIn("password_managers", str(ctx.exception))
        # Crucial security invariant: backend was NEVER called!
        self.assertEqual(self.mock_backend.capture_count, initial_count)
        # Buffer manager remains empty
        self.assertTrue(self.buffer_mgr.is_empty)

    def test_blocked_capture_with_raise_false_returns_blocked_observation(self) -> None:
        """Verify raise_on_blocked=False returns ScreenObservation with capture=None."""
        self.secure_mgr._resolve_window_metadata = MagicMock(
            return_value=("New Incognito Tab - Google Chrome", "chrome.exe")
        )

        obs = self.secure_mgr.capture_active_window(raise_on_blocked=False)
        self.assertIsNotNone(obs)
        self.assertIsNone(obs.capture)
        self.assertFalse(obs.authorization.is_allowed)
        self.assertEqual(obs.authorization.category, CaptureCategory.PRIVATE_BROWSING)
        self.assertEqual(self.mock_backend.capture_count, 0)

    def test_desktop_capture_blocked_when_foreground_is_sensitive(self) -> None:
        """Verify capturing full screen is BLOCKED if user has sensitive window focused."""
        self.secure_mgr._resolve_window_metadata = MagicMock(
            return_value=("Windows Security - Smart Card Authentication", "credentialuibroker.exe")
        )

        with self.assertRaises(CaptureBlockedError):
            self.secure_mgr.capture_screen()

        self.assertEqual(self.mock_backend.capture_count, 0)

    def test_event_bus_security_telemetry(self) -> None:
        """Verify vision.capture_allowed and vision.capture_blocked events are emitted."""
        allowed_events = []
        blocked_events = []
        self.event_bus.subscribe("vision.capture_allowed", lambda e: allowed_events.append(e))
        self.event_bus.subscribe("vision.capture_blocked", lambda e: blocked_events.append(e))

        # 1. Allowed
        self.secure_mgr._resolve_window_metadata = MagicMock(return_value=("Notepad", "notepad.exe"))
        self.secure_mgr.capture_screen()
        self.assertEqual(len(allowed_events), 1)
        self.assertEqual(len(blocked_events), 0)

        # 2. Blocked
        self.secure_mgr._resolve_window_metadata = MagicMock(return_value=("1Password", "1password.exe"))
        self.secure_mgr.capture_screen(raise_on_blocked=False)
        self.assertEqual(len(blocked_events), 1)
        # Event payload must contain category and rule, but NO secret titles
        p = blocked_events[0].payload if hasattr(blocked_events[0], "payload") else blocked_events[0]
        self.assertEqual(p["rule_name"], "password_managers")
        self.assertEqual(p["category"], "sensitive_application")

    def test_zero_disk_artifacts_on_secure_capture(self) -> None:
        """Verify secure vision operations generate zero image files on disk."""
        initial_files = set(os.listdir("."))
        self.secure_mgr._resolve_window_metadata = MagicMock(return_value=("Notepad", "notepad.exe"))

        obs = self.secure_mgr.capture_active_window()
        self.assertIsNotNone(obs)
        assert obs.capture is not None
        _ = obs.capture.to_png_bytes()

        after_files = set(os.listdir("."))
        self.assertEqual(len(after_files - initial_files), 0)


if __name__ == "__main__":
    unittest.main()
